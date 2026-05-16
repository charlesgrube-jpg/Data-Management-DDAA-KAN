#!/usr/bin/env python3
"""
Run Imperceptible Attack Locally
"""

import os
import sys
import argparse
import torch
import torchaudio
import torch.nn as nn
import numpy as np
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

# Attempt to import attack (must be in pipeline/attacks/ or root)
try:
    from pipeline.attacks.imperceptible_attack import ImperceptibleAttack
except ImportError:
    try:
        from imperceptible_attack import ImperceptibleAttack
    except ImportError:
        print("Error: Could not import ImperceptibleAttack. check pipeline/attacks/")
        sys.exit(1)

def load_detector(model_name, checkpoint_path, device):
    """
    Load the detector model from checkpoint.
    Tries to import from root files first, then checks inline definitions.
    """
    model = None
    
    # 1. Try importing from specific files (if synced)
    try:
        if model_name == "pinn":
            from pinn_detector import PINNDetector
            model = PINNDetector()
        elif model_name == "kan":
            from kan_detector import KANDetector
            model = KANDetector()
        elif model_name == "transformer":
            from transformer_detector import TransformerDetector
            model = TransformerDetector()
        elif model_name == "neural_ode":
            from neural_ode_detector import NeuralODEDetector
            model = NeuralODEDetector()
    except ImportError as e:
        print(f"[Warn] Could not import from {model_name}_detector.py: {e}")
        print("[Info] Attempting fallback to scripts.train_detector_csv definitions...")

    # 2. Fallback: Import from train_detector_csv if not found
    if model is None:
        try:
            from scripts.train_detector_csv import (
                TransformerDetector, KANDetector, NeuralODEDetector, PINNDetector
            )
            if model_name == "pinn": model = PINNDetector()
            elif model_name == "kan": model = KANDetector()
            elif model_name == "transformer": model = TransformerDetector()
            elif model_name == "neural_ode": model = NeuralODEDetector()
        except ImportError as e:
            print(f"[Error] Could not import model classes from scripts: {e}")
            sys.exit(1)

    if model is None:
        print(f"[Error] Unknown model name: {model_name}")
        sys.exit(1)

    # Load weights
    print(f"[Loader] Loading weights from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Handle different checkpoint formats
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint
        
    # Remove 'module.' prefix if saved from DataParallel
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('module.'):
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v
            
    model.load_state_dict(new_state_dict)
    model.to(device)
    model.eval()
    return model

def load_audio(audio_path, target_sr=16000, max_duration=4.0):
    """Load and preprocess audio."""
    if not os.path.exists(audio_path):
        # Generate dummy audio if file doesn't exist (for testing)
        print(f"[Warn] Audio file {audio_path} not found. Generating white noise.")
        return torch.randn(1, int(target_sr * max_duration)), target_sr

    waveform, sr = torchaudio.load(audio_path)
    
    # Resample
    if sr != target_sr:
        resampler = torchaudio.transforms.Resample(sr, target_sr)
        waveform = resampler(waveform)
        
    # Mix to mono
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
        
    # Trim/Pad
    target_samples = int(target_sr * max_duration)
    if waveform.shape[1] > target_samples:
        waveform = waveform[:, :target_samples]
    elif waveform.shape[1] < target_samples:
        padding = target_samples - waveform.shape[1]
        waveform = torch.nn.functional.pad(waveform, (0, padding))
        
    return waveform, target_sr

def main():
    parser = argparse.ArgumentParser(description="Run Adversarial Attack")
    parser.add_argument("--model", type=str, required=True, choices=["pinn", "kan", "transformer", "neural_ode"])
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--audio", type=str, default="test_audio.wav")
    parser.add_argument("--target", type=int, default=1, help="Target label (1=fake, 0=real)")
    parser.add_argument("--epsilon", type=float, default=0.01)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--output", type=str, default="adv_example.wav")
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # 1. Load Model
    model = load_detector(args.model, args.checkpoint, device)
    print("[Success] Model loaded.")
    
    # 2. Load Audio
    # Check if we have a feature extractor (simple valid CQT fallback)
    # The attack script implies we might need one if the model expects features.
    # Our models in train_detector_csv.py mostly imply features or raw.
    # BaseDetector (Transformer/KAN/PINN) typically take features (84x100).
    
    print("Preparing Feature Extractor...")
    try:
        from pipeline.features.cqt_extractor import CQTExtractor
        feature_extractor = CQTExtractor(sample_rate=16000, device=str(device)).extract
    except ImportError:
        print("[Info] Using fallback CQT extractor")
        def simple_cqt(audio_np):
            # audio_np is (samples,)
            with torch.no_grad():
                res = torchaudio.transforms.MelSpectrogram(
                    sample_rate=16000, n_mels=84
                )(torch.tensor(audio_np)).numpy()
                res = np.log(res + 1e-9)
            return res
        feature_extractor = simple_cqt

    audio, sr = load_audio(args.audio)
    audio = audio.to(device)
    
    # 3. Setup Attack
    print(f"Starting Attack (Target Label: {args.target})...")
    attacker = ImperceptibleAttack(
        epsilon=args.epsilon,
        max_iterations=args.iters,
        device=str(device)
    )
    
    # 4. Run
    # Warning: The attack script expects `audio` as input, but `model` typically expects features.
    # The `ImperceptibleAttack.attack` method handles `feature_extractor` which bridges this.
    # It perturbs AUDIO, extracts FEATURES, feeds to MODEL.
    
    adv_audio_delta, info = attacker.attack(
        model=model,
        audio=audio.squeeze(0), # Attack expects 1D or Batch? Script says "clean audio waveform (samples,) or (batch, samples)"
        target_label=args.target,
        feature_extractor=feature_extractor,
        verbose=True
    )
    
    # 5. Save
    adv_audio = audio + adv_audio_delta
    adv_audio = adv_audio.cpu()
    # Normalize to prevent clipping on save
    adv_audio = torch.clamp(adv_audio, -1, 1)
    
    torchaudio.save(args.output, adv_audio, sr)
    print(f"\n[Done] Adversarial example saved to {args.output}")
    print(f"Success Rate: {info['success_rate']}")
    print(f"Final SNR: {info['snr']:.2f} dB")

if __name__ == "__main__":
    main()
