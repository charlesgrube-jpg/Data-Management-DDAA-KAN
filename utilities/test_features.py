
import numpy as np
import torch
import soundfile as sf
import os
import sys
from pathlib import Path

# Add parent path
sys.path.append(str(Path(__file__).parent.parent))

from pipeline.features.cqt_extractor import CQTExtractor
from pipeline.features.lfcc_extractor import LFCCExtractor
from pipeline.config import load_config

def test_features():
    # Load config
    config = load_config("config.yaml")
    
    # Init extractors
    print("Initializing extractors...")
    cqt = CQTExtractor(
        n_bins=config.features.cqt.n_bins,
        hop_length=config.features.cqt.hop_length,
        fmin=config.features.cqt.fmin,
        device="cpu"
    )
    
    lfcc = LFCCExtractor(
        n_lfcc=config.features.lfcc.n_lfcc,
        n_filters=config.features.lfcc.n_filters,
        n_fft=config.features.lfcc.n_fft,
        hop_length=config.features.lfcc.hop_length,
        device="cpu"
    )
    
    # Generate dummy audio
    sr = 16000
    duration = 4.0
    t = np.linspace(0, duration, int(sr*duration))
    # Sweep
    audio = np.sin(2 * np.pi * 440 * t + 100 * t**2)
    # Cast to float32
    audio = audio.astype(np.float32)
    
    print(f"Audio shape: {audio.shape}")
    
    # Test CQT
    try:
        cqt_feat = cqt.extract(audio)
        print(f"✅ CQT Success. Shape: {cqt_feat.shape}") # Expect [1, n_bins, time]
    except Exception as e:
        print(f"❌ CQT Failed: {e}")
        import traceback
        traceback.print_exc()

    # Test LFCC
    try:
        lfcc_feat = lfcc.extract(audio)
        print(f"✅ LFCC Success. Shape: {lfcc_feat.shape}") # Expect [1, n_lfcc, time]
    except Exception as e:
        print(f"❌ LFCC Failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_features()
