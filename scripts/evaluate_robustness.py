#!/usr/bin/env python3
"""
Evaluate Adversarial Robustness of Deepfake Detectors

Usage:
    python scripts/evaluate_robustness.py --model pinn --checkpoint "Best Models/pinn_best.pt" --attack fgsm
    python scripts/evaluate_robustness.py --model pinn --checkpoint "Best Models/pinn_best.pt" --attack pgd --num_samples 1000
    python scripts/evaluate_robustness.py --model pinn --checkpoint "Best Models/pinn_best.pt" --attack cleverhans --num_samples 500
"""

import argparse
import csv
import sys
import json
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import torchaudio

# ─── Path Setup ──────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from pipeline.attacks.cleverhans.cleverhans_attack_pytorch import CleverHansAttack
from pipeline.attacks.cleverhans.evaluation import evaluate_adversarial_robustness_cleverhans
from pipeline.attacks.standard_attacks import FGSMAttack, PGDAttack

# Import all four detector classes from their respective training scripts
try:
    from train_pinn import PINNDetector
except ImportError:
    PINNDetector = None

try:
    from train_kan import KANDetector
except ImportError:
    KANDetector = None

try:
    from train_ode import ODEDetector
except ImportError:
    ODEDetector = None


# ─── Dataset ─────────────────────────────────────────────────────────────────

class CSVAudioDataset(torch.utils.data.Dataset):
    """Raw-audio dataset reading from CSV."""
    SAMPLE_RATE = 16000
    MAX_LENGTH = 16000 * 5  # 5 seconds

    def __init__(self, csv_path: Path, max_samples: int = None):
        self.samples = []
        project_root = Path(__file__).resolve().parent.parent

        with open(csv_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                p = row.get('full_path', row.get('filename', '')).strip()
                label = 1 if row.get('label', 'real') in ['synthetic', '1'] else 0
                for candidate in [
                    Path(p),                           # absolute or cwd-relative
                    project_root / p,                  # relative to project root
                    project_root / "official_dataset" / p,   # cluster data location
                    project_root / "completed_correct_dataset" / p,
                    csv_path.parent / p                # relative to CSV dir
                ]:
                    if candidate.exists():
                        self.samples.append((str(candidate), label))
                        break

        import random
        random.seed(42)  # Deterministic shuffle for reproducibility
        random.shuffle(self.samples)
        if max_samples:
            self.samples = self.samples[:max_samples]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        try:
            wav, sr = torchaudio.load(path)
            if wav.shape[0] > 1:
                wav = wav.mean(0, keepdim=True)
            wav = wav.squeeze(0)
            if sr != self.SAMPLE_RATE:
                wav = torchaudio.transforms.Resample(sr, self.SAMPLE_RATE)(wav.unsqueeze(0)).squeeze(0)
            if wav.shape[0] > self.MAX_LENGTH:
                wav = wav[:self.MAX_LENGTH]
            elif wav.shape[0] < self.MAX_LENGTH:
                wav = F.pad(wav, (0, self.MAX_LENGTH - wav.shape[0]))
            wav = torch.nan_to_num(wav / (wav.abs().max() + 1e-6))
            return wav, label
        except Exception:
            return torch.zeros(self.MAX_LENGTH), label


# ─── Helpers ─────────────────────────────────────────────────────────────────

def compute_snr(clean: torch.Tensor, adv: torch.Tensor) -> float:
    signal_power = (clean ** 2).mean()
    noise_power = ((adv - clean) ** 2).mean()
    return (10 * torch.log10(signal_power / (noise_power + 1e-10))).item()


def evaluate_standard_attack(model, dataloader, attack, device, num_samples=None):
    """Evaluation loop for FGSM and PGD."""
    results = {
        'total_samples': 0, 'successful_attacks': 0, 'snr_values': [],
        'clean_correct': 0, 'adversarial_correct': 0
    }
    model.eval()

    for i, (audio, label) in enumerate(tqdm(dataloader, desc="Evaluating")):
        if num_samples and i * audio.shape[0] >= num_samples:
            break

        audio = audio.to(device)
        label = label.to(device)

        with torch.no_grad():
            clean_pred = model(audio).argmax(dim=-1)
        results['clean_correct'] += (clean_pred == label).sum().item()

        delta = attack.attack(audio, label)
        adv_audio = torch.clamp(audio + delta, -1.0, 1.0)

        with torch.no_grad():
            adv_pred = model(adv_audio).argmax(dim=-1)

        results['successful_attacks'] += (adv_pred != label).sum().item()
        results['adversarial_correct'] += (adv_pred == label).sum().item()
        results['snr_values'].append(compute_snr(audio, adv_audio))
        results['total_samples'] += audio.shape[0]

    n = max(results['total_samples'], 1)
    results['attack_success_rate'] = results['successful_attacks'] / n
    results['clean_accuracy'] = results['clean_correct'] / n
    results['adversarial_accuracy'] = results['adversarial_correct'] / n
    results['mean_snr'] = float(np.mean(results['snr_values'])) if results['snr_values'] else 0.0
    results['std_snr'] = float(np.std(results['snr_values'])) if results['snr_values'] else 0.0
    return results


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Evaluate adversarial robustness")
    parser.add_argument("--model",      type=str, required=True,
                        choices=["transformer", "kan", "neural_ode", "pinn"])
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--attack",     type=str, default="cleverhans",
                        choices=["fgsm", "pgd", "cleverhans"])
    parser.add_argument("--data_dir",   type=str, default="./unified_dataset")
    parser.add_argument("--split",      type=str, default="test")
    parser.add_argument("--epsilon",    type=float, default=0.005)
    parser.add_argument("--pgd_steps",  type=int, default=40)
    parser.add_argument("--pgd_alpha",  type=float, default=0.001)
    parser.add_argument("--num_samples",type=int, default=None)
    parser.add_argument("--num_iter_stage1", type=int, default=50)
    parser.add_argument("--num_iter_stage2", type=int, default=100)
    parser.add_argument("--output_dir", type=str, default="./robustness_results")
    parser.add_argument("--device",     type=str, default="auto")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") \
        if args.device == "auto" else torch.device(args.device)
    print(f"Device: {device} | Attack: {args.attack.upper()} | Model: {args.model} | ε={args.epsilon}")

    # Load model — route to the correct detector class based on --model
    print(f"Loading checkpoint: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location="cpu")

    if args.model == "pinn":
        if PINNDetector is None:
            raise RuntimeError("Could not import PINNDetector from train_pinn.py")
        model = PINNDetector()
    elif args.model == "kan":
        if KANDetector is None:
            raise RuntimeError("Could not import KANDetector from train_kan.py")
        model = KANDetector()
    elif args.model == "neural_ode":
        if ODEDetector is None:
            raise RuntimeError("Could not import ODEDetector from train_ode.py")
        model = ODEDetector()
    elif args.model == "transformer":
        # The transformer checkpoint was trained with PINNDetector architecture
        # (same Wav2Vec2 + Transformer backbone, just no physics loss during training)
        if PINNDetector is None:
            raise RuntimeError("Could not import PINNDetector from train_pinn.py")
        model = PINNDetector()
    else:
        raise ValueError(f"Unknown model type: {args.model}")

    try:
        model.load_state_dict(checkpoint["model_state_dict"])
        print("Checkpoint loaded (strict).")
    except Exception as e:
        print(f"[WARN] Strict load failed ({e}), trying non-strict...")
        info = model.load_state_dict(checkpoint["model_state_dict"], strict=False)
        print(f"  Missing: {info.missing_keys}")
        print(f"  Unexpected: {info.unexpected_keys}")

    model = model.to(device)
    model.eval()
    print("Model loaded.")

    # Dataset
    csv_path = Path(args.data_dir) / f"{args.split}.csv"
    if not csv_path.exists():
        csv_path = Path(args.data_dir) / "val.csv"
        print(f"[WARN] {args.split}.csv not found, using val.csv")
    print(f"Dataset: {csv_path} (max={args.num_samples or 'ALL'})")

    dataset = CSVAudioDataset(csv_path, max_samples=args.num_samples)
    print(f"Loaded {len(dataset)} samples.")

    batch_size = 1 if args.attack == "cleverhans" else 16
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    # Run attack
    print(f"\n{'='*60}")
    print(f"ATTACK: {args.attack.upper()} on {args.model.upper()}")
    print(f"{'='*60}\n")

    if args.attack == "fgsm":
        attack = FGSMAttack(model=model, epsilon=args.epsilon, device=device)
        adv_results = evaluate_standard_attack(model, dataloader, attack, device, args.num_samples)

    elif args.attack == "pgd":
        attack = PGDAttack(model=model, epsilon=args.epsilon,
                           alpha=args.pgd_alpha, steps=args.pgd_steps, device=device)
        adv_results = evaluate_standard_attack(model, dataloader, attack, device, args.num_samples)

    else:  # cleverhans
        attack = CleverHansAttack(model=model, device=device, initial_bound=args.epsilon,
                                  num_iter_stage1=args.num_iter_stage1,
                                  num_iter_stage2=args.num_iter_stage2)
        adv_results = evaluate_adversarial_robustness_cleverhans(
            model, dataloader, attack, device,
            num_samples=args.num_samples or len(dataset)
        )

    # Print summary
    robustness_drop = adv_results['clean_accuracy'] - adv_results['adversarial_accuracy']
    print(f"\n{'='*60}")
    print(f"RESULTS: {args.attack.upper()} | {args.model.upper()}")
    print(f"{'='*60}")
    print(f"  Clean Accuracy:       {adv_results['clean_accuracy']:.4f}")
    print(f"  Adversarial Accuracy: {adv_results['adversarial_accuracy']:.4f}")
    print(f"  Attack Success Rate:  {adv_results['attack_success_rate']:.4f}")
    print(f"  Robustness Drop:      {robustness_drop:.4f} ({robustness_drop*100:.1f}%)")
    print(f"  Mean SNR:             {adv_results['mean_snr']:.1f} dB")
    print(f"  Samples:              {adv_results['total_samples']}")

    # Save JSON
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results = {
        'model': args.model,
        'checkpoint': args.checkpoint,
        'attack': args.attack,
        'epsilon': args.epsilon,
        'clean_accuracy': adv_results['clean_accuracy'],
        'adversarial_accuracy': adv_results['adversarial_accuracy'],
        'attack_success_rate': adv_results['attack_success_rate'],
        'robustness_drop': robustness_drop,
        'mean_snr_db': adv_results['mean_snr'],
        'total_samples': adv_results['total_samples'],
        'timestamp': datetime.now().isoformat()
    }
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = output_dir / f"{args.model}_{args.attack}_{ts}.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {output_file}")


if __name__ == "__main__":
    main()
