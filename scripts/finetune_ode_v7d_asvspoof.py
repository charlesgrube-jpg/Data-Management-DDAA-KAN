#!/usr/bin/env python3
"""
Fine-tune NeuralODE v7d on ASVspoof 2019 LA.

CRITICAL: Imports ODEDetector from train_ode_v7d (NOT train_ode).
The v7d model returns (logits, z_traj) — all inference calls must unpack this.
Checkpoint: models_ode_v7/ode_v7_best.pt (v7d epoch 11, val AUC=91.32%)

Same 2-phase protocol as PINN/KAN/Transformer:
  Phase 1 (epochs 1-5): head only, backbone frozen
  Phase 2 (epochs 6-10): unfreeze top 4 wav2vec2 encoder layers
"""

import argparse
import sys
import json
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from tqdm import tqdm
import torchaudio
import torchaudio.transforms as T_audio
from sklearn.metrics import roc_curve, roc_auc_score

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

# CRITICAL: import from train_ode_v7d, NOT train_ode
from train_ode_v7d import ODEDetector


# ── Dataset ────────────────────────────────────────────────────────────────────
class ASVspoofDataset(Dataset):
    SAMPLE_RATE = 16000
    MAX_LENGTH  = 16000 * 5  # 5 seconds

    def __init__(self, audio_dir: Path, protocol_file: Path, augment=False):
        self.augment = augment
        self.samples = []
        with open(protocol_file) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                filename = parts[1]
                label    = 0 if parts[4] == "bonafide" else 1
                for ext in [".flac", ".wav"]:
                    p = audio_dir / f"{filename}{ext}"
                    if p.exists():
                        self.samples.append((str(p), label))
                        break

        import random
        random.seed(42)
        random.shuffle(self.samples)
        real = sum(1 for _, l in self.samples if l == 0)
        fake = sum(1 for _, l in self.samples if l == 1)
        print(f"  Loaded {len(self.samples)} samples ({real} real, {fake} fake)")

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
                wav = T_audio.Resample(sr, self.SAMPLE_RATE)(wav.unsqueeze(0)).squeeze(0)
            orig_len = min(wav.shape[0], self.MAX_LENGTH)
            if wav.shape[0] > self.MAX_LENGTH:
                wav = wav[:self.MAX_LENGTH]
                orig_len = self.MAX_LENGTH
            elif wav.shape[0] < self.MAX_LENGTH:
                wav = F.pad(wav, (0, self.MAX_LENGTH - wav.shape[0]))
            wav = torch.nan_to_num(wav / (wav.abs().max() + 1e-6))
            if self.augment and torch.rand(1).item() < 0.3:
                wav = wav + torch.randn_like(wav) * 0.003
            return wav, label, orig_len
        except Exception:
            return torch.zeros(self.MAX_LENGTH), label, self.MAX_LENGTH


def collate_fn(batch):
    w, l, le = zip(*batch)
    return torch.stack(w), torch.tensor(l, dtype=torch.long), torch.tensor(le, dtype=torch.long)


# ── Metrics ────────────────────────────────────────────────────────────────────
def compute_eer(labels, scores):
    fpr, tpr, _ = roc_curve(labels, scores, pos_label=1)
    fnr = 1.0 - tpr
    idx = np.nanargmin(np.abs(fpr - fnr))
    return float((fpr[idx] + fnr[idx]) / 2.0)


def make_mask(wav, lens, device):
    """Length-based attention mask — CRITICAL for v7d (not value-based)."""
    return (torch.arange(wav.shape[1], device=device)
            .unsqueeze(0).expand(wav.shape[0], -1)
            < lens.unsqueeze(1).to(device)).long()


# ── Evaluation ─────────────────────────────────────────────────────────────────
@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    all_labels, all_scores = [], []
    for wav, labels, lens in tqdm(loader, desc="  Eval"):
        wav  = wav.to(device)
        mask = make_mask(wav, lens, device)
        out  = model(wav, attention_mask=mask)
        # v7d returns (logits, z_traj) — unpack
        logits = out[0] if isinstance(out, (tuple, list)) else out
        probs  = torch.softmax(logits, dim=-1)[:, 1]
        all_labels.extend(labels.tolist())
        all_scores.extend(probs.cpu().tolist())

    labels_np = np.array(all_labels)
    scores_np = np.array(all_scores)
    preds_np  = (scores_np > 0.5).astype(int)
    acc = (preds_np == labels_np).mean()
    auc = roc_auc_score(labels_np, scores_np)
    eer = compute_eer(labels_np, scores_np)
    return dict(accuracy=float(acc), auc_roc=float(auc), eer=float(eer))


# ── Training ───────────────────────────────────────────────────────────────────
def finetune(model, train_loader, eval_loader, device, epochs, lr, out_dir,
             unfreeze_epoch=5, unfreeze_layers=4, lr_backbone=5e-6):
    # Phase 1: only head params are trainable (backbone frozen)
    head_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = AdamW(head_params, lr=lr, weight_decay=0.01)

    # ASVspoof LA is imbalanced (spoof >> real)
    criterion = nn.CrossEntropyLoss(
        weight=torch.tensor([1.0, 0.1]).to(device)
    )
    best_eer = float("inf")
    best_metrics = {}

    for epoch in range(1, epochs + 1):
        # Phase 2: unfreeze top backbone layers
        if unfreeze_epoch > 0 and epoch == unfreeze_epoch:
            encoder_layers = list(model.wav2vec2.encoder.layers)
            for layer in encoder_layers[-unfreeze_layers:]:
                for p in layer.parameters():
                    p.requires_grad = True
            optimizer.add_param_group({
                "params": [p for p in model.wav2vec2.parameters() if p.requires_grad],
                "lr": lr_backbone,
            })
            n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(f"  [Phase 2] Unfroze top {unfreeze_layers} encoder layers. "
                  f"Trainable: {n_trainable:,}")

        model.train()
        total_loss, n_batches = 0, 0
        for wav, labels, lens in tqdm(train_loader, desc=f"  Epoch {epoch}/{epochs}"):
            wav, labels = wav.to(device), labels.to(device)
            mask = make_mask(wav, lens, device)
            optimizer.zero_grad()

            out = model(wav, attention_mask=mask)
            # v7d returns (logits, z_traj) — unpack
            logits = out[0] if isinstance(out, (tuple, list)) else out

            loss = criterion(logits, labels)
            if torch.isnan(loss):
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], 1.0
            )
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1

        metrics = evaluate(model, eval_loader, device)
        print(f"  [NeuralODE v7d] Epoch {epoch}/{epochs} | "
              f"Loss: {total_loss / max(n_batches, 1):.4f} | "
              f"ACC: {metrics['accuracy']:.4f} | "
              f"AUC: {metrics['auc_roc']:.4f} | "
              f"EER: {metrics['eer'] * 100:.2f}%")

        if metrics["eer"] < best_eer:
            best_eer = metrics["eer"]
            best_metrics = metrics
            save_path = out_dir / "neural_ode_v7d_asvspoof_finetuned.pt"
            torch.save({
                "model_state_dict": model.state_dict(),
                **metrics,
            }, save_path)
            print(f"  → Saved best (EER={best_eer * 100:.2f}%)")

    return best_metrics


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--asvspoof_root", required=True)
    parser.add_argument("--train_protocol", required=True)
    parser.add_argument("--eval_protocol", required=True)
    parser.add_argument("--checkpoint", required=True,
                        help="Path to v7d checkpoint (models_ode_v7/ode_v7_best.pt)")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lr_backbone", type=float, default=5e-6)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--unfreeze_epoch", type=int, default=5)
    parser.add_argument("--unfreeze_layers", type=int, default=4)
    parser.add_argument("--output_dir", default="asvspoof_finetuned_v2")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    asvroot = Path(args.asvspoof_root)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load datasets
    print("Loading ASVspoof train set...")
    train_ds = ASVspoofDataset(
        asvroot / "ASVspoof2019_LA_train" / "flac",
        Path(args.train_protocol), augment=True
    )
    print("Loading ASVspoof eval set...")
    eval_ds = ASVspoofDataset(
        asvroot / "ASVspoof2019_LA_eval" / "flac",
        Path(args.eval_protocol), augment=False
    )

    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True, num_workers=4, pin_memory=True,
                              collate_fn=collate_fn)
    eval_loader  = DataLoader(eval_ds, batch_size=args.batch_size,
                              shuffle=False, num_workers=4, pin_memory=True,
                              collate_fn=collate_fn)

    # Load v7d model
    print(f"Loading ODEDetector v7d from {args.checkpoint}...")
    model = ODEDetector()
    ckpt  = torch.load(args.checkpoint, map_location="cpu")
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model = model.to(device)

    # Freeze backbone for Phase 1
    for param in model.wav2vec2.parameters():
        param.requires_grad = False

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    print(f"  Total params: {total:,} | Trainable (Phase 1 head only): {trainable:,}")

    # Fine-tune
    best = finetune(
        model, train_loader, eval_loader, device,
        epochs=args.epochs, lr=args.lr, out_dir=out_dir,
        unfreeze_epoch=args.unfreeze_epoch,
        unfreeze_layers=args.unfreeze_layers,
        lr_backbone=args.lr_backbone,
    )

    print(f"\nBest results — NeuralODE v7d on ASVspoof 2019 LA:")
    print(f"  ACC: {best['accuracy']:.4f} | AUC: {best['auc_roc']:.4f} | "
          f"EER: {best['eer'] * 100:.2f}%")

    ts  = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = out_dir / f"neural_ode_v7d_asvspoof_results_{ts}.json"
    with open(out, "w") as f:
        json.dump({
            "model": "neural_ode_v7d",
            "type": "asvspoof_finetuned",
            "timestamp": datetime.now().isoformat(),
            "best": best,
        }, f, indent=2)
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
