#!/usr/bin/env python3
"""
regen_trajectory_v3.py — Regenerate trajectory_pca.png with fixed visualization.

Fixes vs. previous version:
  - Independent axis limits per panel (real cluster no longer crushed by fake range)
  - linewidth=3.0, alpha=0.80 (lines are now clearly visible)
  - n_display=10 (fewer, cleaner trajectories)
  - Directional arrows at trajectory midpoints
  - 95% confidence ellipses at t=0 and t=1
  - Per-panel zoom to actual data extent

Usage:
    python scripts/regen_trajectory_v3.py \
        --ode_ckpt "Best Models/ode_best.pt" \
        --data_csv unified_dataset/test.csv \
        --out      papers/figures/trajectory_pca.png \
        --n_samples 40
"""

import argparse, importlib.util, sys, csv, random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torchaudio

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

# ── Helpers ──────────────────────────────────────────────────────────────────

SAMPLE_RATE = 16_000
MAX_LEN     = SAMPLE_RATE * 5      # 5 s


def load_audio(path: str) -> torch.Tensor:
    wav, sr = torchaudio.load(path)
    if wav.shape[0] > 1:
        wav = wav.mean(0, keepdim=True)
    wav = wav.squeeze(0)
    if sr != SAMPLE_RATE:
        wav = torchaudio.functional.resample(wav, sr, SAMPLE_RATE)
    if wav.shape[0] > MAX_LEN:
        wav = wav[:MAX_LEN]
    else:
        wav = F.pad(wav, (0, MAX_LEN - wav.shape[0]))
    wav = torch.nan_to_num(wav / (wav.abs().max() + 1e-6))
    return wav


def _load_module(script_path, module_name):
    spec   = importlib.util.spec_from_file_location(module_name, script_path)
    mod    = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return mod


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ode_ckpt",  required=True)
    ap.add_argument("--data_csv",  default="unified_dataset/test.csv")
    ap.add_argument("--out",       default="papers/figures/trajectory_pca.png")
    ap.add_argument("--n_samples", type=int, default=40,
                    help="Total samples to load (balanced real/fake)")
    ap.add_argument("--n_display", type=int, default=10,
                    help="Trajectories to show per panel")
    ap.add_argument("--device",    default="auto")
    args = ap.parse_args()

    device = (torch.device("cuda") if torch.cuda.is_available()
              else torch.device("cpu")) \
             if args.device == "auto" else torch.device(args.device)
    print(f"[main] device={device}")

    # ── Load ODE model ──────────────────────────────────────────────────────
    print(f"[main] Loading ODE checkpoint: {args.ode_ckpt}")
    mod   = _load_module(ROOT / "scripts" / "train_ode.py", "train_ode")
    model = mod.ODEDetector()
    ckpt  = torch.load(args.ode_ckpt, map_location="cpu", weights_only=False)
    state = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state, strict=False)
    model = model.to(device).eval()
    print("[main] Model loaded ✓")

    # ── Load balanced audio batch ───────────────────────────────────────────
    csv_path = ROOT / args.data_csv
    if not csv_path.exists():
        csv_path = csv_path.parent / "val.csv"
    print(f"[main] Reading CSV: {csv_path}")

    with open(csv_path) as f:
        rows = list(csv.DictReader(f))
    random.seed(42)
    random.shuffle(rows)

    real_wavs, fake_wavs = [], []
    n_each = args.n_samples // 2

    for row in rows:
        p     = row.get("full_path", row.get("filename", "")).strip()
        label = 1 if row.get("label", "real") in ["synthetic", "1"] else 0
        for cand in [Path(p), ROOT / p,
                     ROOT / "official_dataset" / p,
                     ROOT / "completed_correct_dataset" / p]:
            if cand.exists():
                bucket = fake_wavs if label == 1 else real_wavs
                if len(bucket) < n_each:
                    bucket.append(load_audio(str(cand)))
                break
        if len(real_wavs) >= n_each and len(fake_wavs) >= n_each:
            break

    print(f"[main] Loaded {len(real_wavs)} real, {len(fake_wavs)} fake clips")
    real_x = torch.stack(real_wavs).to(device)
    fake_x = torch.stack(fake_wavs).to(device)

    # ── Extract trajectories ────────────────────────────────────────────────
    from pipeline.interpretability.ode_interpret import (
        extract_ode_trajectories, plot_trajectory_pca)

    print("[main] Extracting real trajectories …")
    _, traj_real, _ = extract_ode_trajectories(
        model, real_x,
        torch.zeros(real_x.shape[0], dtype=torch.long))

    print("[main] Extracting fake trajectories …")
    _, traj_fake, _ = extract_ode_trajectories(
        model, fake_x,
        torch.ones(fake_x.shape[0], dtype=torch.long))

    print(f"[main] traj_real shape: {traj_real.shape}")
    print(f"[main] traj_fake shape: {traj_fake.shape}")

    # Diagnose trajectory movement
    real_disp = (traj_real[:, -1, :] - traj_real[:, 0, :]).norm(dim=-1)
    fake_disp = (traj_fake[:, -1, :] - traj_fake[:, 0, :]).norm(dim=-1)
    print(f"[main] Real displacement  mean={real_disp.mean():.4f}  "
          f"max={real_disp.max():.4f}")
    print(f"[main] Fake displacement  mean={fake_disp.mean():.4f}  "
          f"max={fake_disp.max():.4f}")

    # ── Generate figure ─────────────────────────────────────────────────────
    out_path = ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[main] Plotting → {out_path}")
    plot_trajectory_pca(
        traj_real, traj_fake,
        n_display=args.n_display,
        save_path=str(out_path),
    )
    print("[main] Done ✓")


if __name__ == "__main__":
    main()
