#!/usr/bin/env python3
"""
SOTA Neural ODE Audio Deepfake Detection (Paper 2).

Features:
- **Neural ODE Classifier Head**
  - Models latent feature evolution in continuous time
  - Uses adaptive step size solver (Dopri5) or fixed RK4
  - Captures temporal artifacts/irregularities better than RNNs
- **SOTA Backbone (Wav2Vec2)**
  - Multi-layer fusion
  - Attentive Statistics Pooling (ASP)
  - SpecAugment
- **Robust Training**
  - Differential Learning Rates
  - 48h / 20 Epoch configuration

Usage:
    pip install torchdiffeq
    python scripts/train_ode.py --epochs 20

Fixes applied vs original:
  1. ODEFunc now takes time `t` as a real input (time-aware / non-autonomous dynamics).
  2. NeuralODEClassifier returns the full ODE trajectory so the physics loss
     can constrain the ODE states directly (not the transformer features).
  3. TemporalSmoothnessLoss is now applied to ODE trajectory states, making
     the physics loss and ODE dynamics coherent.
  4. integration_time registered as a buffer (no per-call .to(device)).
  5. Amplitude augmentation moved before normalisation so it isn't cancelled out.
"""

import argparse
import csv
import math
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from tqdm import tqdm
from sklearn.metrics import f1_score, accuracy_score

try:
    from transformers import Wav2Vec2Model, Wav2Vec2Config
    import torchaudio
except ImportError:
    print("Install: pip install transformers torchaudio")
    exit(1)

# Try importing torchdiffeq, else use simple fallback
try:
    from torchdiffeq import odeint
    HAS_TORCHDIFFEQ = True
except ImportError:
    HAS_TORCHDIFFEQ = False
    print("[Warning] torchdiffeq not found. Using simple RK4 solver.")


# =============================================================================
# Neural ODE Components
# =============================================================================
class ODEFunc(nn.Module):
    """
    Defines the dynamics of the system: dz/dt = f(z, t)

    FIX: time `t` is now concatenated to `z` before passing through the network,
    making the dynamics non-autonomous (time-aware).  The original code accepted
    `t` as an argument but silently ignored it, limiting expressiveness.
    """
    def __init__(self, hidden_dim: int):
        super().__init__()
        # Input is [z || t], so +1 for the scalar time dimension.
        self.net = nn.Sequential(
            nn.Linear(hidden_dim + 1, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.01)
                nn.init.constant_(m.bias, 0)

        self.nfe = 0  # Number of function evaluations

    def forward(self, t: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        self.nfe += 1
        # Broadcast scalar t to match batch size, then concatenate.
        # t may be a 0-d tensor from torchdiffeq or a plain float from RK4.
        t_val = t if isinstance(t, torch.Tensor) else torch.tensor(t, dtype=z.dtype, device=z.device)
        t_expanded = t_val.expand(z.shape[0], 1)          # (B, 1)
        zt = torch.cat([z, t_expanded], dim=-1)            # (B, hidden+1)
        return self.net(zt)


def rk4_step(func, t, z, dt):
    k1 = func(t,          z)
    k2 = func(t + dt / 2, z + dt / 2 * k1)
    k3 = func(t + dt / 2, z + dt / 2 * k2)
    k4 = func(t + dt,     z + dt * k3)
    return z + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)


class NeuralODEClassifier(nn.Module):
    """
    FIX: returns the full ODE trajectory (all time steps) rather than only
    the final state, so callers can apply physics losses over the trajectory.
    """

    # Number of time points at which the trajectory is evaluated.
    N_STEPS = 11  # t = 0, 0.1, 0.2, ..., 1.0

    def __init__(self, input_dim: int, hidden_dim: int = 128):
        super().__init__()
        # FIX: LayerNorm to bound z0 and prevent immediate divergence
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Tanh()
        )
        self.ode_func   = ODEFunc(hidden_dim)
        self.classifier = nn.Linear(hidden_dim, 2)

        # FIX: register as buffer so .to(device) is handled automatically.
        t_span = torch.linspace(0, 1, self.N_STEPS)
        self.register_buffer("integration_time", t_span)

    def forward(self, x: torch.Tensor):
        """
        Returns
        -------
        logits      : (B, 2)
        z_trajectory: (B, N_STEPS, hidden_dim)  – full ODE trajectory
        """
        z0 = self.input_proj(x)  # (B, hidden_dim)

        if HAS_TORCHDIFFEQ:
            # odeint returns (N_STEPS, B, hidden_dim)
            z_t = odeint(
                self.ode_func,
                z0,
                self.integration_time,
                method="rk4",
                options={"step_size": 0.025}, # FIX: Smaller step size
            )
            z_trajectory = z_t.permute(1, 0, 2)   # (B, N_STEPS, hidden_dim)
        else:
            steps = 40 # FIX: Combined with step_size 0.025 equivalent (1.0/0.025 = 40)
            dt    = 1.0 / steps
            z     = z0
            t     = 0.0
            traj  = [z0.unsqueeze(1)]
            for _ in range(steps):
                z = rk4_step(self.ode_func, t, z, dt)
                t += dt
                # simple sampling for trajectory (approximate)
                if len(traj) < self.N_STEPS: 
                    # Store roughly every 4th step to match N_STEPS=11
                    if len(traj) * (steps // (self.N_STEPS - 1)) <= _ + 1:
                         traj.append(z.unsqueeze(1))
            
            # Ensure we have N_STEPS
            while len(traj) < self.N_STEPS:
                 traj.append(z.unsqueeze(1))
                 
            z_trajectory = torch.cat(traj, dim=1)  # (B, N_STEPS, hidden_dim)

        z_final = z_trajectory[:, -1, :]           # (B, hidden_dim)
        
        # FIX: NaN Guard
        if torch.isnan(z_final).any():
             # Fallback to initial state if integration exploded
             z_final = torch.where(torch.isnan(z_final), z0, z_final)
             
        logits  = self.classifier(z_final)          # (B, 2)
        return logits, z_trajectory


# =============================================================================
# Standard Components
# =============================================================================
class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return self.dropout(x + self.pe[:, : x.size(1), :])


class AttentiveStatisticsPooling(nn.Module):
    def __init__(self, hidden_size: int, attention_dim: int = 128):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, attention_dim),
            nn.Tanh(),
            nn.Linear(attention_dim, 1),
        )

    def forward(self, x):
        attn_weights = F.softmax(self.attention(x), dim=1)
        mean = (x * attn_weights).sum(dim=1)
        var  = ((x ** 2) * attn_weights).sum(dim=1) - mean ** 2
        std  = torch.sqrt(var.clamp(min=1e-6))
        return torch.cat([mean, std], dim=1)


class FeatureSpecAugment(nn.Module):
    def __init__(self, time_mask_param: int = 20, freq_mask_param: int = 48):
        super().__init__()
        self.time_mask_param = time_mask_param
        self.freq_mask_param = freq_mask_param

    def forward(self, x):
        if not self.training:
            return x
        B, T, D = x.shape
        if T > self.time_mask_param:
            t  = random.randint(0, self.time_mask_param)
            t0 = random.randint(0, T - self.time_mask_param)
            x[:, t0 : t0 + t, :] = 0
        if D > self.freq_mask_param:
            f  = random.randint(0, self.freq_mask_param)
            f0 = random.randint(0, D - self.freq_mask_param)
            x[:, :, f0 : f0 + f] = 0
        return x


# =============================================================================
# Physics-Informed Loss
# =============================================================================
class TemporalSmoothnessLoss(nn.Module):
    """
    Penalises large first-order differences along the time axis.

    FIX: Previously this was applied to transformer hidden states, which have
    nothing to do with the ODE.  It is now applied to the ODE trajectory
    (B, N_STEPS, hidden_dim), making the physics regularisation coherent with
    the ODE dynamics.
    """

    def forward(self, z_trajectory: torch.Tensor) -> torch.Tensor:
        # z_trajectory: (B, N_STEPS, hidden_dim)
        if z_trajectory.dim() != 3:
            return torch.tensor(0.0, device=z_trajectory.device)
        diff = z_trajectory[:, 1:, :] - z_trajectory[:, :-1, :]  # (B, N_STEPS-1, D)
        return diff.pow(2).mean(dim=(1, 2))                        # (B,)


# =============================================================================
# Main Model
# =============================================================================
class ODEDetector(nn.Module):
    """
    Neural ODE-based Deepfake Detector.
    Backbone : Wav2Vec2 + ASP
    Head     : Neural ODE (Latent Evolution)
    """

    def __init__(
        self,
        model_name: str = "facebook/wav2vec2-base-960h",
        num_layers: int = 4,
        dropout: float = 0.15,
        ode_hidden: int = 256,
    ):
        super().__init__()

        print(f"[ODE] Loading {model_name}...")
        config = Wav2Vec2Config.from_pretrained(model_name)
        config.output_hidden_states = True
        self.wav2vec2  = Wav2Vec2Model.from_pretrained(model_name, config=config)
        hidden_size    = config.hidden_size
        self.hidden_size = hidden_size

        # Freeze encoder initially.
        for param in self.wav2vec2.parameters():
            param.requires_grad = False

        self.layer_weights = nn.Parameter(torch.ones(config.num_hidden_layers + 1))
        self.spec_augment  = FeatureSpecAugment(time_mask_param=20, freq_mask_param=64)

        self.downsample = nn.Sequential(
            nn.Conv1d(hidden_size, hidden_size, kernel_size=3, stride=1, padding=1, groups=hidden_size),
            nn.Conv1d(hidden_size, hidden_size, kernel_size=1),
            nn.AvgPool1d(kernel_size=2, stride=2),
            nn.GELU(),
            nn.Conv1d(hidden_size, hidden_size, kernel_size=3, stride=1, padding=1, groups=hidden_size),
            nn.Conv1d(hidden_size, hidden_size, kernel_size=1),
            nn.AvgPool1d(kernel_size=2, stride=2),
            nn.GELU(),
        )

        self.pos_encoder = SinusoidalPositionalEncoding(hidden_size, max_len=2000, dropout=dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size, nhead=8, dim_feedforward=hidden_size * 4,
            dropout=dropout, activation="gelu", batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.asp      = AttentiveStatisticsPooling(hidden_size, attention_dim=128)
        self.ode_head = NeuralODEClassifier(hidden_size * 2, hidden_dim=ode_hidden)

        # FIX: physics loss now operates on ODE trajectory, not transformer states.
        self.phys_loss_fn = TemporalSmoothnessLoss()

        backend = "torchdiffeq" if HAS_TORCHDIFFEQ else "Custom RK4"
        print(f"[ODE] Head initialised. Latent dim: {ode_hidden}. Backend: {backend}")

    def unfreeze_encoder(self):
        for param in self.wav2vec2.parameters():
            param.requires_grad = True
        print("[ODE] Encoder unfrozen")

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def save_checkpoint(self, path, **kwargs):
        torch.save({"model_state_dict": self.state_dict(), **kwargs}, path)

    def forward(self, input_values, attention_mask=None, return_physics=False):
        outputs  = self.wav2vec2(input_values, attention_mask=attention_mask, output_hidden_states=True)
        all_layers = torch.stack(outputs.hidden_states)           # (L, B, T, D)
        weights    = F.softmax(self.layer_weights, dim=0).view(-1, 1, 1, 1)
        x = (all_layers * weights).sum(0)                         # (B, T, D)
        x = self.spec_augment(x)

        x = x.transpose(1, 2)
        x = self.downsample(x)
        x = x.transpose(1, 2)

        x = self.pos_encoder(x)
        x = self.transformer(x)                                   # (B, T, D)

        x_pooled = self.asp(x)                                    # (B, 2D)

        # FIX: ode_head now returns (logits, trajectory).
        logits, z_trajectory = self.ode_head(x_pooled)

        if return_physics:
            # FIX: physics loss applied to ODE trajectory — coherent with ODE dynamics.
            phys_loss = self.phys_loss_fn(z_trajectory)           # (B,)
            return logits, phys_loss

        return logits


# =============================================================================
# Dataset & Training
# =============================================================================
class AugmentedAudioDataset(Dataset):
    SAMPLE_RATE = 16000
    MAX_LENGTH  = 16000 * 5

    def __init__(self, csv_path: Path, max_samples: int = None, augment: bool = True):
        self.samples = []
        self.augment = augment
        with open(csv_path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                p = row.get("full_path", row.get("filename", ""))
                l = 1 if row.get("label", "real") in ["synthetic", "1"] else 0
                if p and not Path(p).exists():
                    p_alt = Path("official_dataset") / p
                    if p_alt.exists():
                        p = str(p_alt)
                if p and Path(p).exists():
                    self.samples.append((p, l))
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
            if sr != 16000:
                wav = torchaudio.transforms.Resample(sr, 16000)(wav.unsqueeze(0)).squeeze(0)

            # FIX: normalise first, then apply amplitude augmentation so the
            # gain shift is not immediately cancelled by the re-normalisation
            # that used to follow it in the original code.
            wav = wav / (wav.abs().max() + 1e-6)

            if self.augment:
                wav = wav * (10 ** (random.uniform(-0.3, 0.3) / 20))
                if random.random() < 0.3:
                    wav = wav + torch.randn_like(wav) * 0.005

            orig_len = wav.shape[0]
            if orig_len > self.MAX_LENGTH:
                start = random.randint(0, orig_len - self.MAX_LENGTH) if self.augment else 0
                wav   = wav[start : start + self.MAX_LENGTH]
                orig_len = self.MAX_LENGTH
            elif orig_len < self.MAX_LENGTH:
                wav = F.pad(wav, (0, self.MAX_LENGTH - orig_len))

            wav = torch.nan_to_num(wav)
            return wav, label, orig_len
        except:
            return torch.zeros(self.MAX_LENGTH), label, self.MAX_LENGTH


def collate_fn(b):
    w, l, le = zip(*b)
    return torch.stack(w), torch.tensor(l, dtype=torch.long), torch.tensor(le, dtype=torch.long)


def train_epoch(model, loader, optimizer, criterion, device, physics_weight=0.1):
    model.train()
    total_loss, valid_count = 0, 0
    all_preds, all_labels   = [], []
    pbar = tqdm(loader, desc="Training")

    for wavs, labels, lens in pbar:
        wavs, labels = wavs.to(device), labels.to(device)
        mask = (
            torch.arange(wavs.shape[1], device=device)
            .expand(len(lens), -1) < lens.unsqueeze(1).to(device)
        ).long()

        optimizer.zero_grad()
        logits, phys_loss = model(wavs, attention_mask=mask, return_physics=True)
        ce_loss = criterion(logits, labels)

        # Physics regularisation: encourage smooth ODE trajectories for real speech.
        masked_phys = (phys_loss * (labels == 0).float()).mean()
        loss = ce_loss + physics_weight * masked_phys

        if torch.isnan(loss):
            continue
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss  += loss.item()
        valid_count += 1
        all_preds.extend(logits.argmax(1).cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        pbar.set_postfix({"loss": f"{loss.item():.4f}"})

    acc = accuracy_score(all_labels, all_preds) if all_labels else 0
    return total_loss / max(valid_count, 1), acc


@torch.no_grad()
def eval_epoch(model, loader, criterion, device):
    model.eval()
    total_loss, all_preds, all_labels = 0, [], []
    pbar = tqdm(loader, desc="Validating")

    for wavs, labels, lens in pbar:
        wavs, labels = wavs.to(device), labels.to(device)
        mask = (
            torch.arange(wavs.shape[1], device=device)
            .expand(len(lens), -1) < lens.unsqueeze(1).to(device)
        ).long()
        logits = model(wavs, attention_mask=mask)
        loss   = criterion(logits, labels)
        if not torch.isnan(loss):
            total_loss += loss.item()
        all_preds.extend(logits.argmax(1).cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        pbar.set_postfix({"val_loss": f"{loss.item():.4f}"})

    return (
        total_loss / len(loader),
        accuracy_score(all_labels, all_preds),
        f1_score(all_labels, all_preds, average="binary", zero_division=0),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs",        type=int,   default=20)
    parser.add_argument("--batch_size",    type=int,   default=48)
    parser.add_argument("--freeze_epochs", type=int,   default=5)
    parser.add_argument("--lr",            type=float, default=1e-4)
    parser.add_argument("--output_dir",    type=str,   default="models_ode")
    parser.add_argument("--physics_weight", type=float, default=0.1, help="Weight for physics smoothness loss")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    train_ds = AugmentedAudioDataset("unified_dataset/train.csv", augment=True)
    val_ds   = AugmentedAudioDataset("unified_dataset/val.csv",   augment=False)

    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,  num_workers=8, collate_fn=collate_fn)
    val_dl   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False, num_workers=8, collate_fn=collate_fn)

    model = ODEDetector().to(device)
    print(f"ODE Parameters: {model.count_parameters():,}")

    ode_params   = [p for n, p in model.named_parameters() if "ode_head"  in n]
    other_params = [p for n, p in model.named_parameters() if "ode_head" not in n and "wav2vec2" not in n]

    optimizer = AdamW(
        [{"params": ode_params, "lr": args.lr}, {"params": other_params, "lr": 5e-4}],
        weight_decay=0.01,
    )
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
    criterion = nn.CrossEntropyLoss(weight=torch.tensor([1.0, 1.65]).to(device))

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    best_acc = 0
    for epoch in range(1, args.epochs + 1):
        if epoch == args.freeze_epochs + 1:
            print("\n=== Unfreezing Encoder ===")
            model.unfreeze_encoder()

            encoder_params = [p for n, p in model.named_parameters() if "wav2vec2"  in n]
            ode_params     = [p for n, p in model.named_parameters() if "ode_head"  in n]
            other_params   = [p for n, p in model.named_parameters() if "ode_head" not in n and "wav2vec2" not in n]

            optimizer = AdamW(
                [
                    {"params": encoder_params, "lr": 1e-5},
                    {"params": ode_params,     "lr": args.lr},
                    {"params": other_params,   "lr": 5e-4},
                ],
                weight_decay=0.01,
            )
            scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)

        tl, ta = train_epoch(model, train_dl, optimizer, criterion, device, physics_weight=args.physics_weight)
        vl, va, vf = eval_epoch(model, val_dl, criterion, device)
        scheduler.step()

        phase = "P1" if epoch <= args.freeze_epochs else "P2"
        print(
            f"[{phase}] Epoch {epoch:3d} | "
            f"TL: {tl:.4f} TA: {ta:.4f} | VL: {vl:.4f} VA: {va:.4f} VF: {vf:.4f}"
        )

        if va > best_acc:
            best_acc = va
            model.save_checkpoint(f"{args.output_dir}/ode_best.pt", val_acc=va)
            print("  -> Saved Best")


if __name__ == "__main__":
    main()
