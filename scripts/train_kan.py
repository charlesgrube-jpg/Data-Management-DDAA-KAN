#!/usr/bin/env python3
"""
SOTA KAN-based Audio Deepfake Detection (Paper 1).

Features:
- **Kolmogorov-Arnold Network (KAN) Classifier Head**
  - Uses learnable B-Spline activation functions
  - Replaces standard MLP for better adversarial robustness
- **SOTA Backbone (Wav2Vec2)**
  - Multi-layer fusion
  - Attentive Statistics Pooling (ASP)
  - SpecAugment
- **Robust Training**
  - Differential Learning Rates
  - 48h / 20 Epoch configuration

Usage:
    python scripts/train_kan.py --epochs 20
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


# =============================================================================
# KAN Layer Implementation (B-Splines) - Self Contained
# =============================================================================
class KANLinear(nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        grid_size=5,
        spline_order=3,
        scale_noise=0.1,
        scale_base=1.0,
        scale_spline=1.0,
        enable_standalone_scale_spline=True,
        base_activation=torch.nn.SiLU,
        grid_eps=0.02,
        grid_range=[-1, 1],
    ):
        super(KANLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        self.spline_order = spline_order

        h = (grid_range[1] - grid_range[0]) / grid_size
        grid = (
            (
                torch.arange(-spline_order, grid_size + spline_order + 1) * h
                + grid_range[0]
            )
            .expand(in_features, -1)
            .contiguous()
        )
        self.register_buffer("grid", grid)

        self.base_weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.spline_weight = nn.Parameter(
            torch.Tensor(out_features, in_features, grid_size + spline_order)
        )
        if enable_standalone_scale_spline:
            self.spline_scaler = nn.Parameter(
                torch.Tensor(out_features, in_features)
            )

        self.scale_noise = scale_noise
        self.scale_base = scale_base
        self.scale_spline = scale_spline
        self.enable_standalone_scale_spline = enable_standalone_scale_spline
        self.base_activation = base_activation()
        self.grid_eps = grid_eps

        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.kaiming_uniform_(self.base_weight, a=math.sqrt(5) * self.scale_base)
        self.spline_weight.data.uniform_(-self.scale_noise, self.scale_noise)
        
        if self.enable_standalone_scale_spline:
            torch.nn.init.kaiming_uniform_(self.spline_scaler, a=math.sqrt(5) * self.scale_spline)

    def b_splines(self, x: torch.Tensor):
        """
        Compute the B-spline bases for the given input tensor.
        Args:
            x: (batch_size, in_features)
        Returns:
            (batch_size, in_features, grid_size + spline_order)
        """
        assert x.dim() == 2 and x.size(1) == self.in_features

        grid: torch.Tensor = self.grid
        x = x.unsqueeze(-1)
        bases = ((x >= grid[:, :-1]) & (x < grid[:, 1:])).to(x.dtype)
        for k in range(1, self.spline_order + 1):
            bases = (
                (x - grid[:, : -(k + 1)])
                / (grid[:, k:-1] - grid[:, : -(k + 1)])
                * bases[:, :, :-1]
            ) + (
                (grid[:, k + 1 :] - x)
                / (grid[:, k + 1 :] - grid[:, 1:(-k)])
                * bases[:, :, 1:]
            )

        assert bases.size() == (
            x.size(0),
            self.in_features,
            self.grid_size + self.spline_order,
        )
        return bases

    def curve2coeff(self, x: torch.Tensor, y: torch.Tensor):
        """
        Compute the coefficients of the curve that passes through the given points.
        Args:
            x: (batch_size, in_features) - locations
            y: (batch_size, in_features, out_features) - values
        Returns:
            (out_features, in_features, grid_size + spline_order) - coefficients
        """
        A = self.b_splines(x).transpose(0, 1)  # (in_features, batch_size, grid_size + spline_order)
        B = y.transpose(0, 1)  # (in_features, batch_size, out_features)
        solution = torch.linalg.lstsq(
            A, B
        ).solution  # (in_features, grid_size + spline_order, out_features)
        result = solution.permute(2, 0, 1)  # (out_features, in_features, grid_size + spline_order)
        return result.contiguous()

    @property
    def scaled_spline_weight(self):
        return self.spline_weight * (
            self.spline_scaler.unsqueeze(-1)
            if self.enable_standalone_scale_spline
            else 1.0
        )

    def forward(self, x: torch.Tensor):
        assert x.dim() == 2 and x.size(1) == self.in_features

        base_output = F.linear(self.base_activation(x), self.base_weight)
        
        # B-spline activation
        # Normalize input to grid range [-1, 1] for best spline performance
        # x_norm = torch.tanh(x)  # Optional normalization
        
        bsplines = self.b_splines(x)  # (batch, in, grid+order)
        
        # (batch, in, grid+order) * (out, in, grid+order) -> (batch, out)
        # We need efficient contraction
        spline_output = F.linear(
            bsplines.view(x.size(0), -1),
            self.scaled_spline_weight.view(self.out_features, -1),
        )
        
        return base_output + spline_output


# =============================================================================
# Standard Components (Reused from train.py)
# =============================================================================
class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))
    
    def forward(self, x):
        return self.dropout(x + self.pe[:, :x.size(1), :])

class AttentiveStatisticsPooling(nn.Module):
    def __init__(self, hidden_size: int, attention_dim: int = 128):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, attention_dim),
            nn.Tanh(),
            nn.Linear(attention_dim, 1)
        )
    
    def forward(self, x):
        attn_weights = F.softmax(self.attention(x), dim=1)
        mean = (x * attn_weights).sum(dim=1)
        var = ((x ** 2) * attn_weights).sum(dim=1) - mean ** 2
        std = torch.sqrt(var.clamp(min=1e-6))
        return torch.cat([mean, std], dim=1)

class FeatureSpecAugment(nn.Module):
    def __init__(self, time_mask_param: int = 20, freq_mask_param: int = 48):
        super().__init__()
        self.time_mask_param = time_mask_param
        self.freq_mask_param = freq_mask_param
    
    def forward(self, x):
        if not self.training: return x
        B, T, D = x.shape
        if T > self.time_mask_param:
            t, t0 = random.randint(0, self.time_mask_param), random.randint(0, T - self.time_mask_param)
            x[:, t0:t0+t, :] = 0
        if D > self.freq_mask_param:
            f, f0 = random.randint(0, self.freq_mask_param), random.randint(0, D - self.freq_mask_param)
            x[:, :, f0:f0+f] = 0
        return x


# =============================================================================
# KAN Classifier Model
# =============================================================================
class KANDetector(nn.Module):
    """
    KAN-based Deepfake Detector.
    Backbone: Wav2Vec2 + ASP
    Head: KAN Layers (instead of MLP)
    """
    
    def __init__(
        self,
        model_name: str = "facebook/wav2vec2-base-960h",
        num_classes: int = 2,
        num_layers: int = 4,
        dropout: float = 0.15,
        kan_grid_size: int = 5
    ):
        super().__init__()
        
        print(f"[KAN] Loading {model_name}...")
        config = Wav2Vec2Config.from_pretrained(model_name)
        config.output_hidden_states = True
        self.wav2vec2 = Wav2Vec2Model.from_pretrained(model_name, config=config)
        
        hidden_size = config.hidden_size
        self.hidden_size = hidden_size
        
        # Freeze Encoder Initially
        for param in self.wav2vec2.parameters():
            param.requires_grad = False
            
        self.layer_weights = nn.Parameter(torch.ones(config.num_hidden_layers + 1))
        self.spec_augment = FeatureSpecAugment(time_mask_param=20, freq_mask_param=64)
        
        # Downsample
        self.downsample = nn.Sequential(
            nn.Conv1d(hidden_size, hidden_size, kernel_size=3, stride=1, padding=1, groups=hidden_size),
            nn.Conv1d(hidden_size, hidden_size, kernel_size=1),
            nn.AvgPool1d(kernel_size=2, stride=2),
            nn.GELU(),
            nn.Conv1d(hidden_size, hidden_size, kernel_size=3, stride=1, padding=1, groups=hidden_size),
            nn.Conv1d(hidden_size, hidden_size, kernel_size=1),
            nn.AvgPool1d(kernel_size=2, stride=2),
            nn.GELU()
        )
        
        self.pos_encoder = SinusoidalPositionalEncoding(hidden_size, max_len=2000, dropout=dropout)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size, nhead=8, dim_feedforward=hidden_size * 4,
            dropout=dropout, activation='gelu', batch_first=True, norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.asp = AttentiveStatisticsPooling(hidden_size, attention_dim=128)
        
        # KAN Head
        input_dim = hidden_size * 2  # Mean + Std
        
        # KAN Layers: [input_dim] -> [hidden] -> [classes]
        # Using a deeper KAN allows for learning more complex boundaries
        kan_hidden = hidden_size // 2
        
        self.kan_head = nn.Sequential(
            nn.LayerNorm(input_dim),
            KANLinear(input_dim, kan_hidden, grid_size=kan_grid_size),
            # Note: KANLinear internally includes activation (SiLU + Spline), so no extra non-linearity needed here?
            # Actually, stacking KANs requires no extra activation generally.
            nn.Dropout(dropout),
            KANLinear(kan_hidden, num_classes, grid_size=kan_grid_size)
        )
        
        print(f"[KAN] Head initialized: {input_dim} -> {kan_hidden} -> {num_classes} (Grid={kan_grid_size})")

    def unfreeze_encoder(self):
        for param in self.wav2vec2.parameters():
            param.requires_grad = True
        print("[KAN] Encoder unfrozen")

    def forward(self, input_values, attention_mask=None):
        outputs = self.wav2vec2(input_values, attention_mask=attention_mask, output_hidden_states=True)
        
        all_layers = torch.stack(outputs.hidden_states)
        weights = F.softmax(self.layer_weights, dim=0).view(-1, 1, 1, 1)
        x = (all_layers * weights).sum(0)
        
        x = self.spec_augment(x)
        
        x = x.transpose(1, 2)
        x = self.downsample(x)
        x = x.transpose(1, 2)
        
        x = self.pos_encoder(x)
        x = self.transformer(x)
        
        x = self.asp(x)  # (B, 2*D)
        
        # KAN Classification
        return self.kan_head(x)
    
    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def save_checkpoint(self, path, **kwargs):
        torch.save({'model_state_dict': self.state_dict(), **kwargs}, path)

# =============================================================================
# Dataset & Training (Same as train.py)
# =============================================================================
class AugmentedAudioDataset(Dataset):
    SAMPLE_RATE = 16000
    MAX_LENGTH = 16000 * 5
    
    def __init__(self, csv_path: Path, max_samples: int = None, augment: bool = True):
        self.samples = []
        self.augment = augment
        with open(csv_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                p = row.get('full_path', row.get('filename', ''))
                l = 1 if row.get('label', 'real') in ['synthetic', '1'] else 0
                
                # Fix for relocated dataset (official_dataset)
                if p and not Path(p).exists():
                    p_alt = Path("official_dataset") / p
                    if p_alt.exists():
                        p = str(p_alt)
                        
                if p and Path(p).exists(): self.samples.append((p, l))
        random.shuffle(self.samples)
        if max_samples: self.samples = self.samples[:max_samples]
    
    def __len__(self): return len(self.samples)
    
    def __getitem__(self, idx):
        path, label = self.samples[idx]
        try:
            wav, sr = torchaudio.load(path)
            if wav.shape[0] > 1: wav = wav.mean(0, keepdim=True)
            wav = wav.squeeze(0)
            if sr != 16000: wav = torchaudio.transforms.Resample(sr, 16000)(wav.unsqueeze(0)).squeeze(0)
            
            if self.augment:
                wav = wav * (10 ** (random.uniform(-0.3, 0.3)/20))
                if random.random() < 0.3: wav += torch.randn_like(wav) * 0.005
            
            orig_len = wav.shape[0]
            if orig_len > self.MAX_LENGTH:
                start = random.randint(0, orig_len - self.MAX_LENGTH) if self.augment else 0
                wav = wav[start:start+self.MAX_LENGTH]
                orig_len = self.MAX_LENGTH
            elif orig_len < self.MAX_LENGTH:
                wav = F.pad(wav, (0, self.MAX_LENGTH - orig_len))
                
            wav = torch.nan_to_num(wav / (wav.abs().max() + 1e-6))
            return wav, label, orig_len
        except: return torch.zeros(self.MAX_LENGTH), label, self.MAX_LENGTH

def collate_fn(b):
    w, l, le = zip(*b)
    return torch.stack(w), torch.tensor(l, dtype=torch.long), torch.tensor(le, dtype=torch.long)

def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, valid_count = 0, 0
    all_preds, all_labels = [], []
    pbar = tqdm(loader, desc="Training")
    
    for wavs, labels, lens in pbar:
        wavs, labels = wavs.to(device), labels.to(device)
        mask = (torch.arange(wavs.shape[1], device=device).expand(len(lens), -1) < lens.unsqueeze(1).to(device)).long()
        
        optimizer.zero_grad()
        logits = model(wavs, attention_mask=mask)
        loss = criterion(logits, labels)
        
        if torch.isnan(loss): continue
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        total_loss += loss.item()
        valid_count += 1
        all_preds.extend(logits.argmax(1).cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
    return total_loss / max(valid_count, 1), accuracy_score(all_labels, all_preds) if all_labels else 0

@torch.no_grad()
def eval_epoch(model, loader, criterion, device):
    model.eval()
    total_loss, all_preds, all_labels = 0, [], []
    # Add progress bar for validation
    pbar = tqdm(loader, desc="Validating")
    
    for wavs, labels, lens in pbar:
        wavs, labels = wavs.to(device), labels.to(device)
        mask = (torch.arange(wavs.shape[1], device=device).expand(len(lens), -1) < lens.unsqueeze(1).to(device)).long()
        logits = model(wavs, attention_mask=mask)
        loss = criterion(logits, labels)
        if not torch.isnan(loss): total_loss += loss.item()
        all_preds.extend(logits.argmax(1).cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        
        # Optional: Update postfix with current loss
        pbar.set_postfix({'val_loss': f'{loss.item():.4f}'})
        
    return total_loss / len(loader), accuracy_score(all_labels, all_preds), f1_score(all_labels, all_preds, average='binary', zero_division=0)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=48)
    parser.add_argument("--freeze_epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-3, help="Higher LR for KAN")
    parser.add_argument("--output_dir", type=str, default="models_kan")
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    train_ds = AugmentedAudioDataset("unified_dataset/train.csv", augment=True)
    val_ds = AugmentedAudioDataset("unified_dataset/val.csv", augment=False)
    
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=8, collate_fn=collate_fn)
    val_dl = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=8, collate_fn=collate_fn)
    
    model = KANDetector().to(device)
    print(f"KAN Parameters: {model.count_parameters():,}")
    
    # Optimizer with Differential LR (Fixed for Phase 2)
    kan_params = [p for n, p in model.named_parameters() if 'kan_head' in n]
    other_params = [p for n, p in model.named_parameters() if 'kan_head' not in n and 'wav2vec2' not in n]
    # Initially backbone is frozen
    
    optimizer = AdamW([
        {'params': kan_params, 'lr': args.lr},  # KAN needs higher LR (1e-3 usually good)
        {'params': other_params, 'lr': 5e-4}
    ], weight_decay=0.01)
    
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
    criterion = nn.CrossEntropyLoss(weight=torch.tensor([1.0, 1.65]).to(device))
    
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    best_acc = 0
    for epoch in range(1, args.epochs + 1):
        if epoch == args.freeze_epochs + 1:
            print("\n=== Unfreezing Encoder ===")
            model.unfreeze_encoder()
            
            # Reconstruct optimizer with differential LR
            encoder_params = [p for n, p in model.named_parameters() if 'wav2vec2' in n]
            kan_params = [p for n, p in model.named_parameters() if 'kan_head' in n]
            other_params = [p for n, p in model.named_parameters() if 'kan_head' not in n and 'wav2vec2' not in n]
            
            optimizer = AdamW([
                {'params': encoder_params, 'lr': 1e-5},  # Gentle
                {'params': kan_params, 'lr': args.lr},   # Aggressive
                {'params': other_params, 'lr': 5e-4}
            ], weight_decay=0.01)
            scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
            
        tl, ta = train_epoch(model, train_dl, optimizer, criterion, device)
        vl, va, vf = eval_epoch(model, val_dl, criterion, device)
        scheduler.step()
        
        phase = "P1" if epoch <= args.freeze_epochs else "P2"
        print(f"[{phase}] Epoch {epoch:3d} | TL: {tl:.4f} TA: {ta:.4f} | VL: {vl:.4f} VA: {va:.4f} VF: {vf:.4f}")
        
        if va > best_acc:
            best_acc = va
            model.save_checkpoint(f"{args.output_dir}/kan_best.pt", val_acc=va)
            print("  -> Saved Best")

if __name__ == "__main__":
    main()
