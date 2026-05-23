#!/usr/bin/env python3
"""
High-Performance Wav2Vec2 Audio Deepfake Detection.

Performance-focused improvements:
- Multi-layer hidden state fusion (all encoder layers)
- Attentive Statistics Pooling (mean + std)
- Depthwise separable downsampling with anti-aliasing
- SpecAugment on encoder features
- SAM optimizer for generalization

Usage:
    python scripts/train_sota_v2.py --epochs 50
"""

import argparse
import csv
import math
import random
from pathlib import Path
from contextlib import contextmanager

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
# SAM Optimizer (Sharpness-Aware Minimization)
# =============================================================================
class SAM(torch.optim.Optimizer):
    """Sharpness-Aware Minimization for better generalization."""
    
    def __init__(self, params, base_optimizer, rho=0.05, **kwargs):
        defaults = dict(rho=rho, **kwargs)
        super().__init__(params, defaults)
        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups
    
    @torch.no_grad()
    def first_step(self, zero_grad=False):
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)
            for p in group["params"]:
                if p.grad is None:
                    continue
                e_w = p.grad * scale
                p.add_(e_w)
                self.state[p]["e_w"] = e_w
        if zero_grad:
            self.zero_grad()
    
    @torch.no_grad()
    def second_step(self, zero_grad=False):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                p.sub_(self.state[p]["e_w"])
        self.base_optimizer.step()
        if zero_grad:
            self.zero_grad()
    
    def _grad_norm(self):
        shared_device = self.param_groups[0]["params"][0].device
        norm = torch.norm(
            torch.stack([
                p.grad.norm(p=2).to(shared_device)
                for group in self.param_groups
                for p in group["params"]
                if p.grad is not None
            ]),
            p=2
        )
        return norm


# =============================================================================
# Sinusoidal Positional Encoding
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


# =============================================================================
# Attentive Statistics Pooling
# =============================================================================
class AttentiveStatisticsPooling(nn.Module):
    """
    ASP: Captures both mean and standard deviation with attention.
    Standard in speaker verification and anti-spoofing.
    """
    
    def __init__(self, hidden_size: int, attention_dim: int = 128):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, attention_dim),
            nn.Tanh(),
            nn.Linear(attention_dim, 1)
        )
    
    def forward(self, x):
        # x: (B, T, D)
        attn_weights = F.softmax(self.attention(x), dim=1)  # (B, T, 1)
        
        # Weighted mean
        mean = (x * attn_weights).sum(dim=1)  # (B, D)
        
        # Weighted std
        var = ((x ** 2) * attn_weights).sum(dim=1) - mean ** 2
        std = torch.sqrt(var.clamp(min=1e-6))  # (B, D)
        
        # Concatenate mean and std
        return torch.cat([mean, std], dim=1)  # (B, 2*D)


# =============================================================================
# SpecAugment on Features
# =============================================================================
class FeatureSpecAugment(nn.Module):
    """Apply time and feature masking to encoder outputs."""
    
    def __init__(self, time_mask_param: int = 20, freq_mask_param: int = 48):
        super().__init__()
        self.time_mask_param = time_mask_param
        self.freq_mask_param = freq_mask_param
    
    def forward(self, x):
        # x: (B, T, D)
        if not self.training:
            return x
        
        B, T, D = x.shape
        
        # Time masking
        if T > self.time_mask_param:
            t = random.randint(0, self.time_mask_param)
            t0 = random.randint(0, T - t)
            x = x.clone()
            x[:, t0:t0+t, :] = 0
        
        # Feature (frequency) masking
        if D > self.freq_mask_param:
            f = random.randint(0, self.freq_mask_param)
            f0 = random.randint(0, D - f)
            x = x.clone()
            x[:, :, f0:f0+f] = 0
        
        return x


# =============================================================================
# Dataset with Augmentation
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
                audio_path = row.get('full_path', row.get('filename', ''))
                label_str = row.get('label', 'real')
                label = 1 if label_str == 'synthetic' or str(label_str) == '1' else 0
                
                # Fix for relocated dataset (official_dataset)
                if audio_path and not Path(audio_path).exists():
                    p_alt = Path("official_dataset") / audio_path
                    if p_alt.exists():
                        audio_path = str(p_alt)
                
                if audio_path and Path(audio_path).exists():
                    self.samples.append((audio_path, label))
        
        random.shuffle(self.samples)
        if max_samples:
            self.samples = self.samples[:max_samples]
        print(f"[Dataset] {csv_path.name}: {len(self.samples)} samples, augment={augment}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        audio_path, label = self.samples[idx]
        try:
            waveform, sr = torchaudio.load(audio_path)
            if waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0, keepdim=True)
            waveform = waveform.squeeze(0)
            
            if sr != self.SAMPLE_RATE:
                waveform = torchaudio.transforms.Resample(sr, self.SAMPLE_RATE)(waveform.unsqueeze(0)).squeeze(0)
            
            if self.augment:
                # Random gain
                gain = 10 ** (random.uniform(-0.3, 0.3) / 20)
                waveform = waveform * gain
                # Random noise
                if random.random() < 0.3:
                    waveform = waveform + torch.randn_like(waveform) * 0.005
            
            orig_len = waveform.shape[0]
            
            if orig_len > self.MAX_LENGTH:
                start = random.randint(0, orig_len - self.MAX_LENGTH) if self.augment else 0
                waveform = waveform[start:start + self.MAX_LENGTH]
                orig_len = self.MAX_LENGTH
            elif orig_len < self.MAX_LENGTH:
                waveform = F.pad(waveform, (0, self.MAX_LENGTH - orig_len))
            
            if waveform.abs().max() > 0:
                waveform = waveform / waveform.abs().max()
            waveform = torch.nan_to_num(waveform, nan=0.0, posinf=1.0, neginf=-1.0)
            
            return waveform, label, orig_len
        except:
            return torch.zeros(self.MAX_LENGTH), label, self.MAX_LENGTH


def collate_fn(batch):
    waveforms, labels, lengths = zip(*batch)
    return torch.stack(waveforms), torch.tensor(labels, dtype=torch.long), torch.tensor(lengths, dtype=torch.long)


# =============================================================================
# High-Performance Model
# =============================================================================
class HighPerformanceDetector(nn.Module):
    """
    Performance-optimized deepfake detector with:
    - Multi-layer hidden state fusion
    - Attentive Statistics Pooling
    - Depthwise separable downsampling
    - SpecAugment on features
    """
    
    def __init__(
        self,
        model_name: str = "facebook/wav2vec2-base-960h",
        num_classes: int = 2,
        num_layers: int = 4,
        dropout: float = 0.15
    ):
        super().__init__()
        
        print(f"[Model] Loading {model_name} with hidden states...")
        config = Wav2Vec2Config.from_pretrained(model_name)
        config.output_hidden_states = True  # Enable all hidden states
        self.wav2vec2 = Wav2Vec2Model.from_pretrained(model_name, config=config)
        
        hidden_size = config.hidden_size  # 768
        num_hidden_layers = config.num_hidden_layers + 1  # +1 for embedding layer
        self.hidden_size = hidden_size
        
        # Initially freeze encoder
        self.freeze_encoder()
        
        # Learnable layer weights for multi-layer fusion
        self.layer_weights = nn.Parameter(torch.ones(num_hidden_layers))
        
        # SpecAugment on features
        self.spec_augment = FeatureSpecAugment(time_mask_param=20, freq_mask_param=64)
        
        # Depthwise separable downsampling with anti-aliasing
        self.downsample = nn.Sequential(
            nn.Conv1d(hidden_size, hidden_size, kernel_size=3, stride=1, padding=1, groups=hidden_size),
            nn.Conv1d(hidden_size, hidden_size, kernel_size=1),
            nn.AvgPool1d(kernel_size=2, stride=2),  # Anti-aliasing
            nn.GELU(),
            nn.Conv1d(hidden_size, hidden_size, kernel_size=3, stride=1, padding=1, groups=hidden_size),
            nn.Conv1d(hidden_size, hidden_size, kernel_size=1),
            nn.AvgPool1d(kernel_size=2, stride=2),  # 4x total reduction
            nn.GELU()
        )
        
        # Positional encoding
        self.pos_encoder = SinusoidalPositionalEncoding(hidden_size, max_len=2000, dropout=dropout)
        
        # Transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size, nhead=8,
            dim_feedforward=hidden_size * 4,
            dropout=dropout, activation='gelu',
            batch_first=True, norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Attentive Statistics Pooling (outputs 2*D)
        self.asp = AttentiveStatisticsPooling(hidden_size, attention_dim=128)
        
        # Classification head
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_size * 2),
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 2, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout / 2),
            nn.Linear(hidden_size, num_classes)
        )
        
        print(f"[Model] Multi-layer fusion ({num_hidden_layers} layers) + ASP + DepthwiseSep downsample")
    
    def freeze_encoder(self):
        for param in self.wav2vec2.parameters():
            param.requires_grad = False
        print("[Model] Encoder frozen")
    
    def unfreeze_encoder(self):
        for param in self.wav2vec2.parameters():
            param.requires_grad = True
        print("[Model] Encoder unfrozen")
    
    def forward(self, input_values, attention_mask=None):
        B = input_values.shape[0]
        
        # Get all hidden states
        outputs = self.wav2vec2(input_values, attention_mask=attention_mask, output_hidden_states=True)
        
        # Multi-layer fusion with learnable weights
        all_layers = torch.stack(outputs.hidden_states)  # (num_layers, B, T, D)
        weights = F.softmax(self.layer_weights, dim=0).view(-1, 1, 1, 1)
        x = (all_layers * weights).sum(0)  # Weighted average: (B, T, D)
        
        # SpecAugment
        x = self.spec_augment(x)
        
        # Depthwise separable downsampling
        x = x.transpose(1, 2)  # (B, D, T)
        x = self.downsample(x)
        x = x.transpose(1, 2)  # (B, T//4, D)
        
        # Positional encoding + Transformer
        x = self.pos_encoder(x)
        x = self.transformer(x)
        
        # Attentive Statistics Pooling
        x = self.asp(x)  # (B, 2*D)
        
        # Classification
        return self.head(x)
    
    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def save_checkpoint(self, path, **kwargs):
        torch.save({'model_state_dict': self.state_dict(), **kwargs}, path)


# =============================================================================
# Training with SAM
# =============================================================================
def train_epoch_sam(model, loader, optimizer, criterion, device, use_sam=True):
    model.train()
    total_loss = 0
    all_preds, all_labels = [], []
    valid_count = 0
    
    pbar = tqdm(loader, desc="Training")
    for waveforms, labels, lengths in pbar:
        waveforms = waveforms.to(device)
        labels = labels.to(device)
        
        max_len = waveforms.shape[1]
        attention_mask = (torch.arange(max_len, device=device).expand(len(lengths), -1) < lengths.unsqueeze(1).to(device)).long()
        
        if use_sam and isinstance(optimizer, SAM):
            # SAM first step
            logits = model(waveforms, attention_mask=attention_mask)
            loss = criterion(logits, labels)
            
            if torch.isnan(loss) or torch.isinf(loss):
                continue
            
            loss.backward()
            optimizer.first_step(zero_grad=True)
            
            # SAM second step
            logits = model(waveforms, attention_mask=attention_mask)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.second_step(zero_grad=True)
        else:
            optimizer.zero_grad()
            logits = model(waveforms, attention_mask=attention_mask)
            loss = criterion(logits, labels)
            
            if torch.isnan(loss) or torch.isinf(loss):
                continue
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
        
        total_loss += loss.item()
        valid_count += 1
        preds = logits.argmax(dim=1)
        all_preds.extend(preds.detach().cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    acc = accuracy_score(all_labels, all_preds) if all_labels else 0.0
    return total_loss / max(valid_count, 1), acc


@torch.no_grad()
def eval_epoch(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    all_preds, all_labels = [], []
    
    for waveforms, labels, lengths in loader:
        waveforms = waveforms.to(device)
        labels = labels.to(device)
        
        max_len = waveforms.shape[1]
        attention_mask = (torch.arange(max_len, device=device).expand(len(lengths), -1) < lengths.unsqueeze(1).to(device)).long()
        
        logits = model(waveforms, attention_mask=attention_mask)
        loss = criterion(logits, labels)
        
        if not torch.isnan(loss):
            total_loss += loss.item()
        
        preds = logits.argmax(dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
    
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='binary', zero_division=0)
    return total_loss / len(loader), acc, f1


def main():
    parser = argparse.ArgumentParser(description="High-Performance Deepfake Detection")
    parser.add_argument("--csv_dir", type=str, default="unified_dataset")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--freeze_epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--use_sam", action="store_true", help="Use SAM optimizer")
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--output_dir", type=str, default="models")
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    csv_dir = Path(args.csv_dir)
    
    print("\n" + "="*60)
    train_dataset = AugmentedAudioDataset(csv_dir / "train.csv", args.max_samples, augment=True)
    val_dataset = AugmentedAudioDataset(csv_dir / "val.csv", args.max_samples, augment=False)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True, collate_fn=collate_fn)
    
    # Model
    model = HighPerformanceDetector(
        model_name="facebook/wav2vec2-base-960h",
        num_classes=2,
        num_layers=4,
        dropout=0.15
    )
    model = model.to(device)
    
    print(f"\nTrainable Parameters (frozen): {model.count_parameters():,}")
    
    # Optimizer
    classifier_params = [p for n, p in model.named_parameters() if 'wav2vec2' not in n and p.requires_grad]
    
    if args.use_sam:
        print("[Optimizer] Using SAM for better generalization")
        optimizer = SAM(classifier_params, AdamW, lr=args.lr, weight_decay=0.01)
    else:
        optimizer = AdamW(classifier_params, lr=args.lr, weight_decay=0.01)
    
    scheduler = CosineAnnealingWarmRestarts(optimizer if not args.use_sam else optimizer.base_optimizer, T_0=10, T_mult=2)
    # Class weighting for imbalance (Real: 198k, Fake: 120k)
    # Weight for class 1 (fake) should be higher: ~1.65
    class_weights = torch.tensor([1.0, 1.65]).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    print(f"\n{'='*60}")
    print(f"Phase 1: Classifier only (epochs 1-{args.freeze_epochs})")
    print(f"Phase 2: Full fine-tuning (epochs {args.freeze_epochs+1}-{args.epochs})")
    print(f"{'='*60}")
    
    best_acc = 0
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for epoch in range(1, args.epochs + 1):
        if epoch == args.freeze_epochs + 1:
            print("\n" + "="*60)
            print("Phase 2: Unfreezing encoder")
            print("="*60)
            model.unfreeze_encoder()
            
            # Differential Learning Rates
            encoder_params = [p for n, p in model.named_parameters() if 'wav2vec2' in n]
            head_params = [p for n, p in model.named_parameters() if 'wav2vec2' not in n]
            
            param_groups = [
                {'params': encoder_params, 'lr': 1e-5},  # Gentle fine-tuning for backbone
                {'params': head_params, 'lr': args.lr}   # Continue aggressive training for head
            ]

            if args.use_sam:
                optimizer = SAM(param_groups, AdamW, lr=args.lr, weight_decay=0.01)
            else:
                optimizer = AdamW(param_groups, weight_decay=0.01)
            scheduler = CosineAnnealingWarmRestarts(optimizer if not args.use_sam else optimizer.base_optimizer, T_0=10, T_mult=2)
            print(f"Trainable Parameters: {model.count_parameters():,}")
        
        train_loss, train_acc = train_epoch_sam(model, train_loader, optimizer, criterion, device, args.use_sam)
        val_loss, val_acc, val_f1 = eval_epoch(model, val_loader, criterion, device)
        scheduler.step()
        
        phase = "P1" if epoch <= args.freeze_epochs else "P2"
        print(f"[{phase}] Epoch {epoch:3d}/{args.epochs} | "
              f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | Val F1: {val_f1:.4f}")
        
        if val_acc > best_acc:
            best_acc = val_acc
            model.save_checkpoint(output_dir / "high_perf_best.pt", epoch=epoch, val_acc=val_acc, val_f1=val_f1)
            print(f"  → Saved best model (val_acc: {val_acc:.4f})")
    
    print("\n" + "="*60)
    print(f"Training Complete! Best Val Acc: {best_acc:.4f}")
    print("="*60)


if __name__ == "__main__":
    main()
