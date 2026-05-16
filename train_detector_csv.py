#!/usr/bin/env python3
"""
Train Defense Models for Deepfake Audio Detection (CSV Version)

Uses the unified_dataset CSVs created by prepare_training_data.py

Usage:
    python scripts/train_detector_csv.py --model transformer --epochs 50
    python scripts/train_detector_csv.py --model kan --csv_dir unified_dataset
"""

import argparse
import csv
import os
import sys
import random
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

try:
    from pipeline.models import get_detector
except ImportError:
    print("Warning: Could not import pipeline.models, using inline definitions")
    get_detector = None


# ============== INLINE MODEL DEFINITIONS (FALLBACK) ==============
# Used when pipeline.models is unavailable or needs SSL support

class BaseDetector(nn.Module):
    """Base class for all detectors."""
    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class TransformerDetector(BaseDetector):
    """Transformer-based audio deepfake detector."""
    
    def __init__(self, n_bins=84, hidden_dim=256, num_layers=4, num_classes=2, input_dim=None, **kwargs):
        super().__init__()
        self.input_dim = input_dim or (n_bins * 100)
        
        # Input projection
        self.input_proj = nn.Linear(self.input_dim, hidden_dim)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=8, dim_feedforward=hidden_dim*4, 
            dropout=0.1, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, num_classes)
        )
    
    def forward(self, x, **kwargs):
        # Flatten if needed: (B, 84, 100) -> (B, 8400) or (B, 768) for SSL
        if x.dim() > 2:
            x = x.flatten(1)
        
        x = self.input_proj(x).unsqueeze(1)  # (B, 1, hidden_dim)
        x = self.transformer(x)
        x = x.mean(dim=1)  # Global average pooling
        return self.classifier(x)


class KANDetector(BaseDetector):
    """KAN-based audio deepfake detector (simplified)."""
    
    def __init__(self, n_bins=84, hidden_dim=256, hidden_dims=None, num_classes=2, input_dim=None, **kwargs):
        super().__init__()
        self.input_dim = input_dim or (n_bins * 100)
        
        if hidden_dims is None:
            hidden_dims = [hidden_dim, hidden_dim // 2, hidden_dim // 4]
        
        layers = []
        in_dim = self.input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, h_dim))
            layers.append(nn.SiLU())  # KAN-like smooth activation
            layers.append(nn.LayerNorm(h_dim))
            in_dim = h_dim
        
        self.backbone = nn.Sequential(*layers)
        self.classifier = nn.Linear(hidden_dims[-1], num_classes)
    
    def forward(self, x, **kwargs):
        if x.dim() > 2:
            x = x.flatten(1)
        x = self.backbone(x)
        return self.classifier(x)


class NeuralODEDetector(BaseDetector):
    """Neural ODE-based audio deepfake detector (simplified)."""
    
    def __init__(self, n_bins=84, hidden_dim=256, num_ode_blocks=4, num_classes=2, input_dim=None, **kwargs):
        super().__init__()
        self.input_dim = input_dim or (n_bins * 100)
        
        self.input_proj = nn.Linear(self.input_dim, hidden_dim)
        
        # Simplified ODE blocks (residual connections simulate continuous dynamics)
        blocks = []
        for _ in range(num_ode_blocks):
            blocks.append(nn.Sequential(
                nn.LayerNorm(hidden_dim),
                nn.Linear(hidden_dim, hidden_dim),
                nn.Tanh(),
                nn.Linear(hidden_dim, hidden_dim)
            ))
        self.ode_blocks = nn.ModuleList(blocks)
        
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, num_classes)
        )
    
    def forward(self, x, **kwargs):
        if x.dim() > 2:
            x = x.flatten(1)
        x = self.input_proj(x)
        
        # Residual ODE-like dynamics
        for block in self.ode_blocks:
            x = x + 0.1 * block(x)  # Euler step
        
        return self.classifier(x)


class PINNDetector(BaseDetector):
    """Physics-Informed Neural Network detector (simplified)."""
    
    def __init__(self, n_bins=84, hidden_dim=256, hidden_dims=None, num_classes=2, input_dim=None, **kwargs):
        super().__init__()
        self.input_dim = input_dim or (n_bins * 100)
        
        if hidden_dims is None:
            hidden_dims = [hidden_dim, hidden_dim // 2, hidden_dim // 4]
        
        layers = []
        in_dim = self.input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, h_dim))
            layers.append(nn.Tanh())  # Smooth for physics
            in_dim = h_dim
        
        self.backbone = nn.Sequential(*layers)
        self.classifier = nn.Linear(hidden_dims[-1], num_classes)
        
        # Physics branch (simplified)
        self.physics_head = nn.Linear(hidden_dims[-1], 1)
    
    def forward(self, x, return_physics=False, **kwargs):
        if x.dim() > 2:
            x = x.flatten(1)
        features = self.backbone(x)
        logits = self.classifier(features)
        
        if return_physics:
            physics = self.physics_head(features)
            return logits, physics
        return logits


def get_detector_inline(model_name, **kwargs):
    """Factory function for creating detectors."""
    models = {
        'transformer': TransformerDetector,
        'kan': KANDetector,
        'neural_ode': NeuralODEDetector,
        'pinn': PINNDetector
    }
    if model_name not in models:
        raise ValueError(f"Unknown model: {model_name}. Available: {list(models.keys())}")
    return models[model_name](**kwargs)


# Use inline if pipeline.models not available
if get_detector is None:
    get_detector = get_detector_inline
# ============== END INLINE MODEL DEFINITIONS ==============

try:
    from pipeline.features.cqt_extractor import CQTExtractor
except ImportError:
    # Fallback CQT extractor
    import torchaudio
    
    class CQTExtractor:
        def __init__(self, sample_rate=16000, n_bins=84, bins_per_octave=12, 
                     hop_length=512, f_min=32.7):
            self.sample_rate = sample_rate
            self.n_bins = n_bins
            self.hop_length = hop_length
            self.f_min = f_min
            
        def extract_file(self, file_path):
            waveform, sr = torchaudio.load(file_path)
            if sr != self.sample_rate:
                resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
                waveform = resampler(waveform)
            
            # Simple mel spectrogram as fallback
            mel = torchaudio.transforms.MelSpectrogram(
                sample_rate=self.sample_rate,
                n_mels=self.n_bins,
                hop_length=self.hop_length,
                f_min=self.f_min
            )
            features = mel(waveform).squeeze(0).numpy()
            features = np.log(features + 1e-9)
            return features


class BundleDataset(Dataset):
    """
    Dataset that loads from pre-bundled .pt files.
    Loads entire dataset in ~30 seconds instead of 40+ minutes.
    """
    
    def __init__(self, bundle_path: str):
        self.bundle_path = Path(bundle_path)
        
        print(f"[Bundle] Loading {self.bundle_path.name}...")
        bundle = torch.load(self.bundle_path, weights_only=True)
        self.features = bundle['features']
        self.labels = bundle['labels']
        
        size_gb = self.bundle_path.stat().st_size / 1e9
        print(f"[Bundle] Loaded {len(self.labels)} samples ({size_gb:.2f} GB)")
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx].item()


class CSVAudioDataset(Dataset):
    """
    Dataset that loads from unified CSV files.
    """
    
    def __init__(
        self,
        csv_path: str,
        feature_extractor: CQTExtractor = None,
        max_samples: int = None,
        target_time: int = 100
    ):
        self.csv_path = Path(csv_path)
        self.feature_extractor = feature_extractor or CQTExtractor()
        self.target_time = target_time
        
        self.samples = []
        self.preloaded_features = []
        self.preloaded_labels = []
        self.is_preloaded = False
        
        # Load from CSV
        with open(self.csv_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Check for pre-extracted feature path first
                if 'feature_path' in row:
                    file_path = row['feature_path']
                    is_preextracted = True
                else:
                    # Fallback to audio path
                    file_path = row.get('full_path', row.get('filename', ''))
                    is_preextracted = False

                label_str = row.get('label', 'real')
                # Handle both string labels and numeric (from manifest)
                if label_str == 'synthetic' or str(label_str) == '1':
                    label = 1
                else:
                    label = 0
                
                if file_path:
                    # For pre-extracted, we assume path implies existence or we check it
                    if is_preextracted or Path(file_path).exists():
                        self.samples.append((file_path, label, is_preextracted))
        
        # Shuffle and limit
        random.shuffle(self.samples)
        if max_samples:
            self.samples = self.samples[:max_samples]
        
        print(f"[CSV Dataset] {self.csv_path.name}: {len(self.samples)} samples")
        
        # PRE-LOAD ALL FEATURES INTO RAM for maximum GPU utilization
        if len(self.samples) > 0 and self.samples[0][2]:  # If using pre-extracted
            print(f"[CSV Dataset] Pre-loading {len(self.samples)} features into RAM...")
            for file_path, label, is_preextracted in tqdm(self.samples, desc="Loading to RAM", leave=False):
                try:
                    features = np.load(file_path)
                    features = torch.from_numpy(features).float()
                    
                    # Ensure consistent shape
                    if features.shape[-1] < self.target_time:
                        pad_width = self.target_time - features.shape[-1]
                        features = torch.nn.functional.pad(features, (0, pad_width))
                    elif features.shape[-1] > self.target_time:
                        features = features[..., :self.target_time]
                    
                    self.preloaded_features.append(features)
                    self.preloaded_labels.append(label)
                except Exception as e:
                    # Skip bad files
                    continue
            
            # Stack into tensors for fast indexing
            if self.preloaded_features:
                self.preloaded_features = torch.stack(self.preloaded_features)
                self.preloaded_labels = torch.tensor(self.preloaded_labels, dtype=torch.long)
                self.is_preloaded = True
                print(f"[CSV Dataset] Loaded {len(self.preloaded_labels)} samples into RAM ({self.preloaded_features.nbytes / 1e9:.2f} GB)")
    
    def __len__(self):
        if self.is_preloaded:
            return len(self.preloaded_labels)
        return len(self.samples)
    
    def __getitem__(self, idx):
        if self.is_preloaded:
            # Fast path: already in RAM as tensor
            return self.preloaded_features[idx], self.preloaded_labels[idx].item()
        
        # Slow path: load from disk
        file_path, label, is_preextracted = self.samples[idx]
        
        try:
            if is_preextracted:
                # Load .npy feature file directly
                features = np.load(file_path)
            else:
                # Extract from audio on the fly
                features = self.feature_extractor.extract_file(file_path)
                
            features = torch.from_numpy(features).float()
            
            # Ensure consistent shape
            if features.shape[-1] < self.target_time:
                pad_width = self.target_time - features.shape[-1]
                features = torch.nn.functional.pad(features, (0, pad_width))
            elif features.shape[-1] > self.target_time:
                features = features[..., :self.target_time]
            
            return features, label
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            # Return zero features on error
            features = torch.zeros(84, self.target_time)
            return features, label


def train_epoch(model, dataloader, optimizer, criterion, device, use_physics=False):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for features, labels in tqdm(dataloader, desc="Training", leave=False):
        features = features.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        
        if use_physics and hasattr(model, 'training_step'):
            loss, _ = model.training_step(features, labels, criterion)
            with torch.no_grad():
                logits = model(features)
        else:
            logits = model(features)
            loss = criterion(logits, labels)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += loss.item() * features.size(0)
        preds = logits.argmax(dim=-1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    
    return {'loss': total_loss / total, 'accuracy': correct / total}


def evaluate(model, dataloader, criterion, device):
    """Evaluate model."""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for features, labels in tqdm(dataloader, desc="Evaluating", leave=False):
            features = features.to(device)
            labels = labels.to(device)
            
            logits = model(features)
            loss = criterion(logits, labels)
            
            total_loss += loss.item() * features.size(0)
            preds = logits.argmax(dim=-1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())
    
    # Compute metrics
    try:
        from sklearn.metrics import f1_score, precision_score, recall_score
        f1 = f1_score(all_labels, all_preds, average='binary')
        precision = precision_score(all_labels, all_preds, average='binary')
        recall = recall_score(all_labels, all_preds, average='binary')
    except:
        f1 = precision = recall = 0.0
    
    return {
        'loss': total_loss / total,
        'accuracy': correct / total,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }


def main():
    parser = argparse.ArgumentParser(description="Train deepfake detector from CSV")
    
    parser.add_argument("--model", type=str, default="transformer",
                       choices=["transformer", "kan", "neural_ode", "pinn"])
    parser.add_argument("--csv_dir", type=str, default="unified_dataset",
                       help="Directory with train.csv, val.csv, test.csv")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=4)
    
    args = parser.parse_args()
    
    # Set seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # CSV paths
    # CSV paths - Prefer feature-extracted CSVs if available
    csv_dir = Path(args.csv_dir)
    
    def get_best_csv(base_name):
        feat_csv = csv_dir / f"{base_name}_with_features.csv"
        normal_csv = csv_dir / f"{base_name}.csv"
        if feat_csv.exists():
            print(f"[{base_name}] Found pre-extracted features CSV: {feat_csv.name}")
            return feat_csv
        return normal_csv

    train_csv = get_best_csv("train")
    val_csv = get_best_csv("val")
    test_csv = get_best_csv("test")
    
    # Check for SSL bundled .pt files (PREFERRED - SOTA features!)
    train_ssl_bundle = csv_dir / "train_ssl_bundle.pt"
    val_ssl_bundle = csv_dir / "val_ssl_bundle.pt"
    test_ssl_bundle = csv_dir / "test_ssl_bundle.pt"
    
    # Check for CQT bundled .pt files (fast loading)
    train_bundle = csv_dir / "train_bundle.pt"
    val_bundle = csv_dir / "val_bundle.pt"
    test_bundle = csv_dir / "test_bundle.pt"
    
    feature_dim = 8400  # Default CQT: 84 * 100
    feature_type = "cqt"
    
    if train_ssl_bundle.exists() and val_ssl_bundle.exists() and test_ssl_bundle.exists():
        print("[MODE] Using SSL bundled .pt files (SOTA features!)")
        train_dataset = BundleDataset(train_ssl_bundle)
        val_dataset = BundleDataset(val_ssl_bundle)
        test_dataset = BundleDataset(test_ssl_bundle)
        feature_dim = train_dataset.features.shape[-1]  # 768 for wav2vec2-base
        feature_type = "ssl"
        print(f"[SSL] Feature dimension: {feature_dim}")
    elif train_bundle.exists() and val_bundle.exists() and test_bundle.exists():
        print("[MODE] Using CQT bundled .pt files (instant loading!)")
        train_dataset = BundleDataset(train_bundle)
        val_dataset = BundleDataset(val_bundle)
        test_dataset = BundleDataset(test_bundle)
    else:
        print("[MODE] Using CSV files (slower loading...)")
        # Feature extractor (force CPU to avoid CUDA fork issues, model uses GPU)
        feature_extractor = CQTExtractor(device="cpu")
        
        # Datasets
        train_dataset = CSVAudioDataset(train_csv, feature_extractor, args.max_samples)
        val_dataset = CSVAudioDataset(val_csv, feature_extractor, args.max_samples)
        test_dataset = CSVAudioDataset(test_csv, feature_extractor, args.max_samples)
    
    # Dataloaders (num_workers=0 to avoid CUDA fork issues with nnAudio GPU)
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=0, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=0, pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=0, pin_memory=True
    )
    
    # Create model
    if get_detector is None:
        print("ERROR: Cannot import get_detector. Check pipeline/models.")
        sys.exit(1)
    
    model_kwargs = {
        "n_bins": 84 if feature_type == "cqt" else 1,  # 1 for SSL (no bins, just embedding)
        "hidden_dim": args.hidden_dim,
        "num_classes": 2,
        "input_dim": feature_dim  # 8400 for CQT, 768 for SSL
    }
    
    if args.model == "transformer":
        model_kwargs["num_layers"] = args.num_layers
    elif args.model == "kan":
        model_kwargs["hidden_dims"] = [args.hidden_dim, args.hidden_dim // 2, args.hidden_dim // 4]
    elif args.model == "neural_ode":
        model_kwargs["num_ode_blocks"] = args.num_layers
    elif args.model == "pinn":
        model_kwargs["hidden_dims"] = [args.hidden_dim, args.hidden_dim // 2, args.hidden_dim // 4]
    
    model = get_detector(args.model, **model_kwargs)
    model = model.to(device)
    
    print(f"\nModel: {args.model}")
    print(f"Parameters: {model.count_parameters():,}")
    
    # Training setup with class weighting for imbalanced data
    # Count class distribution from training set
    if hasattr(train_dataset, 'labels'):
        labels = train_dataset.labels
    elif hasattr(train_dataset, 'preloaded_labels'):
        labels = train_dataset.preloaded_labels
    else:
        # Fallback: assume balanced
        labels = torch.tensor([0, 1])
    
    class_counts = torch.bincount(labels.long())
    total = class_counts.sum().float()
    class_weights = total / (len(class_counts) * class_counts.float())
    class_weights = class_weights.to(device)
    print(f"Class distribution: {class_counts.tolist()} -> Weights: {class_weights.tolist()}")
    
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    # Checkpointing
    exp_name = f"{args.model}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    checkpoint_dir = Path(args.checkpoint_dir) / exp_name
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # Training loop
    best_val_acc = 0
    use_physics = (args.model == "pinn")
    
    print(f"\nTraining {args.model} for {args.epochs} epochs...")
    print("=" * 60)
    
    for epoch in range(args.epochs):
        train_metrics = train_epoch(model, train_loader, optimizer, criterion, device, use_physics)
        val_metrics = evaluate(model, val_loader, criterion, device)
        scheduler.step()
        
        print(f"Epoch {epoch+1:3d}/{args.epochs} | "
              f"Train Loss: {train_metrics['loss']:.4f} | "
              f"Train Acc: {train_metrics['accuracy']:.4f} | "
              f"Val Loss: {val_metrics['loss']:.4f} | "
              f"Val Acc: {val_metrics['accuracy']:.4f} | "
              f"Val F1: {val_metrics['f1']:.4f}")
        
        if val_metrics['accuracy'] > best_val_acc:
            best_val_acc = val_metrics['accuracy']
            model.save_checkpoint(
                str(checkpoint_dir / "best_model.pt"),
                optimizer=optimizer,
                epoch=epoch,
                val_accuracy=best_val_acc
            )
            print(f"  → Saved best model (val_acc: {best_val_acc:.4f})")
    
    # Final test
    print("\n" + "=" * 60)
    print("Final Test Evaluation:")
    test_metrics = evaluate(model, test_loader, criterion, device)
    print(f"  Accuracy:  {test_metrics['accuracy']:.4f}")
    print(f"  F1 Score:  {test_metrics['f1']:.4f}")
    print(f"  Precision: {test_metrics['precision']:.4f}")
    print(f"  Recall:    {test_metrics['recall']:.4f}")
    
    # Save final
    model.save_checkpoint(
        str(checkpoint_dir / "final_model.pt"),
        optimizer=optimizer,
        epoch=args.epochs,
        test_metrics=test_metrics
    )
    
    print(f"\nCheckpoints saved to: {checkpoint_dir}")
    
    # Save results to file
    results_file = checkpoint_dir / "results.txt"
    with open(results_file, 'w') as f:
        f.write(f"Model: {args.model}\n")
        f.write(f"Best Val Accuracy: {best_val_acc:.4f}\n")
        f.write(f"Test Accuracy: {test_metrics['accuracy']:.4f}\n")
        f.write(f"Test F1: {test_metrics['f1']:.4f}\n")
        f.write(f"Test Precision: {test_metrics['precision']:.4f}\n")
        f.write(f"Test Recall: {test_metrics['recall']:.4f}\n")
    

if __name__ == "__main__":
    main()
