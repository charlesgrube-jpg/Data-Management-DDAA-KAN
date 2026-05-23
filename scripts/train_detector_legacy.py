#!/usr/bin/env python3
"""
Train Defense Models for Deepfake Audio Detection

Usage:
    python scripts/train_detector.py --model transformer --epochs 50
    python scripts/train_detector.py --model kan --data_dir ./completed_correct_dataset
"""

import argparse
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

# pipeline.models is not implemented; model classes are defined inline in train_detector_csv.py
# get_detector is not used in this file's inline training logic
from pipeline.features.cqt_extractor import CQTExtractor


class AudioFeatureDataset(Dataset):
    """
    Dataset for CQT features extracted from audio files.
    
    Expects directory structure:
        data_dir/
            train/
                real/
                    audio1.wav, audio2.wav, ...
                synthetic/
                    audio1.wav, audio2.wav, ...
            val/
                real/, synthetic/
            test/
                real/, synthetic/
    """
    
    def __init__(
        self,
        data_dir: str,
        split: str = "train",
        feature_extractor: CQTExtractor = None,
        max_samples: int = None,
        cache_features: bool = True
    ):
        self.data_dir = Path(data_dir)
        self.split = split
        self.feature_extractor = feature_extractor or CQTExtractor()
        self.cache_features = cache_features
        
        self.samples = []
        self._feature_cache = {}
        
        # Load file paths
        split_dir = self.data_dir / split
        
        for label, class_name in [(0, "real"), (1, "synthetic")]:
            class_dir = split_dir / class_name
            if class_dir.exists():
                audio_files = list(class_dir.glob("*.wav"))
                for f in audio_files:
                    self.samples.append((str(f), label))
        
        # Shuffle and optionally limit samples
        random.shuffle(self.samples)
        if max_samples:
            self.samples = self.samples[:max_samples]
        
        print(f"[Dataset] {split}: {len(self.samples)} samples")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        file_path, label = self.samples[idx]
        
        # Check cache
        if self.cache_features and file_path in self._feature_cache:
            features = self._feature_cache[file_path]
        else:
            # Extract features
            features = self.feature_extractor.extract_file(file_path)
            
            if self.cache_features:
                self._feature_cache[file_path] = features
        
        # Convert to tensor
        features = torch.from_numpy(features).float()
        
        # Ensure consistent shape (pad/truncate time dimension)
        target_time = 100  # ~3 seconds at 16kHz with hop_length=512
        if features.shape[1] < target_time:
            pad_width = target_time - features.shape[1]
            features = torch.nn.functional.pad(features, (0, pad_width))
        elif features.shape[1] > target_time:
            features = features[:, :target_time]
        
        return features, label


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    use_physics_loss: bool = False
) -> dict:
    """Train for one epoch."""
    model.train()
    
    total_loss = 0
    correct = 0
    total = 0
    
    for features, labels in tqdm(dataloader, desc="Training", leave=False):
        features = features.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        
        # Forward pass
        if use_physics_loss and hasattr(model, 'training_step'):
            loss, loss_dict = model.training_step(features, labels, criterion)
            # Get logits for accuracy tracking (call forward without physics loss)
            with torch.no_grad():
                logits = model(features)
        else:
            logits = model(features)
            loss = criterion(logits, labels)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Track metrics
        total_loss += loss.item() * features.size(0)
        
        with torch.no_grad():
            preds = logits.argmax(dim=-1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    
    return {
        'loss': total_loss / total,
        'accuracy': correct / total
    }


def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device
) -> dict:
    """Evaluate model on validation/test set."""
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
    
    # Compute additional metrics
    try:
        from sklearn.metrics import f1_score, precision_score, recall_score
        f1 = f1_score(all_labels, all_preds, average='binary')
        precision = precision_score(all_labels, all_preds, average='binary')
        recall = recall_score(all_labels, all_preds, average='binary')
    except ImportError:
        f1 = precision = recall = 0.0
    
    return {
        'loss': total_loss / total,
        'accuracy': correct / total,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }


def main():
    parser = argparse.ArgumentParser(description="Train deepfake audio detector")
    
    # Model
    parser.add_argument("--model", type=str, default="transformer",
                       choices=["transformer", "kan", "neural_ode", "pinn"],
                       help="Model architecture")
    
    # Data
    parser.add_argument("--data_dir", type=str, default="./completed_correct_dataset",
                       help="Dataset directory")
    parser.add_argument("--max_samples", type=int, default=None,
                       help="Maximum samples per split (for debugging)")
    
    # Training
    parser.add_argument("--epochs", type=int, default=50,
                       help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32,
                       help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4,
                       help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-5,
                       help="Weight decay")
    
    # Model-specific
    parser.add_argument("--hidden_dim", type=int, default=256,
                       help="Hidden dimension")
    parser.add_argument("--num_layers", type=int, default=4,
                       help="Number of layers")
    
    # Output
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints",
                       help="Checkpoint directory")
    parser.add_argument("--exp_name", type=str, default=None,
                       help="Experiment name (default: auto-generated)")
    
    # Other
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed")
    parser.add_argument("--device", type=str, default="auto",
                       help="Device (auto, cpu, cuda)")
    parser.add_argument("--num_workers", type=int, default=4,
                       help="Number of data loading workers")
    parser.add_argument("--pin_memory", action="store_true",
                       help="Pin memory for faster GPU transfer")
    
    args = parser.parse_args()
    
    # Set seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    # Device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"Using device: {device}")
    
    # Create feature extractor
    feature_extractor = CQTExtractor()
    
    # Create datasets
    train_dataset = AudioFeatureDataset(
        args.data_dir, "train", feature_extractor, args.max_samples
    )
    val_dataset = AudioFeatureDataset(
        args.data_dir, "val", feature_extractor, args.max_samples
    )
    test_dataset = AudioFeatureDataset(
        args.data_dir, "test", feature_extractor, args.max_samples
    )
    
    # Create dataloaders (optimized for cluster)
    pin_memory = args.pin_memory and torch.cuda.is_available()
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        persistent_workers=args.num_workers > 0
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        persistent_workers=args.num_workers > 0
    )
    test_loader = DataLoader(
        test_dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        persistent_workers=args.num_workers > 0
    )
    
    # Create model
    model_kwargs = {
        "n_bins": 84,
        "hidden_dim": args.hidden_dim,
        "num_classes": 2
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
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    # Setup checkpointing
    exp_name = args.exp_name or f"{args.model}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    checkpoint_dir = Path(args.checkpoint_dir) / exp_name
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # Training loop
    best_val_acc = 0
    use_physics = (args.model == "pinn")
    
    print(f"\nTraining for {args.epochs} epochs...")
    print("-" * 60)
    
    for epoch in range(args.epochs):
        # Train
        train_metrics = train_epoch(
            model, train_loader, optimizer, criterion, device, use_physics
        )
        
        # Validate
        val_metrics = evaluate(model, val_loader, criterion, device)
        
        scheduler.step()
        
        # Print progress
        print(f"Epoch {epoch+1:3d}/{args.epochs} | "
              f"Train Loss: {train_metrics['loss']:.4f} | "
              f"Train Acc: {train_metrics['accuracy']:.4f} | "
              f"Val Loss: {val_metrics['loss']:.4f} | "
              f"Val Acc: {val_metrics['accuracy']:.4f} | "
              f"Val F1: {val_metrics['f1']:.4f}")
        
        # Save best model
        if val_metrics['accuracy'] > best_val_acc:
            best_val_acc = val_metrics['accuracy']
            model.save_checkpoint(
                str(checkpoint_dir / "best_model.pt"),
                optimizer=optimizer,
                epoch=epoch,
                val_accuracy=best_val_acc
            )
            print(f"  → Saved best model (val_acc: {best_val_acc:.4f})")
    
    # Test evaluation
    print("\n" + "=" * 60)
    print("Final Test Evaluation:")
    test_metrics = evaluate(model, test_loader, criterion, device)
    print(f"  Accuracy:  {test_metrics['accuracy']:.4f}")
    print(f"  F1 Score:  {test_metrics['f1']:.4f}")
    print(f"  Precision: {test_metrics['precision']:.4f}")
    print(f"  Recall:    {test_metrics['recall']:.4f}")
    
    # Save final model
    model.save_checkpoint(
        str(checkpoint_dir / "final_model.pt"),
        optimizer=optimizer,
        epoch=args.epochs,
        test_metrics=test_metrics
    )
    
    print(f"\nCheckpoints saved to: {checkpoint_dir}")


if __name__ == "__main__":
    main()
