"""
Exporter Module

Saves processed dataset to disk:
- Audio files organized by split and label
- Comprehensive metadata CSV
- Manifest with checksums for verification
"""

import os
import csv
import json
import shutil
import hashlib
import numpy as np
import soundfile as sf
from pathlib import Path
from typing import Dict, List, Any
from datetime import datetime
from pipeline.config import Config


def export_dataset(
    samples: List[Dict[str, Any]],
    config: Config,
    config_path: str = "config.yaml"
) -> Dict[str, Any]:
    """
    Export all samples to disk with organized structure.
    
    Structure:
        output_dir/
        ├── train/
        │   ├── real/
        │   └── synthetic/
        ├── val/
        │   ├── real/
        │   └── synthetic/
        ├── test/
        │   ├── real/
        │   └── synthetic/
        ├── metadata.csv
        ├── config.yaml (copy)
        └── manifest.json
    
    Returns:
        Export statistics
    """
    output_path = config.output_path
    sr = config.audio.target_sr
    
    # Create directory structure
    for split in ["train", "val", "test"]:
        for label in ["real", "synthetic"]:
            (output_path / split / label).mkdir(parents=True, exist_ok=True)
    
    print(f"[Exporter] Exporting to {output_path}")
    
    # Copy config to output directory (for reproducibility)
    if Path(config_path).exists():
        shutil.copy(config_path, output_path / "config.yaml")
        print(f"[Exporter] Copied config to {output_path / 'config.yaml'}")
    
    # Track metadata and stats
    metadata_rows = []
    stats = {"train": 0, "val": 0, "test": 0, "total": 0, "errors": 0}
    manifest = {"files": [], "export_date": datetime.now().isoformat()}
    
    for idx, sample in enumerate(samples):
        try:
            # Determine path
            split = sample.get("split", "train")
            is_synthetic = sample.get("is_synthetic", False)
            label_dir = "synthetic" if is_synthetic else "real"
            
            # Generate filename
            source_id = sample.get("source_id", f"src_{idx:06d}")
            chunk_idx = sample.get("chunk_idx", 0)
            filename = f"{source_id}_chunk{chunk_idx:03d}.wav"
            
            rel_path = f"{split}/{label_dir}/{filename}"
            full_path = output_path / rel_path
            
            # Save audio
            audio = sample.get("audio")
            if audio is None:
                stats["errors"] += 1
                continue
            
            sf.write(str(full_path), audio, sr, subtype='PCM_16')
            
            # Compute checksum
            file_hash = compute_file_hash(full_path)
            
            # Build metadata row (exclude audio array)
            row = {
                "filename": rel_path,
                "split": split,
                "label": "synthetic" if is_synthetic else "real",
                "is_synthetic": is_synthetic,
                "source_id": source_id,
                "chunk_idx": chunk_idx,
                "speaker_id": sample.get("speaker_id", ""),
                "transcript": sample.get("transcript", ""),
                "gender": sample.get("gender", ""),
                "accent": sample.get("accent", ""),
                "duration": sample.get("duration", len(audio) / sr),
                "quality_tier": sample.get("quality_tier", "clean"),
                "generator": sample.get("generator", ""),
                "method": sample.get("method", ""),
                # New fields for complete tracking
                "attack_category": sample.get("method", "") if is_synthetic else "",  # tts/vc
                "source_real_file": sample.get("source_real_file", ""),  # Link to real source
                "snr_db": sample.get("snr_db", ""),  # SNR when noise applied
            }
            metadata_rows.append(row)
            
            # Add to manifest
            manifest["files"].append({
                "path": rel_path,
                "sha256": file_hash,
                "size": os.path.getsize(full_path)
            })
            
            stats[split] += 1
            stats["total"] += 1
            
            if stats["total"] % 500 == 0:
                print(f"[Exporter] Exported {stats['total']} samples...")
                
        except Exception as e:
            print(f"[Exporter] Error exporting sample {idx}: {e}")
            stats["errors"] += 1
    
    # Save metadata CSV
    metadata_path = output_path / config.output.metadata_file
    save_metadata_csv(metadata_rows, metadata_path)
    print(f"[Exporter] Saved metadata to {metadata_path}")
    
    # Save manifest
    manifest["stats"] = stats
    manifest_path = output_path / "manifest.json"
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)
    print(f"[Exporter] Saved manifest to {manifest_path}")
    
    # Save split files (convenience)
    for split in ["train", "val", "test"]:
        split_rows = [r for r in metadata_rows if r["split"] == split]
        split_path = output_path / f"{split}.csv"
        save_metadata_csv(split_rows, split_path)
    
    print(f"[Exporter] Export complete: {stats['total']} samples, {stats['errors']} errors")
    
    return stats


def save_metadata_csv(rows: List[Dict], path: Path):
    """Save metadata rows to CSV file."""
    if not rows:
        return
    
    fieldnames = list(rows[0].keys())
    
    with open(path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def compute_file_hash(path: Path) -> str:
    """Compute SHA256 hash of file."""
    sha256 = hashlib.sha256()
    
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            sha256.update(chunk)
    
    return sha256.hexdigest()


def load_metadata(dataset_path: str) -> List[Dict[str, Any]]:
    """
    Load metadata from exported dataset.
    
    Args:
        dataset_path: Path to dataset directory
        
    Returns:
        List of metadata dictionaries
    """
    metadata_path = Path(dataset_path) / "metadata.csv"
    
    with open(metadata_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        return list(reader)


def verify_export(dataset_path: str) -> Dict[str, Any]:
    """
    Verify exported dataset integrity using manifest.
    
    Returns:
        Verification results
    """
    dataset_path = Path(dataset_path)
    manifest_path = dataset_path / "manifest.json"
    
    with open(manifest_path, 'r') as f:
        manifest = json.load(f)
    
    results = {"passed": True, "checked": 0, "failed": 0, "missing": 0}
    
    for file_info in manifest["files"]:
        file_path = dataset_path / file_info["path"]
        
        if not file_path.exists():
            results["missing"] += 1
            results["passed"] = False
            continue
        
        actual_hash = compute_file_hash(file_path)
        if actual_hash != file_info["sha256"]:
            results["failed"] += 1
            results["passed"] = False
            continue
        
        results["checked"] += 1
    
    return results


def generate_dataset_card(
    config: Config,
    stats: Dict[str, Any],
    output_path: Path
) -> str:
    """
    Generate a README dataset card for documentation.
    """
    card = f"""# Audio Deepfake Detection Dataset

## Overview
- **Generated**: {datetime.now().strftime("%Y-%m-%d %H:%M")}
- **Source**: {config.source.dataset_name}
- **Language**: {config.source.language}
- **Total Samples**: {stats.get('total', 'N/A')}

## Splits
| Split | Samples |
|-------|---------|
| Train | {stats.get('train', 0)} |
| Val | {stats.get('val', 0)} |
| Test | {stats.get('test', 0)} |

## Audio Specifications
- **Sample Rate**: {config.audio.target_sr} Hz
- **Chunk Duration**: {config.segmentation.chunk_duration}s
- **Normalization**: RMS to {config.audio.normalize_db} dBFS

## Synthesis Methods
- **TTS Models**: {', '.join(config.synthesis.tts_models)}
- **VC Models**: {', '.join(config.synthesis.vc_models)}

## Quality Tiers
{json.dumps(config.effects.quality_distribution, indent=2)}

## File Structure
```
{config.output.base_dir}/
├── train/
│   ├── real/
│   └── synthetic/
├── val/
├── test/
├── metadata.csv
└── manifest.json
```

## Metadata Fields
- `filename`: Relative path to audio file
- `split`: train/val/test
- `label`: real/synthetic
- `speaker_id`: Unique speaker identifier
- `transcript`: Text content
- `gender`: Speaker gender
- `accent`: Speaker accent
- `duration`: Audio duration in seconds
- `quality_tier`: clean/mobile/noisy
- `generator`: Synthesis model used (synthetic only)
- `method`: tts/vc (synthetic only)

## Usage
```python
import pandas as pd
import librosa

# Load metadata
df = pd.read_csv("metadata.csv")

# Load audio
audio, sr = librosa.load(df.iloc[0]["filename"], sr=16000)
```

## Citation
[Add citation here]
"""
    
    card_path = output_path / "README.md"
    with open(card_path, 'w') as f:
        f.write(card)
    
    return card


if __name__ == "__main__":
    from pipeline.config import load_config
    
    cfg = load_config("../config.yaml")
    
    # Test with mock data
    mock_samples = []
    for i in range(10):
        mock_samples.append({
            "audio": np.random.randn(48000) * 0.1,
            "source_id": f"mock_{i:03d}",
            "chunk_idx": 0,
            "speaker_id": f"speaker_{i % 3}",
            "transcript": f"Test sentence {i}",
            "is_synthetic": i % 2 == 0,
            "split": ["train", "val", "test"][i % 3],
            "quality_tier": "clean"
        })
    
    stats = export_dataset(mock_samples, cfg)
    print(f"Exported: {stats}")
