"""
External Dataset Downloader and Ingester 🌐

Downloads external deepfake datasets and prepares them for the pipeline.
Crucially, validates and assigns strict 'split' labels (e.g., 'test') to ensure
no leakage into training data.

Supported Datasets:
1. ElevenLabs (Source: Hugging Face 'skypro1111/elevenlabs_dataset') -> Split: TEST
2. ASVspoof (Optional/Placeholder) -> Split: TEST
"""

import os
import json
import shutil
import zipfile
from pathlib import Path
from typing import List, Dict, Any
from tqdm import tqdm
import soundfile as sf

# Configuration
EXTERNAL_DIR = Path("./external_data")
METADATA_FILE = EXTERNAL_DIR / "metadata_external.json"

DATASETS = {
    "elevenlabs": {
        "source": "huggingface",
        "repo_id": "skypro1111/elevenlabs_dataset",
        "split_assignment": "test",
        "subdir": "elevenlabs",
        "method_label": "elevenlabs"
    }
}

def download_huggingface_dataset(repo_id: str, local_dir: Path):
    """Download dataset snapshot from Hugging Face Hub."""
    print(f"[External] Downloading {repo_id} from Hugging Face...")
    try:
        from huggingface_hub import snapshot_download
        # Download strictly the audio files if possible, or the whole thing
        # This dataset seems to be flat or zipped. Let's grab everything.
        snapshot_download(
            repo_id=repo_id,
            local_dir=local_dir,
            repo_type="dataset",
            allow_patterns=["*.wav", "*.mp3", "*.flac", "*.zip"],
            ignore_patterns=[".gitattributes", "README.md"]
        )
        print(f"[External] Download complete: {local_dir}")
        
        # Unzip any zip files
        for zip_file in local_dir.glob("*.zip"):
            print(f"[External] Extracting {zip_file.name}...")
            with zipfile.ZipFile(zip_file, 'r') as zf:
                zf.extractall(local_dir)
            
            # Organize: If extraction created a subfolder, move content up?
            # For now, recursive scan handles nested folders, so we are good.
            # Optionally delete zip to save space
            # zip_file.unlink() 
            
        return True
    except ImportError:
        print("[External] Error: 'huggingface_hub' not installed. pip install huggingface_hub")
        return False
    except Exception as e:
        print(f"[External] Download failed: {e}")
        return False

def scan_and_register(dataset_name: str, config: Dict) -> List[Dict[str, Any]]:
    """Scan directory and create metadata entries."""
    target_dir = EXTERNAL_DIR / config["subdir"]
    if not target_dir.exists():
        print(f"[External] Directory not found: {target_dir}")
        return []

    entries = []
    print(f"[External] Scanning {target_dir}...")
    
    # Recursive scan for audio
    audio_extensions = {".wav", ".mp3", ".flac", ".ogg"}
    files = [f for f in target_dir.rglob("*") if f.suffix.lower() in audio_extensions]
    
    for fpath in tqdm(files, desc=f"Indexing {dataset_name}"):
        # Get duration
        try:
            info = sf.info(str(fpath))
            duration = info.duration
        except Exception:
            duration = 0.0

        # Create rigorous metadata entry
        entry = {
            "path": str(fpath.resolve()),
            "speaker_id": "external_unknown", # Generic speaker ID
            "split": config["split_assignment"], # CRITICAL: Enforce Split
            "is_synthetic": True,
            "method": config["method_label"],
            "generator": f"{config['method_label']}_external",
            "duration": duration,
            "source": "external"
        }
        entries.append(entry)
        
    print(f"[External] Found {len(entries)} samples for {dataset_name}")
    return entries

def main():
    EXTERNAL_DIR.mkdir(exist_ok=True, parents=True)
    
    all_metadata = []
    
    # Process ElevenLabs
    el_cfg = DATASETS["elevenlabs"]
    el_dir = EXTERNAL_DIR / el_cfg["subdir"]
    
    # 1. Download if empty
    if not el_dir.exists() or not any(el_dir.iterdir()):
        success = download_huggingface_dataset(el_cfg["repo_id"], el_dir)
        if not success:
            print("[External] Skipping ElevenLabs due to download failure.")
    
    # 2. Index
    if el_dir.exists():
        entries = scan_and_register("elevenlabs", el_cfg)
        all_metadata.extend(entries)
        
    # Process RVC (Manual Placeholder)
    rvc_dir = EXTERNAL_DIR / "rvc"
    if rvc_dir.exists():
        # Assume RVC files placed here manually are for VALIDATION/TEST
        # Let's say we split them 50/50 or assign all to val
        pass 
        # (Skipping for now until files exist)

    # Save Master Metadata
    if all_metadata:
        print(f"[External] Saving metadata for {len(all_metadata)} total samples...")
        with open(METADATA_FILE, 'w') as f:
            json.dump(all_metadata, f, indent=2)
        print(f"[External] Saved to {METADATA_FILE}")
    else:
        print("[External] No external data found or downloaded.")

if __name__ == "__main__":
    main()
