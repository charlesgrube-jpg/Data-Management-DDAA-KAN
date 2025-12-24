"""
Download Mozilla Common Voice Sample via HuggingFace

Quick way to get a small sample for testing the pipeline.
Uses streaming to avoid downloading the full 80GB+ dataset.
"""

from datasets import load_dataset
import soundfile as sf
from pathlib import Path
import csv

def download_sample(n_samples=20):
    """Download a small sample of Mozilla Common Voice."""
    
    print("=" * 60)
    print("DOWNLOADING MOZILLA CV SAMPLE VIA HUGGINGFACE")
    print("=" * 60)
    
    # Create output directories
    output_dir = Path("mozilla_cv_data/extracted/cv-corpus-24.0-2025-12-05/en")
    clips_dir = output_dir / "clips"
    clips_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Output: {output_dir}")
    print(f"Samples: {n_samples}")
    
    # Load dataset in streaming mode (no full download)
    print("\n[1/3] Connecting to HuggingFace...")
    try:
        ds = load_dataset(
            "mozilla-foundation/common_voice_13_0",
            "en",
            split="train",
            streaming=True,
            trust_remote_code=True
        )
    except Exception as e:
        print(f"Error loading from HuggingFace: {e}")
        print("\nYou may need to:")
        print("1. Accept the dataset license at:")
        print("   https://huggingface.co/datasets/mozilla-foundation/common_voice_13_0")
        print("2. Login with: huggingface-cli login")
        return False
    
    print("[2/3] Downloading samples...")
    
    # Prepare metadata
    metadata = []
    
    # Download samples
    count = 0
    for sample in ds:
        if count >= n_samples:
            break
        
        # Get audio data
        audio = sample["audio"]
        audio_array = audio["array"]
        sample_rate = audio["sampling_rate"]
        
        # Create filename
        client_id = sample.get("client_id", f"unknown_{count}")[:20]
        filename = f"cv_sample_{count:04d}_{client_id}.wav"
        filepath = clips_dir / filename
        
        # Save audio
        sf.write(str(filepath), audio_array, sample_rate)
        
        # Save metadata
        metadata.append({
            "path": f"clips/{filename}",
            "sentence": sample.get("sentence", ""),
            "client_id": sample.get("client_id", ""),
            "gender": sample.get("gender", ""),
            "age": sample.get("age", ""),
            "accent": sample.get("accent", ""),
        })
        
        count += 1
        print(f"\r  Downloaded: {count}/{n_samples}", end="", flush=True)
    
    print()
    
    # Write metadata TSV (like Mozilla CV format)
    print("[3/3] Writing metadata...")
    tsv_path = output_dir / "validated.tsv"
    with open(tsv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=["path", "sentence", "client_id", "gender", "age", "accent"], delimiter='\t')
        writer.writeheader()
        writer.writerows(metadata)
    
    print(f"\n✓ Saved {count} samples to {clips_dir}")
    print(f"✓ Metadata at {tsv_path}")
    
    print("\n" + "=" * 60)
    print("DOWNLOAD COMPLETE!")
    print("=" * 60)
    print("Now run: python run_pipeline.py")
    
    return True


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--samples", type=int, default=20, help="Number of samples to download")
    args = parser.parse_args()
    
    download_sample(args.samples)
