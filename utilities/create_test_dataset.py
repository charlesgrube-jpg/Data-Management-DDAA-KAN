"""
Create a minimal test dataset for pipeline testing.

Since partial tar.gz files can't be extracted properly,
this creates a small synthetic dataset with real-like structure
to test the pipeline components.
"""

import numpy as np
import soundfile as sf
from pathlib import Path
import csv
import random

# Sample sentences for TTS
SENTENCES = [
    "The quick brown fox jumps over the lazy dog.",
    "Hello, my name is John and I am testing the audio pipeline.",
    "This is a sample sentence for text to speech synthesis.",
    "Machine learning models require diverse training data.",
    "Audio deepfake detection is an important research area.",
    "Common Voice is a dataset created by Mozilla Foundation.",
    "Voice conversion can transform one person's voice to another.",
    "Codec compression can affect the quality of audio signals.",
    "Natural language processing has advanced significantly.",
    "Speech recognition technology continues to improve.",
    "The weather today is sunny with clear skies.",
    "Please remember to save your work frequently.",
    "Artificial intelligence is transforming many industries.",
    "Open source projects enable collaborative development.",
    "Audio processing pipelines handle complex transformations.",
    "Quality assurance is essential for research datasets.",
    "Reproducibility is a cornerstone of scientific research.",
    "Data augmentation helps improve model generalization.",
    "Feature extraction is a key step in audio analysis.",
    "Spectrograms visualize audio frequency content over time.",
]

GENDERS = ["male", "female", "other"]
AGES = ["twenties", "thirties", "fourties", "fifties"]
ACCENTS = ["us", "uk", "australia", "indian", "other"]


def generate_test_audio(duration_sec: float, sr: int = 16000) -> np.ndarray:
    """
    Generate a simple test audio signal.
    Creates a speech-like signal with some variation.
    """
    t = np.linspace(0, duration_sec, int(duration_sec * sr))
    
    # Multiple sine waves to simulate speech harmonics
    freqs = [120, 240, 360, 480]  # Fundamental + harmonics
    audio = np.zeros_like(t)
    
    for i, freq in enumerate(freqs):
        amplitude = 1.0 / (i + 1)  # Decreasing amplitude
        audio += amplitude * np.sin(2 * np.pi * freq * t)
    
    # Add some amplitude modulation (like syllables)
    mod_freq = random.uniform(2, 5)  # 2-5 Hz modulation
    modulation = 0.5 + 0.5 * np.sin(2 * np.pi * mod_freq * t)
    audio *= modulation
    
    # Add slight noise
    audio += np.random.randn(len(audio)) * 0.01
    
    # Normalize
    audio = audio / np.max(np.abs(audio)) * 0.8
    
    return audio.astype(np.float32)


def create_test_dataset(n_samples: int = 20, sr: int = 16000):
    """Create a minimal test dataset."""
    
    print("=" * 60)
    print("CREATING TEST DATASET")
    print("=" * 60)
    
    # Create output directories matching Common Voice structure
    output_dir = Path("mozilla_cv_data/extracted/cv-corpus-24.0-2025-12-05/en")
    clips_dir = output_dir / "clips"
    clips_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Output: {output_dir}")
    print(f"Samples: {n_samples}")
    
    # Generate samples
    metadata = []
    
    for i in range(n_samples):
        # Random duration between 2-6 seconds
        duration = random.uniform(2, 6)
        
        # Generate audio
        audio = generate_test_audio(duration, sr)
        
        # Create unique speaker ID
        speaker_id = f"speaker_{i // 5:03d}"  # Groups of 5 per speaker
        
        # Filename
        filename = f"common_voice_test_{i:05d}.wav"
        filepath = clips_dir / filename
        
        # Save audio
        sf.write(str(filepath), audio, sr)
        
        # Metadata
        metadata.append({
            "path": f"clips/{filename}",
            "sentence": random.choice(SENTENCES),
            "client_id": speaker_id,
            "gender": random.choice(GENDERS),
            "age": random.choice(AGES),
            "accent": random.choice(ACCENTS),
        })
        
        print(f"\r  Generated: {i+1}/{n_samples}", end="", flush=True)
    
    print()
    
    # Write metadata TSV (Common Voice format)
    tsv_path = output_dir / "validated.tsv"
    with open(tsv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(
            f, 
            fieldnames=["path", "sentence", "client_id", "gender", "age", "accent"],
            delimiter='\t'
        )
        writer.writeheader()
        writer.writerows(metadata)
    
    print(f"\n✓ Created {n_samples} audio files in {clips_dir}")
    print(f"✓ Metadata at {tsv_path}")
    
    print("\n" + "=" * 60)
    print("DATASET READY!")
    print("=" * 60)
    print("Now run: python run_pipeline.py")
    
    return True


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--samples", type=int, default=20, 
                        help="Number of samples to generate")
    args = parser.parse_args()
    
    create_test_dataset(args.samples)
