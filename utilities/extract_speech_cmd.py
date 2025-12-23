"""
Extract Speech Commands Samples

The Speech Commands dataset was downloaded via torchaudio.
This script extracts samples using scipy (avoids FFmpeg issues).

Usage:
    python extract_speech_cmd.py
"""

import os
from pathlib import Path
import random


def extract_samples(num_samples: int = 10):
    """
    Extract samples from downloaded Speech Commands dataset.
    """
    print("=" * 50)
    print("EXTRACTING SPEECH COMMANDS SAMPLES")
    print("=" * 50)
    
    # Find the extracted data
    base_dir = Path("./speech_commands_data/SpeechCommands/speech_commands_v0.02")
    output_dir = Path("./test_samples_real")
    output_dir.mkdir(exist_ok=True)
    
    if not base_dir.exists():
        print(f"Dataset not found at {base_dir}")
        return False
    
    try:
        from scipy.io import wavfile
        import numpy as np
        import soundfile as sf
    except ImportError as e:
        print(f"Missing: {e}")
        return False
    
    # Find all command directories (exclude testing/validation lists)
    command_dirs = [
        d for d in base_dir.iterdir() 
        if d.is_dir() and not d.name.startswith('_')
    ]
    
    print(f"\n[1/3] Found {len(command_dirs)} command categories:")
    for d in sorted(command_dirs)[:10]:
        print(f"       - {d.name}")
    if len(command_dirs) > 10:
        print(f"       ... and {len(command_dirs) - 10} more")
    
    print(f"\n[2/3] Extracting {num_samples} random samples...")
    
    # Collect all WAV files
    all_wavs = []
    for cmd_dir in command_dirs:
        for wav_file in cmd_dir.glob("*.wav"):
            all_wavs.append((cmd_dir.name, wav_file))
    
    print(f"       Total available: {len(all_wavs)} samples")
    
    # Random sample
    random.seed(42)
    selected = random.sample(all_wavs, min(num_samples, len(all_wavs)))
    
    extracted = []
    speaker_ids = set()
    
    for i, (command, wav_path) in enumerate(selected):
        try:
            # Read WAV using scipy
            sr, audio = wavfile.read(str(wav_path))
            
            # Convert to float [-1, 1]
            if audio.dtype == np.int16:
                audio = audio.astype(np.float32) / 32768.0
            
            # Get speaker ID from filename (format: {hash}_{index}_{speaker}.wav)
            parts = wav_path.stem.split('_')
            speaker_id = parts[-1] if len(parts) >= 2 else f"speaker_{i}"
            speaker_ids.add(speaker_id)
            
            # Save to output
            out_path = output_dir / f"speech_{command}_{i:02d}.wav"
            sf.write(str(out_path), audio, sr)
            
            # Save metadata
            meta_path = output_dir / f"speech_{command}_{i:02d}.txt"
            with open(meta_path, 'w') as f:
                f.write(f"Transcript: {command}\n")
                f.write(f"Speaker: {speaker_id}\n")
                f.write(f"Duration: {len(audio)/sr:.2f}s\n")
                f.write(f"Source: Google Speech Commands v0.02\n")
            
            extracted.append({
                "command": command,
                "speaker": speaker_id,
                "duration": len(audio) / sr,
                "path": out_path
            })
            
            print(f"       ✓ {out_path.name} ({command}, {len(audio)/sr:.2f}s)")
            
        except Exception as e:
            print(f"       ✗ Failed {wav_path.name}: {e}")
    
    print(f"\n[3/3] Summary:")
    print(f"       Extracted: {len(extracted)} samples")
    print(f"       Unique speakers: {len(speaker_ids)}")
    print(f"       Commands: {set(e['command'] for e in extracted)}")
    print(f"\n       Saved to: {output_dir.absolute()}")
    
    print("\n" + "=" * 50)
    print("SUCCESS! Real speech samples ready.")
    print("=" * 50)
    
    return True


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--num_samples", type=int, default=10)
    args = parser.parse_args()
    
    success = extract_samples(args.num_samples)
    exit(0 if success else 1)
