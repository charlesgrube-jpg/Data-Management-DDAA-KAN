"""
Generate sample TTS audio for quality comparison.

Creates pairs of real + TTS synthetic audio in processed_tts/ folder.
"""

import sys
sys.path.insert(0, '.')

from pipeline.config import load_config
from pipeline.data_gen.mozilla_cv_loader import load_mozilla_cv
from pipeline.synthesizer.gtts_synthesizer import synthesize_tts
import soundfile as sf
from pathlib import Path

def main():
    config = load_config('config.yaml')
    cv_path = 'mozilla_cv_data/extracted/cv-corpus-24.0-2025-12-05/en'
    out_dir = Path('processed_tts')
    out_dir.mkdir(exist_ok=True)

    count = 0
    max_samples = 5
    
    print(f"Generating {max_samples} real/TTS pairs...\n")
    
    for sample in load_mozilla_cv(cv_path, config, split='train'):
        if count >= max_samples:
            break
        
        transcript = sample['transcript']
        sr = sample['sample_rate']
        
        print(f"[{count+1}/{max_samples}] {transcript[:60]}...")
        
        # Save real audio
        real_path = out_dir / f'sample_{count:02d}_real.wav'
        sf.write(str(real_path), sample['audio'], sr)
        real_dur = len(sample['audio']) / sr
        print(f"  Real: {real_path.name} ({real_dur:.2f}s)")
        
        # Generate TTS
        syn = synthesize_tts(transcript, sample_rate=sr)
        if syn is not None:
            syn_path = out_dir / f'sample_{count:02d}_tts.wav'
            sf.write(str(syn_path), syn, sr)
            syn_dur = len(syn) / sr
            print(f"  TTS:  {syn_path.name} ({syn_dur:.2f}s)")
        else:
            print("  TTS:  FAILED")
        
        count += 1
        print()

    print("=" * 50)
    print(f"Done! Generated {count} pairs in {out_dir}/")
    print("Compare real vs TTS quality by listening to the files.")

if __name__ == "__main__":
    main()
