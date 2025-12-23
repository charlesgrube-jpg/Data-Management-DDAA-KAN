"""
Run Pipeline with Mozilla Common Voice

Generates paired real/synthetic audio dataset for deepfake detection.

Usage:
    python run_pipeline.py
"""

import sys
import time
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from pipeline.config import load_config
from pipeline.data_gen.mozilla_cv_loader import load_mozilla_cv, get_available_clips
from pipeline.standardizer.preprocessor import preprocess_audio
from pipeline.standardizer.segmenter import segment_audio
from pipeline.effects.effects import apply_effects, apply_effects_to_pair, select_quality_tier
from pipeline.data_gen.splitter import create_speaker_disjoint_splits
from pipeline.data_gen.validator import validate_dataset, generate_validation_report
from pipeline.data_gen.exporter import export_dataset
from pipeline.synthesizer.gtts_synthesizer import synthesize_tts


def run_mozilla_cv_pipeline():
    """Run pipeline with Mozilla Common Voice samples."""
    
    print("=" * 60)
    print("PIPELINE WITH MOZILLA COMMON VOICE")
    print("=" * 60)
    
    start_time = time.time()
    
    # Load config
    config = load_config("config.yaml")
    
    # Use timestamped output folder for unique runs
    output_path = config.get_timestamped_output_path()
    # Override config's output.base_dir for this run
    config.output.base_dir = str(output_path)
    
    # Path to extracted Mozilla CV data
    cv_path = Path("mozilla_cv_data/extracted/cv-corpus-24.0-2025-12-05/en")
    
    if not cv_path.exists():
        print(f"\n[ERROR] Mozilla CV data not found at {cv_path}")
        print("Run: python download_partial.py")
        return False
    
    print(f"\nSource: {cv_path}")
    print(f"Output: {output_path} (timestamped)")
    
    # Check what clips are available
    available_clips = get_available_clips(str(cv_path))
    print(f"Available audio clips: {len(available_clips)}")
    
    if len(available_clips) == 0:
        print("[ERROR] No audio clips found in clips/ directory")
        return False
    
    # ============== PHASE 1: LOAD MOZILLA CV ==============
    print("\n" + "-" * 40)
    print("PHASE 1: Loading Mozilla Common Voice")
    print("-" * 40)
    
    working_set = []
    source_count = 0
    skip_count = 0
    
    for sample in load_mozilla_cv(str(cv_path), config, split="train"):
        source_count += 1
        
        # Phase 2: Preprocess
        processed_audio, preprocess_meta = preprocess_audio(
            sample["audio"],
            sample["sample_rate"],
            config
        )
        
        if processed_audio is None:
            reason = preprocess_meta.get('skip_reason', 'unknown')
            print(f"  Skipped: {reason}")
            skip_count += 1
            continue
        
        # Phase 3: Segment
        source_id = f"cv_{sample['source_idx']:06d}"
        chunks = segment_audio(
            processed_audio,
            source_id=source_id,
            config=config,
            metadata=sample
        )
        
        duration = len(sample["audio"]) / sample["sample_rate"]
        print(f"  Loaded: {sample['file_path'].split('/')[-1]} ({duration:.1f}s) -> {len(chunks)} chunks")
        print(f"          \"{sample['transcript'][:50]}...\"")
        
        for chunk in chunks:
            # Phase 4: Synthesize using TTS
            
            transcript = sample.get("transcript", "")
            syn_audio = synthesize_tts(transcript, sample_rate=config.audio.target_sr)
            
            if syn_audio is None:
                print(f"    TTS failed for chunk {chunk['chunk_idx']}, skipping")
                continue
            
            # Phase 5: Apply effects
            quality_tier = select_quality_tier(config)
            
            # Apply same effects to both real and synthetic for fairness
            real_processed, syn_processed, quality_tier = apply_effects_to_pair(
                chunk["audio"],
                syn_audio,
                config.audio.target_sr,
                config
            )
            
            base_labels = {
                "source_id": source_id,
                "speaker_id": sample.get("speaker_id", source_id),
                "chunk_idx": chunk["chunk_idx"],
                "transcript": transcript,
                "gender": sample.get("gender", "unknown"),
                "accent": sample.get("accent", "unknown"),
                "age": sample.get("age", "unknown"),
                "quality_tier": quality_tier,
                "duration": chunk["duration"],
                # Track SNR based on quality tier
                "snr_db": 15 if quality_tier == "noisy" else "",
            }
            
            # Generate real filename for pairing
            real_filename = f"{source_id}_chunk{chunk['chunk_idx']:03d}.wav"
            
            # Add real sample
            working_set.append({
                **base_labels,
                "audio": real_processed,
                "is_synthetic": False,
                "generator": None,
                "method": None,
                "source_real_file": "",  # Real files don't have a source
            })
            
            # Add synthetic sample (real TTS!)
            working_set.append({
                **base_labels,
                "audio": syn_processed,
                "is_synthetic": True,
                "generator": "gTTS-2.5.4",  # Include version
                "method": "tts",
                "source_real_file": f"real/{real_filename}",  # Link to real source
            })
    
    print(f"\n  Sources loaded: {source_count}")
    print(f"  Skipped: {skip_count}")
    print(f"  Working set: {len(working_set)} samples")
    
    if len(working_set) == 0:
        print("[ERROR] No samples in working set!")
        return False
    
    # ============== PHASE 6: SPLIT ==============
    print("\n" + "-" * 40)
    print("PHASE 6: Creating Speaker-Disjoint Splits")
    print("-" * 40)
    
    working_set = create_speaker_disjoint_splits(working_set, config)
    
    # ============== PHASE 7: VALIDATE ==============
    print("\n" + "-" * 40)
    print("PHASE 7: Validating Dataset")
    print("-" * 40)
    
    validation_results = validate_dataset(working_set, config)
    report = generate_validation_report(validation_results)
    print(report)
    
    # ============== PHASE 8: EXPORT ==============
    print("\n" + "-" * 40)
    print("PHASE 8: Exporting Dataset")
    print("-" * 40)
    
    export_stats = export_dataset(working_set, config)
    
    # ============== DONE ==============
    elapsed = time.time() - start_time
    
    print("\n" + "=" * 60)
    print("MOZILLA CV PIPELINE COMPLETE!")
    print("=" * 60)
    print(f"Time: {elapsed:.1f} seconds")
    print(f"Output: {config.output.base_dir}")
    print(f"Total samples: {export_stats['total']}")
    print(f"  Train: {export_stats['train']}")
    print(f"  Val: {export_stats['val']}")
    print(f"  Test: {export_stats['test']}")
    
    return True


if __name__ == "__main__":
    success = run_mozilla_cv_pipeline()
    exit(0 if success else 1)
