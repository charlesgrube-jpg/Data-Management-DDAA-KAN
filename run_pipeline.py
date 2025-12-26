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
from pipeline.effects.effects import apply_effects, apply_effects_to_pair
from pipeline.data_gen.splitter import create_speaker_disjoint_splits
from pipeline.data_gen.validator import validate_dataset, generate_validation_report
from pipeline.data_gen.exporter import export_dataset
from pipeline.synthesizer.gtts_synthesizer import synthesize_tts
from pipeline.synthesizer.synthesizer import get_synthesizer_manager


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
    # Use config path if available, otherwise default (compatibility)
    default_path = "mozilla_cv_data/cv-corpus-24.0-2025-12-05/en"
    cv_path_str = getattr(config.source, "data_path", default_path)
    cv_path = Path(cv_path_str)
    
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
    
    # Initialize synthesizer manager for TTS/VC
    synth_manager = get_synthesizer_manager(config)
    vc_device = getattr(config.synthesis, 'vc_device', 'cpu')
    synth_manager.initialize(device=vc_device)
    
    working_set = []
    source_count = 0
    skip_count = 0
    
    # Initialize progress bar
    from tqdm import tqdm
    total_samples = config.source.max_samples if config.source.max_samples else len(available_clips)
    
    # Create the generator
    loader_gen = load_mozilla_cv(str(cv_path), config, split=config.source.split)
    
    # Wrap in tqdm
    for sample in tqdm(loader_gen, total=total_samples, desc="Processing Samples", unit="sample"):
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
            # Phase 4: Synthesize using TTS/VC based on strategy
            transcript = sample.get("transcript", "")
            
            # Use synthesizer manager to get TTS and/or VC results
            tts_result, vc_result = synth_manager.synthesize(
                chunk["audio"],
                config.audio.target_sr,
                transcript
            )
            
            # Pick which synthetic to use based on strategy
            syn_result = synth_manager.pick_synthetic(tts_result, vc_result)
            
            if syn_result is None:
                if config.synthesis.fallback_on_error:
                    # Use original audio as fallback (for testing)
                    print(f"    Synthesis failed, using original audio as placeholder")
                    syn_result = {
                        "audio": chunk["audio"],
                        "generator": "none",
                        "method": "none"
                    }
                else:
                    print(f"    Synthesis failed for chunk {chunk['chunk_idx']}, skipping")
                    continue
            
            syn_audio = syn_result["audio"]
            
            # Phase 5: Apply effects (channel + codec)
            # Apply same effects to both real and synthetic for fairness
            real_processed, syn_processed, quality_tier, codec_tier = apply_effects_to_pair(
                chunk["audio"],
                syn_audio,
                config.audio.target_sr,
                config
            )
            
            # Parse codec tier for metadata
            codec_type, codec_bitrate = "none", None
            if codec_tier != "none":
                parts = codec_tier.split("_")
                if len(parts) == 2:
                    codec_type = parts[0]
                    codec_bitrate = parts[1]
            
            base_labels = {
                "source_id": source_id,
                "speaker_id": sample.get("speaker_id", source_id),
                "chunk_idx": chunk["chunk_idx"],
                "transcript": transcript,
                "gender": sample.get("gender", "unknown"),
                "accent": sample.get("accent", "unknown"),
                "age": sample.get("age", "unknown"),
                "quality_tier": quality_tier,
                "codec_type": codec_type,
                "codec_bitrate": codec_bitrate if codec_bitrate else "",
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
            
            # Add synthetic sample with proper metadata
            working_set.append({
                **base_labels,
                "audio": syn_processed,
                "is_synthetic": True,
                "generator": syn_result["generator"],
                "method": syn_result["method"],
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
    # ============== DONE ==============
    elapsed = time.time() - start_time
    total_mb_processed = sum(s.get('original_size', 0) for s in processed_samples) / 1024 / 1024
    throughput = total_mb_processed / elapsed if elapsed > 0 else 0
    
    # Estimate for 3.5 GB (Common Voice English partial)
    target_size_mb = 3584 # 3.5 GB
    estimated_seconds = (target_size_mb / throughput) if throughput > 0 else 0
    estimated_hours = estimated_seconds / 3600
    
    print("\n" + "=" * 60)
    print("MOZILLA CV PIPELINE COMPLETE!")
    print("=" * 60)
    print(f"Time: {elapsed:.1f} seconds")
    print(f"Throughput: {throughput:.2f} MB/s")
    if throughput > 0:
        print(f"📉 ESTIMATED TIME for 3.5GB Dataset: {estimated_hours:.2f} hours")
        
    print(f"Output: {config.output.base_dir}")
    print(f"Total samples: {len(processed_samples)}")
    print(f"  Train: {len(train_metadata)}")
    print(f"  Val: {len(val_metadata)}")
    print(f"  Test: {len(test_metadata)}")
    
    return True


if __name__ == "__main__":
    success = run_mozilla_cv_pipeline()
    exit(0 if success else 1)
