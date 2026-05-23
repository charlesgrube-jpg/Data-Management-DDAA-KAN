"""
Main Pipeline Orchestrator

Coordinates all pipeline stages:
1. Load source dataset
2. Preprocess audio
3. Segment into chunks
4. Synthesize (TTS + VC)
5. Apply effects
6. Split by speaker
7. Validate
8. Export
"""

import sys
import time
from pathlib import Path
from typing import Optional, List, Dict, Any

from pipeline.config import load_config, Config
from .loader import load_source_dataset
from pipeline.standardizer.preprocessor import preprocess_audio
from pipeline.standardizer.segmenter import segment_audio
from pipeline.synthesizer.synthesizer import synthesize_audio
from pipeline.effects.effects import apply_effects_to_pair, select_quality_tier
from .splitter import create_speaker_disjoint_splits
from .validator import validate_dataset, generate_validation_report
from .exporter import export_dataset, generate_dataset_card


def run_pipeline(config_path: str = "config.yaml") -> Dict[str, Any]:
    """
    Execute the full dataset processing pipeline.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Pipeline results including stats and validation
    """
    start_time = time.time()
    
    # ============== LOAD CONFIG ==============
    print("=" * 60)
    print("AUDIO DEEPFAKE DATASET PIPELINE")
    print("=" * 60)
    
    config = load_config(config_path)
    print(f"\nConfiguration loaded from: {config_path}")
    print(f"Source: {config.source.dataset_name}")
    print(f"Output: {config.output.base_dir}")
    
    # ============== PHASE 1: LOAD ==============
    print("\n" + "-" * 40)
    print("PHASE 1: Loading Dataset")
    print("-" * 40)
    
    working_set: List[Dict[str, Any]] = []
    source_count = 0
    skip_count = 0
    error_count = 0
    
    # ============== PHASES 2-5: PROCESS EACH SAMPLE ==============
    print("\n" + "-" * 40)
    print("PHASES 2-5: Processing Samples")
    print("-" * 40)
    
    for sample in load_source_dataset(config):
        source_count += 1
        
        try:
            # Phase 2: Preprocess
            processed_audio, preprocess_meta = preprocess_audio(
                sample["audio"],
                sample["sample_rate"],
                config
            )
            
            if processed_audio is None:
                skip_count += 1
                continue
            
            # Phase 3: Segment
            source_id = f"src_{sample['source_idx']:06d}"
            chunks = segment_audio(
                processed_audio,
                source_id=source_id,
                config=config,
                metadata=sample
            )
            
            # Process each chunk
            for chunk in chunks:
                # Phase 4: Synthesize
                synthetic = synthesize_audio(
                    chunk["audio"],
                    config.audio.target_sr,
                    chunk.get("transcript", ""),
                    config
                )
                
                if synthetic is None and config.synthesis.fallback_on_error:
                    continue  # Skip if synthesis failed
                
                # Phase 5: Apply effects to pair (same degradation to both for fairness)
                if synthetic:
                    real_audio, syn_audio, quality_tier = apply_effects_to_pair(
                        chunk["audio"],
                        synthetic["audio"],
                        config.audio.target_sr,
                        config
                    )
                else:
                    quality_tier = select_quality_tier(config)
                    from pipeline.effects.effects import apply_effects
                    real_audio = apply_effects(chunk["audio"], config.audio.target_sr, quality_tier)
                    syn_audio = None
                
                # Build base labels
                base_labels = {
                    "source_id": source_id,
                    "speaker_id": chunk.get("speaker_id", source_id),
                    "chunk_idx": chunk["chunk_idx"],
                    "transcript": chunk.get("transcript", ""),
                    "gender": chunk.get("gender", "unknown"),
                    "accent": chunk.get("accent", "unknown"),
                    "quality_tier": quality_tier,
                    "duration": chunk["duration"],
                }
                
                # Add real sample
                working_set.append({
                    **base_labels,
                    "audio": real_audio,
                    "is_synthetic": False,
                    "generator": None,
                    "method": None,
                })
                
                # Add synthetic sample
                if syn_audio is not None:
                    working_set.append({
                        **base_labels,
                        "audio": syn_audio,
                        "is_synthetic": True,
                        "generator": synthetic.get("generator", "unknown"),
                        "method": synthetic.get("method", "unknown"),
                    })
                
        except Exception as e:
            error_count += 1
            if error_count <= 5:
                print(f"[Pipeline] Error processing sample {source_count}: {e}")
        
        # Progress report
        if source_count % 100 == 0:
            print(f"[Pipeline] Processed {source_count} sources, "
                  f"{len(working_set)} samples in working set")
    
    print(f"\n[Pipeline] Processing complete:")
    print(f"  Sources processed: {source_count}")
    print(f"  Sources skipped: {skip_count}")
    print(f"  Errors: {error_count}")
    print(f"  Working set size: {len(working_set)}")
    
    # ============== PHASE 6: SPLIT ==============
    print("\n" + "-" * 40)
    print("PHASE 6: Creating Splits")
    print("-" * 40)
    
    working_set = create_speaker_disjoint_splits(working_set, config)
    
    # ============== PHASE 7: VALIDATE ==============
    print("\n" + "-" * 40)
    print("PHASE 7: Validating Dataset")
    print("-" * 40)
    
    validation_results = validate_dataset(working_set, config)
    report = generate_validation_report(validation_results)
    print(report)
    
    if not validation_results["all_passed"]:
        print("\n[Pipeline] WARNING: Validation issues found!")
        response = input("Continue with export? (y/n): ")
        if response.lower() != 'y':
            print("[Pipeline] Aborted by user.")
            return {"status": "aborted", "validation": validation_results}
    
    # ============== PHASE 8: EXPORT ==============
    print("\n" + "-" * 40)
    print("PHASE 8: Exporting Dataset")
    print("-" * 40)
    
    export_stats = export_dataset(working_set, config)
    
    # Generate dataset card
    generate_dataset_card(config, export_stats, config.output_path)
    
    # Save validation report
    report_path = config.output_path / "validation_report.txt"
    with open(report_path, 'w') as f:
        f.write(report)
    
    # ============== DONE ==============
    elapsed = time.time() - start_time
    
    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE")
    print("=" * 60)
    print(f"Total time: {elapsed/60:.1f} minutes")
    print(f"Output: {config.output.base_dir}")
    print(f"Samples: {export_stats['total']}")
    
    return {
        "status": "success",
        "elapsed_seconds": elapsed,
        "source_count": source_count,
        "export_stats": export_stats,
        "validation": validation_results
    }


def run_pilot(
    config_path: str = "config.yaml",
    num_samples: int = 100
) -> Dict[str, Any]:
    """
    Run pipeline on a small subset for testing.
    
    Temporarily overrides max_samples config.
    """
    config = load_config(config_path)
    
    # Override for pilot
    original_max = config.source.max_samples
    config.source.max_samples = num_samples
    
    print(f"[Pilot] Running with {num_samples} samples (pilot mode)")
    
    # Can't use run_pipeline directly since config is already loaded
    # Instead, modify the config file temporarily or implement inline
    
    # For now, just inform user
    print(f"[Pilot] Set 'max_samples: {num_samples}' in config.yaml and run main pipeline")
    
    config.source.max_samples = original_max
    return {"status": "pilot_info_provided"}


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Audio Deepfake Dataset Pipeline")
    parser.add_argument("--config", default="config.yaml", help="Path to config file")
    parser.add_argument("--pilot", type=int, default=None, help="Run pilot with N samples")
    
    args = parser.parse_args()
    
    if args.pilot:
        run_pilot(args.config, args.pilot)
    else:
        result = run_pipeline(args.config)
        print(f"\nResult: {result['status']}")
