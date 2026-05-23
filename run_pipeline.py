"""
Run Pipeline with Mozilla Common Voice

Generates paired real/synthetic audio dataset for deepfake detection.

Usage:
    python run_pipeline.py
"""

import sys
import time
import json
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from pipeline.config import load_config
from pipeline.data_gen.mozilla_cv_loader import load_mozilla_cv, get_available_clips
from pipeline.standardizer.preprocessor import preprocess_audio
from pipeline.standardizer.segmenter import segment_audio
from pipeline.effects.effects import apply_effects, apply_effects_to_pair
from pipeline.data_gen.splitter import assign_split
from pipeline.data_gen.validator import validate_dataset, generate_validation_report
from pipeline.data_gen.exporter import export_dataset
from pipeline.synthesizer.gtts_synthesizer import synthesize_tts
from pipeline.synthesizer.synthesizer import get_synthesizer_manager
from pipeline.features.cqt_extractor import CQTExtractor
from pipeline.features.lfcc_extractor import LFCCExtractor
import torch
import numpy as np
import concurrent.futures
from tqdm import tqdm


def process_single_sample(sample, config, synth_manager, vc_device):
    """Process a single audio sample: preprocess, segment, synthesize, effects."""
    
    # Lazy Load Audio if needed
    if sample.get("audio") is None:
        try:
            import librosa
            # Load with librosa to get float32 arrays (standard for this pipeline)
            # print(f"[Worker] Loading {sample['file_path']}...") 
            audio, sr = librosa.load(sample["file_path"], sr=config.audio.target_sr)
            sample["audio"] = audio
            sample["sample_rate"] = sr
        except Exception as e:
            print(f"[Worker] Failed to load audio for {sample.get('source_idx')} ({sample.get('file_path')}): {e}")
            return [], 0 # Fail gracefully

    # Extra Safety Check
    if sample.get("audio") is None:
        print(f"[Worker] CRITICAL: Audio is still None after loading logic! {sample.get('file_path')}")
        return [], 0

    # Phase 1.5: Assign Split Immediately (V3 Strategy)
    # This ensures we know if a sample is train/test BEFORE generating audio.
    try:
        assign_split(sample, config)
    except Exception as e:
        print(f"[Worker] Split assignment failed: {e}")
        return [], 0
    
    # Phase 2: Preprocess
    try:
        processed_audio, preprocess_meta = preprocess_audio(
            sample["audio"],
            sample["sample_rate"],
            config
        )
    except Exception as e:
        print(f"[Worker] Preprocessing failed for {sample.get('source_idx')}: {e}")
        # Print detailed type info
        # import traceback
        # traceback.print_exc()
        return [], 0
    
    if processed_audio is None:
        return [], 1  # 0 results, 1 skipped
    
    # Phase 3: Segment
    source_id = f"cv_{sample['source_idx']:06d}"
    chunks = segment_audio(
        processed_audio,
        source_id=source_id,
        config=config,
        metadata=sample
    )
    
    results = []
    
    for chunk in chunks:
        # Phase 4: Synthesize using TTS/VC based on strategy and split
        transcript = sample.get("transcript", "")
        split = sample.get("split", "train")
        
        # --- VOICE SELECTION LOGIC (V3) ---
        selected_voice = None
        
        # Determine which pool to use based on split
        if split == "test":
            # STRICT: Must come from Test Pool
            pool = config.synthesis.voice_pools.test
            # if not pool: WARN?
            if pool:
                selected_voice = np.random.choice(pool)
        else:
            # TRAIN/VAL: Can use Train Pool OR Train-Only Models
            pool = config.synthesis.voice_pools.train
            train_only = config.synthesis.train_only_models
            
            # Weighted selection: 70% from Main Pool, 30% from Train-Only (if available)
            if train_only and np.random.random() < 0.3:
                selected_voice = np.random.choice(train_only)
            elif pool:
                selected_voice = np.random.choice(pool)
        
        # ---  PRE-DECIDE: TTS or RVC (50/50 based on hash) ---
        # This avoids running BOTH and then discarding one.
        import hashlib
        sample_key = f"{sample.get('source_idx', 0)}_{sample.get('speaker_id', 'unk')}"
        decision_hash = int(hashlib.md5(sample_key.encode()).hexdigest(), 16) % 100
        use_rvc = config.synthesis.enable_vc and (decision_hash < 50)  # 50% RVC if enabled
        
        if use_rvc:
            # RVC ONLY path (skip TTS entirely)
            tts_result = None
            vc_result = synth_manager.synthesize_vc_only(
                chunk["audio"],
                config.audio.target_sr,
                split=split
            )
            syn_result = vc_result
        else:
            # TTS ONLY path (skip RVC entirely)
            tts_result, _ = synth_manager.synthesize(
                chunk["audio"],
                config.audio.target_sr,
                transcript,
                voice_preset=selected_voice,
                split=split,
                skip_vc=True  # NEW: Skip RVC computation
            )
            syn_result = tts_result
        
        # Legacy pick_synthetic call removed - we already decided above
        
        if syn_result is None:
            if config.synthesis.fallback_on_error:
                # Use original audio as fallback (for testing)
                syn_result = {
                    "audio": chunk["audio"],
                    "generator": "none",
                    "method": "none"
                }
            else:
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
            "split": sample["split"], # Propagate split to chunk
        }
        
        # Generate real filename for pairing
        real_filename = f"{source_id}_chunk{chunk['chunk_idx']:03d}.wav"
        
        # Add real sample
        results.append({
            **base_labels,
            "audio": real_processed,
            "is_synthetic": False,
            "generator": None,
            "method": None,
            "source_real_file": "",  # Real files don't have a source
        })
        
        # Add synthetic sample with proper metadata
        results.append({
            **base_labels,
            "audio": syn_processed,
            "is_synthetic": True,
            "generator": syn_result["generator"],
            "method": syn_result["method"],
            "source_real_file": f"real/{real_filename}",  # Link to real source
        })
        
    return results, 0  # results, 0 skipped


def run_mozilla_cv_pipeline():
    """Run pipeline with Mozilla Common Voice samples."""
    
    print("=" * 60)
    print("PIPELINE WITH MOZILLA COMMON VOICE (PARALLEL + V3 DISJOINT)")
    print("=" * 60)
    
    start_time = time.time()
    
    # Load config
    config = load_config("config.yaml")
    
    # Use timestamped output folder for unique runs
    output_path = config.get_timestamped_output_path()
    # Override config's output.base_dir for this run
    config.output.base_dir = str(output_path)
    
    # Path to extracted Mozilla CV data
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
    total_samples = config.source.max_samples if config.source.max_samples else len(available_clips)
    
    # Phase 1.8: Ingest External Data (V3 Update)
    if hasattr(config, 'external_datasets'):
         ext_meta_path = Path("external_data/metadata_external.json")
         if ext_meta_path.exists():
              with open(ext_meta_path, 'r') as f:
                  ext_samples = json.load(f)
              
              # Filter enabled datasets and apply max_samples
              enabled_datasets = {k: v for k, v in config.external_datasets.items() if v.enabled}
              final_ext_samples = []
              
              for method, cfg in enabled_datasets.items():
                  # Get samples for this method
                  method_samples = [s for s in ext_samples if s.get('method') == method]
                  
                  # Limit if max_samples is set
                  if cfg.max_samples is not None:
                      method_samples = method_samples[:cfg.max_samples]
                      print(f"[Pipeline] Limited {method} to {len(method_samples)} samples")
                  
                  final_ext_samples.extend(method_samples)
                  
              ext_samples = final_ext_samples
              
              if ext_samples:
                   print(f"[Pipeline] Ingesting {len(ext_samples)} external samples...")
                   
                   import soundfile as sf
                   
                   valid_ext_samples = []
                   for i, s in enumerate(ext_samples):
                       try:
                           path = str(Path(s['path']).resolve())
                           s['audio_path'] = path
                           
                           # Load Audio
                           audio, sr = sf.read(path)
                           
                           # Ensure float32
                           if audio.dtype != np.float32:
                                audio = audio.astype(np.float32)

                           # Use standard preprocessor for resampling/mono/norm
                           processed_audio, meta = preprocess_audio(audio, sr, config)
                           
                           if processed_audio is not None:
                               s['audio'] = processed_audio
                               s['sample_rate'] = config.audio.target_sr
                               s['chunk_idx'] = 0 # Treat as single chunk
                               s['quality_tier'] = 'clean'
                               s['source_id'] = f"ext_{s['method']}_{i:06d}" # Generate unique ID
                               s['duration'] = meta.get('duration', len(processed_audio)/config.audio.target_sr)
                               
                               # Assign Split for External Data too? Usually defined in config
                               dataset_cfg = enabled_datasets.get(s['method'])
                               s['split'] = dataset_cfg.split if dataset_cfg else 'train'

                               valid_ext_samples.append(s)
                           else:
                               print(f"[Pipeline] Skipped external sample {path}: {meta.get('skip_reason')}")

                       except Exception as e:
                           print(f"[Pipeline] Failed to load external sample {s.get('path')}: {e}")
                       
                   working_set.extend(valid_ext_samples)
         else:
              # Only warn if external datasets are enabled in config
              if any(v.enabled for v in config.external_datasets.values()):
                   print("[Pipeline] WARNING: External datasets enabled but metadata_external.json not found.")

    # Collect all samples first to use in executor
    print("Collecting samples...")
    # Pre-fetch samples to list
    all_samples = list(load_mozilla_cv(str(cv_path), config, split=config.source.split))
    
    # CHUNK FILTERING: If running as array job, only process our chunk
    chunk_id = os.environ.get('PIPELINE_CHUNK_ID')
    num_chunks = os.environ.get('PIPELINE_NUM_CHUNKS')
    if chunk_id is not None and num_chunks is not None:
        chunk_id = int(chunk_id)
        num_chunks = int(num_chunks)
        # Filter samples: only those where index % num_chunks == chunk_id
        all_samples = [s for i, s in enumerate(all_samples) if i % num_chunks == chunk_id]
        print(f"[CHUNKED] Filtered to {len(all_samples)} samples for chunk {chunk_id}/{num_chunks}")
    
    total_samples = len(all_samples)
    print(f"Total samples to process: {total_samples}")
    
    num_workers = getattr(config.execution, 'num_workers', 1)
    print(f"Starting parallel execution with {num_workers} workers...")

    # Parallel Processing Loop
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
        # Submit all tasks
        futures = {executor.submit(process_single_sample, sample, config, synth_manager, vc_device): sample for sample in all_samples}
        
        for future in tqdm(concurrent.futures.as_completed(futures), total=total_samples, desc="Processing (Parallel)"):
            source_count += 1
            try:
                results, skipped = future.result()
                skip_count += skipped
                working_set.extend(results)
            except Exception as e:
                print(f"Task failed: {e}")
                # import traceback
                # traceback.print_exc()
                skip_count += 1
    
    print(f"\n  Sources loaded: {source_count}")
    print(f"  Skipped: {skip_count}")
    print(f"  Working set: {len(working_set)} samples")
    
    if len(working_set) == 0:
        print("[ERROR] No samples in working set!")
        return False
    
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

    # ============== PHASE 9: FEATURE EXTRACTION ==============
    print("\n" + "-" * 40)
    print("PHASE 9: Feature Extraction")
    print("-" * 40)
    
    # Create feature processing directories
    features_dir = output_path / "features"
    cqt_dir = features_dir / "cqt"
    lfcc_dir = features_dir / "lfcc"
    cqt_dir.mkdir(parents=True, exist_ok=True)
    lfcc_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize Extractors
    print("Initializing feature extractors...")
    try:
        cqt_extractor = CQTExtractor(
            n_bins=config.features.cqt.n_bins,
            hop_length=config.features.cqt.hop_length,
            fmin=config.features.cqt.fmin,
            device=config.features.device
        )
        lfcc_extractor = LFCCExtractor(
            n_lfcc=config.features.lfcc.n_lfcc,
            n_filters=config.features.lfcc.n_filters,
            n_fft=config.features.lfcc.n_fft,
            hop_length=config.features.lfcc.hop_length,
            device=config.features.device
        )
        
        print(f"Extracting features for {len(working_set)} samples...")
        feat_count = 0
        
        for sample in tqdm(working_set, desc="Extracting Features", unit="file"):
            audio = sample["audio"]
            # Ensure float32
            if audio.dtype != np.float32:
                audio = audio.astype(np.float32)
                
            # Extract CQT
            try:
                cqt = cqt_extractor.extract(audio)
                # Save as .pt
                cqt_path = cqt_dir / f"{sample['source_id']}_chunk{sample['chunk_idx']:03d}_{'syn' if sample['is_synthetic'] else 'real'}.pt"
                torch.save(torch.from_numpy(cqt), cqt_path)
            except Exception as e:
                print(f"Failed CQT for {sample['source_id']}: {e}")

            # Extract LFCC
            try:
                lfcc = lfcc_extractor.extract(audio)
                # Save as .pt
                lfcc_path = lfcc_dir / f"{sample['source_id']}_chunk{sample['chunk_idx']:03d}_{'syn' if sample['is_synthetic'] else 'real'}.pt"
                torch.save(torch.from_numpy(lfcc), lfcc_path)
            except Exception as e:
                print(f"Failed LFCC for {sample['source_id']}: {e}")
                
            feat_count += 1
            
        print(f"Feature extraction complete. Saved to {features_dir}")
        
    except Exception as e:
        print(f"[ERROR] Feature extraction initialized failed: {e}")
        import traceback
        traceback.print_exc()
    
    # ============== DONE ==============
    # ============== DONE ==============
    elapsed = time.time() - start_time
    total_mb_processed = sum(s.get('original_size', 0) for s in working_set) / 1024 / 1024
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
    print(f"Total samples: {len(working_set)}")
    print(f"Pipeline finished successfully.")
    
    return True



if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run TTS/RVC data generation pipeline")
    parser.add_argument('--chunk-id', type=int, default=None, 
                        help='Chunk ID for array job parallelization (0 to num-chunks-1)')
    parser.add_argument('--num-chunks', type=int, default=None,
                        help='Total number of chunks for array job parallelization')
    args = parser.parse_args()
    
    # Pass chunk info to pipeline via environment or globals
    if args.chunk_id is not None and args.num_chunks is not None:
        os.environ['PIPELINE_CHUNK_ID'] = str(args.chunk_id)
        os.environ['PIPELINE_NUM_CHUNKS'] = str(args.num_chunks)
        print(f"[Pipeline] Running in CHUNKED mode: chunk {args.chunk_id} of {args.num_chunks}")
    
    success = run_mozilla_cv_pipeline()
    exit(0 if success else 1)
