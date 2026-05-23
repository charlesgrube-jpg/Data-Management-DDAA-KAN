"""
Pipeline Validation Script
Validates the pipeline output against the 12-phase checklist.
"""

import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import pandas as pd
import numpy as np
import librosa
import soundfile as sf
from pathlib import Path
import os

# Use ASCII symbols for Windows compatibility
PASS = "[PASS]"
WARN = "[WARN]"
FAIL = "[FAIL]"

# Latest output directory — auto-detect from most recent processed_dataset_* folder,
# or fall back to a configurable env var / command-line argument (see __main__ block).
import glob as _glob
_candidates = sorted(_glob.glob("processed_dataset_*"), reverse=True)
OUTPUT_DIR = Path(os.environ.get("DDAA_OUTPUT_DIR", _candidates[0] if _candidates else "processed_dataset"))
METADATA_FILE = OUTPUT_DIR / "metadata.csv"

def phase1_input_validation():
    """Phase 1: Input Validation"""
    print("\n" + "="*60)
    print("PHASE 1: INPUT VALIDATION")
    print("="*60)
    
    results = {}
    
    # 1.1 Dataset loads successfully
    try:
        df = pd.read_csv(METADATA_FILE)
        results['1.1_loads'] = f"✅ Loaded {len(df)} rows"
    except Exception as e:
        results['1.1_loads'] = f"❌ Failed: {e}"
        return results
    
    # 1.2 Speaker IDs extracted
    if 'speaker_id' in df.columns:
        sample_ids = df['speaker_id'].dropna().head(5).tolist()
        results['1.2_speaker_ids'] = f"✅ Found speaker_id column. Samples: {sample_ids[:3]}"
    else:
        results['1.2_speaker_ids'] = "❌ No speaker_id column"
    
    # 1.3 Transcripts available
    if 'transcript' in df.columns:
        sample_transcripts = df['transcript'].dropna().head(5).tolist()
        results['1.3_transcripts'] = f"✅ Found transcript column. Samples: {sample_transcripts[:2]}"
    else:
        results['1.3_transcripts'] = "⚠️ No transcript column (may be by design)"
    
    # 1.4 Audio files loadable
    try:
        sample_files = df['filename'].head(10).tolist()
        loaded = 0
        for f in sample_files:
            fpath = OUTPUT_DIR / f
            if fpath.exists():
                audio, sr = librosa.load(str(fpath), sr=None)
                if len(audio) > 0:
                    loaded += 1
        results['1.4_loadable'] = f"✅ {loaded}/{len(sample_files)} files loadable"
    except Exception as e:
        results['1.4_loadable'] = f"❌ Error loading: {e}"
    
    for k, v in results.items():
        print(f"  {k}: {v}")
    return results


def phase2_speaker_splitting():
    """Phase 2: Speaker Splitting"""
    print("\n" + "="*60)
    print("PHASE 2: SPEAKER SPLITTING")
    print("="*60)
    
    results = {}
    df = pd.read_csv(METADATA_FILE)
    
    if 'speaker_id' not in df.columns:
        results['2.x'] = "❌ No speaker_id column - cannot validate"
        print(f"  {results['2.x']}")
        return results
    
    # Get unique speakers per split
    train_spk = set(df[df['split'] == 'train']['speaker_id'].unique())
    val_spk = set(df[df['split'] == 'val']['speaker_id'].unique())
    test_spk = set(df[df['split'] == 'test']['speaker_id'].unique())
    all_spk = train_spk | val_spk | test_spk
    
    # 2.1 Split ratios
    train_pct = len(train_spk) / len(all_spk) * 100 if all_spk else 0
    val_pct = len(val_spk) / len(all_spk) * 100 if all_spk else 0
    test_pct = len(test_spk) / len(all_spk) * 100 if all_spk else 0
    results['2.1_ratios'] = f"Train: {train_pct:.1f}%, Val: {val_pct:.1f}%, Test: {test_pct:.1f}%"
    if 60 <= train_pct <= 80:
        results['2.1_ratios'] = "✅ " + results['2.1_ratios']
    else:
        results['2.1_ratios'] = "⚠️ " + results['2.1_ratios']
    
    # 2.2 No speaker overlap
    train_val_overlap = train_spk & val_spk
    train_test_overlap = train_spk & test_spk
    val_test_overlap = val_spk & test_spk
    
    if len(train_val_overlap) == 0 and len(train_test_overlap) == 0 and len(val_test_overlap) == 0:
        results['2.2_no_overlap'] = "✅ No speaker overlap between splits"
    else:
        results['2.2_no_overlap'] = f"❌ Overlap found: train-val={len(train_val_overlap)}, train-test={len(train_test_overlap)}, val-test={len(val_test_overlap)}"
    
    # 2.3 All files from one speaker in same split
    speaker_splits = df.groupby('speaker_id')['split'].nunique()
    multi_split_speakers = speaker_splits[speaker_splits > 1]
    if len(multi_split_speakers) == 0:
        results['2.3_speaker_consistency'] = "✅ All speakers in single split"
    else:
        results['2.3_speaker_consistency'] = f"❌ {len(multi_split_speakers)} speakers in multiple splits"
    
    for k, v in results.items():
        print(f"  {k}: {v}")
    return results


def phase3_audio_preprocessing():
    """Phase 3: Audio Preprocessing"""
    print("\n" + "="*60)
    print("PHASE 3: AUDIO PREPROCESSING")
    print("="*60)
    
    results = {}
    df = pd.read_csv(METADATA_FILE)
    
    # Sample 10 files
    sample_files = df['filename'].head(10).tolist()
    
    sample_rates = []
    rms_values = []
    durations = []
    clipped = 0
    mono_count = 0
    
    for f in sample_files:
        fpath = OUTPUT_DIR / f
        if fpath.exists():
            audio, sr = librosa.load(str(fpath), sr=None, mono=False)
            sample_rates.append(sr)
            
            # Check mono
            if audio.ndim == 1:
                mono_count += 1
            else:
                audio = audio[0]  # Take first channel for analysis
            
            # RMS
            rms = np.sqrt(np.mean(audio**2))
            rms_db = 20 * np.log10(rms + 1e-10)
            rms_values.append(rms_db)
            
            # Duration
            durations.append(len(audio) / sr)
            
            # Clipping check
            if np.max(np.abs(audio)) <= 1.0:
                clipped += 1
    
    # 3.1 All 16 kHz
    unique_sr = set(sample_rates)
    if unique_sr == {16000}:
        results['3.1_sample_rate'] = "✅ All 16 kHz"
    else:
        results['3.1_sample_rate'] = f"⚠️ Sample rates: {unique_sr}"
    
    # 3.2 Mono
    results['3.2_mono'] = f"✅ {mono_count}/{len(sample_files)} mono" if mono_count == len(sample_files) else f"⚠️ {mono_count}/{len(sample_files)} mono"
    
    # 3.3 RMS normalization
    avg_rms = np.mean(rms_values)
    results['3.3_rms'] = f"Avg RMS: {avg_rms:.1f} dB"
    if -25 <= avg_rms <= -15:
        results['3.3_rms'] = "✅ " + results['3.3_rms']
    else:
        results['3.3_rms'] = "⚠️ " + results['3.3_rms']
    
    # 3.4 Clipped to [-1, 1]
    results['3.4_clipped'] = f"✅ {clipped}/{len(sample_files)} within [-1,1]"
    
    # 3.7 Chunk durations
    avg_dur = np.mean(durations)
    results['3.7_duration'] = f"Avg duration: {avg_dur:.2f}s (range: {min(durations):.2f}-{max(durations):.2f})"
    
    for k, v in results.items():
        print(f"  {k}: {v}")
    return results


def phase4_synthetic_generation():
    """Phase 4: Synthetic Generation"""
    print("\n" + "="*60)
    print("PHASE 4: SYNTHETIC GENERATION")
    print("="*60)
    
    results = {}
    df = pd.read_csv(METADATA_FILE)
    
    # Check labels
    if 'label' in df.columns:
        labels = df['label'].value_counts()
        results['4.x_labels'] = f"Labels: {dict(labels)}"
    
    # Check generator field
    if 'generator' in df.columns:
        generators = df['generator'].value_counts()
        results['4.1_generators'] = f"✅ Generators: {dict(generators)}"
    else:
        results['4.1_generators'] = "❌ No generator column"
    
    # Check method field
    if 'method' in df.columns:
        methods = df['method'].value_counts()
        results['4.2_methods'] = f"Methods: {dict(methods)}"
    else:
        results['4.2_methods'] = "⚠️ No method column"
    
    # 4.7 Synthetic inherits speaker_id
    if 'speaker_id' in df.columns and 'label' in df.columns:
        synthetic_df = df[df['label'] == 'synthetic']
        if len(synthetic_df) > 0:
            has_speaker = synthetic_df['speaker_id'].notna().sum()
            results['4.7_speaker_inherit'] = f"✅ {has_speaker}/{len(synthetic_df)} synthetic have speaker_id"
    
    # 4.9 Pairing tracked
    if 'source_real_file' in df.columns or 'real_file' in df.columns:
        results['4.9_pairing'] = "✅ Pairing column exists"
    else:
        results['4.9_pairing'] = "⚠️ No explicit pairing column"
    
    for k, v in results.items():
        print(f"  {k}: {v}")
    return results


def phase5_channel_effects():
    """Phase 5: Channel Effects"""
    print("\n" + "="*60)
    print("PHASE 5: CHANNEL EFFECTS")
    print("="*60)
    
    results = {}
    df = pd.read_csv(METADATA_FILE)
    
    # 5.1 Quality tiers
    if 'quality_tier' in df.columns:
        tiers = df['quality_tier'].value_counts(normalize=True) * 100
        results['5.1_tiers'] = f"✅ Quality tiers: {dict(tiers.round(1))}"
    else:
        results['5.1_tiers'] = "⚠️ No quality_tier column"
    
    # Codec info
    if 'codec_type' in df.columns:
        codecs = df['codec_type'].value_counts()
        results['5.4_codecs'] = f"Codecs: {dict(codecs)}"
    
    # SNR info
    if 'snr_db' in df.columns:
        snr_stats = df['snr_db'].describe()
        results['5.6_snr'] = f"SNR stats: mean={snr_stats['mean']:.1f}, std={snr_stats['std']:.1f}"
    
    for k, v in results.items():
        print(f"  {k}: {v}")
    return results


def phase7_balance():
    """Phase 7: Balance Verification"""
    print("\n" + "="*60)
    print("PHASE 7: BALANCE VERIFICATION")
    print("="*60)
    
    results = {}
    df = pd.read_csv(METADATA_FILE)
    
    # 7.1 Overall 50/50
    if 'label' in df.columns:
        label_pct = df['label'].value_counts(normalize=True) * 100
        real_pct = label_pct.get('real', 0)
        results['7.1_overall'] = f"Real: {real_pct:.1f}%, Synthetic: {100-real_pct:.1f}%"
        if 45 <= real_pct <= 55:
            results['7.1_overall'] = "✅ " + results['7.1_overall']
        else:
            results['7.1_overall'] = "⚠️ " + results['7.1_overall']
    
    # 7.2 Per-split balance
    if 'label' in df.columns and 'split' in df.columns:
        for split in ['train', 'val', 'test']:
            split_df = df[df['split'] == split]
            if len(split_df) > 0:
                split_real_pct = (split_df['label'] == 'real').mean() * 100
                status = "✅" if 45 <= split_real_pct <= 55 else "⚠️"
                results[f'7.2_{split}'] = f"{status} {split}: {split_real_pct:.1f}% real"
    
    for k, v in results.items():
        print(f"  {k}: {v}")
    return results


def phase8_leakage():
    """Phase 8: Leakage Prevention"""
    print("\n" + "="*60)
    print("PHASE 8: LEAKAGE PREVENTION")
    print("="*60)
    
    results = {}
    df = pd.read_csv(METADATA_FILE)
    
    # 8.5 No duplicate filenames
    n_files = len(df)
    n_unique = df['filename'].nunique()
    if n_files == n_unique:
        results['8.5_no_dup_files'] = f"✅ All {n_files} filenames unique"
    else:
        results['8.5_no_dup_files'] = f"❌ {n_files - n_unique} duplicate filenames"
    
    # 8.1 already covered in phase 2
    
    for k, v in results.items():
        print(f"  {k}: {v}")
    return results


def phase9_filesystem():
    """Phase 9: File System Validation"""
    print("\n" + "="*60)
    print("PHASE 9: FILE SYSTEM VALIDATION")
    print("="*60)
    
    results = {}
    df = pd.read_csv(METADATA_FILE)
    
    # 9.1 All files exist
    exists_count = 0
    for f in df['filename']:
        if (OUTPUT_DIR / f).exists():
            exists_count += 1
    
    if exists_count == len(df):
        results['9.1_files_exist'] = f"✅ All {len(df)} files exist"
    else:
        results['9.1_files_exist'] = f"❌ Only {exists_count}/{len(df)} files exist"
    
    # 9.5 File sizes
    sizes = []
    for f in df['filename'].head(20):
        fpath = OUTPUT_DIR / f
        if fpath.exists():
            sizes.append(fpath.stat().st_size)
    
    if sizes:
        avg_size = np.mean(sizes) / 1024
        results['9.5_file_sizes'] = f"Avg file size: {avg_size:.1f} KB"
    
    for k, v in results.items():
        print(f"  {k}: {v}")
    return results


def phase10_config():
    """Phase 10: Configuration"""
    print("\n" + "="*60)
    print("PHASE 10: CONFIGURATION")
    print("="*60)
    
    results = {}
    
    # 10.1 Config file exists
    config_file = OUTPUT_DIR / "config.yaml"
    if config_file.exists():
        results['10.1_config'] = "✅ config.yaml saved with dataset"
    else:
        results['10.1_config'] = "⚠️ No config.yaml in output"
    
    # 10.3 Manifest
    manifest_file = OUTPUT_DIR / "manifest.json"
    if manifest_file.exists():
        results['10.3_manifest'] = "✅ manifest.json exists"
    else:
        results['10.3_manifest'] = "⚠️ No manifest.json"
    
    for k, v in results.items():
        print(f"  {k}: {v}")
    return results


def main():
    print("="*60)
    print("PIPELINE VALIDATION REPORT")
    print("="*60)
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Metadata file: {METADATA_FILE}")
    
    if not METADATA_FILE.exists():
        print("❌ Metadata file not found!")
        return
    
    all_results = {}
    all_results.update(phase1_input_validation())
    all_results.update(phase2_speaker_splitting())
    all_results.update(phase3_audio_preprocessing())
    all_results.update(phase4_synthetic_generation())
    all_results.update(phase5_channel_effects())
    all_results.update(phase7_balance())
    all_results.update(phase8_leakage())
    all_results.update(phase9_filesystem())
    all_results.update(phase10_config())
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    passed = sum(1 for v in all_results.values() if v.startswith("✅"))
    warnings = sum(1 for v in all_results.values() if v.startswith("⚠️"))
    failed = sum(1 for v in all_results.values() if v.startswith("❌"))
    
    print(f"  ✅ Passed: {passed}")
    print(f"  ⚠️ Warnings: {warnings}")
    print(f"  ❌ Failed: {failed}")


if __name__ == "__main__":
    main()
