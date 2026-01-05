"""
Verify Data Leakage Script

Analyzes the metadata.csv/json produced by the pipeline to ensure:
1. STRICT SPEAKER DISJOINTNESS: No real speaker appears in both Train and Test.
2. STRICT GENERATOR DISJOINTNESS: No synthetic voice appears in both Train and Test.
3. TEST SET PURITY: Ensure 'train-only' models (MMS) did not leak into Test.
"""

import json
import pandas as pd
from pathlib import Path
import sys

def verify_dataset(output_dir: str):
    output_path = Path(output_dir)
    metadata_path = output_path / "metadata.csv"
    
    if not metadata_path.exists():
        # Try metadata.json
        metadata_path = output_path / "metadata.json"
        if not metadata_path.exists():
            print(f"[ERROR] No metadata file found in {output_dir}")
            return False
            
    print(f"Loading metadata from {metadata_path}...")
    
    if metadata_path.suffix == '.csv':
        df = pd.read_csv(metadata_path)
    else:
        with open(metadata_path, 'r') as f:
            data = json.load(f)
        df = pd.DataFrame(data)
        
    print(f"Loaded {len(df)} samples.")
    
    # 1. Verify Real Speaker Disjointness
    print("\n[1] Verifying Real Speaker Disjointness...")
    train_speakers = set(df[df['split'] == 'train']['speaker_id'].unique())
    test_speakers = set(df[df['split'] == 'test']['speaker_id'].unique())
    
    speaker_overlap = train_speakers.intersection(test_speakers)
    if speaker_overlap:
        print(f"❌ FAILED: Found {len(speaker_overlap)} speakers in both Train/Test!")
        print(f"   Examples: {list(speaker_overlap)[:5]}")
    else:
        print("✅ PASSED: Real speakers are strictly disjoint.")

    # 2. Verify Synthetic Voice Disjointness
    print("\n[2] Verifying Synthetic Voice Disjointness...")
    syn_df = df[df['is_synthetic'] == True]
    
    train_voices = set(syn_df[syn_df['split'] == 'train']['generator'].unique())
    test_voices = set(syn_df[syn_df['split'] == 'test']['generator'].unique())
    
    # Filter out None/NaN if any
    train_voices = {v for v in train_voices if pd.notna(v)}
    test_voices = {v for v in test_voices if pd.notna(v)}
    
    voice_overlap = train_voices.intersection(test_voices)
    
    print(f"   Train Voices: {len(train_voices)}")
    print(f"   Test Voices:  {len(test_voices)}")
    
    if voice_overlap:
        print(f"❌ FAILED: Found {len(voice_overlap)} voices in both Train/Test!")
        print(f"   Leakage: {voice_overlap}")
    else:
        print("✅ PASSED: Synthetic voices are strictly disjoint.")
        
    # 3. Verify MMS Exclusion
    print("\n[3] Verifying Train-Only Model Exclusion (MMS)...")
    mms_in_test = [v for v in test_voices if 'mms' in str(v).lower()]
    
    if mms_in_test:
        print(f"❌ FAILED: Found MMS in Test set!")
        print(f"   Violations: {mms_in_test}")
    else:
        print("✅ PASSED: MMS is successfully banned from Test.")

    # 4. Verify RVC Model Disjointness (NEW - V4)
    print("\n[4] Verifying RVC Model Disjointness...")
    
    # Filter to VC-generated samples (check both 'vc' and 'rvc' tags)
    vc_df = df[df['method'].isin(['vc', 'rvc'])]
    
    if len(vc_df) == 0:
        print("   (No VC samples found, skipping)")
    else:
        # Normalize paths for consistent matching
        def normalize_path(p):
            from pathlib import Path
            return Path(str(p)).resolve().as_posix().lower() if pd.notna(p) else ""
        
        train_vc = set(vc_df[vc_df['split'] == 'train']['generator'].apply(normalize_path))
        test_vc = set(vc_df[vc_df['split'] == 'test']['generator'].apply(normalize_path))
        val_vc = set(vc_df[vc_df['split'] == 'val']['generator'].apply(normalize_path))
        
        # Remove empty strings
        train_vc.discard("")
        test_vc.discard("")
        val_vc.discard("")
        
        # Check overlaps
        train_test = train_vc & test_vc
        train_val = train_vc & val_vc
        
        print(f"   Train VC Models: {len(train_vc)}")
        print(f"   Test VC Models:  {len(test_vc)}")
        print(f"   Val VC Models:   {len(val_vc)}")
        
        if train_test or train_val:
            print(f"❌ FAILED: RVC model overlap detected!")
            if train_test:
                print(f"   Train↔Test: {train_test}")
            if train_val:
                print(f"   Train↔Val: {train_val}")
        else:
            print("✅ PASSED: RVC models are strictly disjoint.")

    return True

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python verify_leakage.py <path_to_output_dir>")
        print("Example: python verify_leakage.py processed_dataset_20250103_120000")
    else:
        verify_dataset(sys.argv[1])
