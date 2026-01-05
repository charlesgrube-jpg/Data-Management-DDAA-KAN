"""
Unquarantine and Scan Script

Scans all files in models/rvc/quarantine using the FIXED pth_security module.
If safe, restores them to their original location.
Mappings are inferred from rvc_model_registry.xml or filename heuristics.
"""

import sys
import shutil
import yaml
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
sys.path.append(str(REPO_ROOT))

from pipeline.utils.pth_security import scan_pth_file, is_safe_to_load

MODELS_DIR = REPO_ROOT / "models" / "rvc"
QUARANTINE_DIR = MODELS_DIR / "quarantine"
REGISTRY_PATH = REPO_ROOT / "utilities" / "rvc_model_registry.yaml"

def load_registry_map():
    """Create a map of filename -> model_id."""
    with open(REGISTRY_PATH, 'r') as f:
        registry = yaml.safe_load(f)
    
    file_map = {}
    for group in ['baseline', 'expansion']:
        for model in registry.get(group, []):
            for fname in model.get('expected_files', []):
                file_map[fname] = model['id']
                # also map lowercase
                file_map[fname.lower()] = model['id']
    return file_map

def restore_file(quarantined_path: Path, model_id: str, original_name: str):
    target_dir = MODELS_DIR / model_id
    target_dir.mkdir(parents=True, exist_ok=True)
    dest = target_dir / original_name
    
    print(f"   ✨ Restoring to {dest}...")
    shutil.move(quarantined_path, dest)

def main():
    if not QUARANTINE_DIR.exists():
        print("No quarantine directory found.")
        return

    print(f"Scanning quarantine: {QUARANTINE_DIR}")
    files = list(QUARANTINE_DIR.glob("*_QUARANTINED*"))
    
    if not files:
        print("Quarantine is empty.")
        return

    file_map = load_registry_map()
    
    for q_file in files:
        print(f"\n🔍 Scanning {q_file.name}...")
        
        # Scan with fixed security module
        result = scan_pth_file(q_file)
        
        if result.is_safe:
            print("   ✅ CLEAN. Restoring...")
            
            # Infer original name and model ID
            # name format: originalname_QUARANTINED.pth (or .index)
            # handle repeated quarantine: _QUARANTINED_1.pth
            
            clean_name = q_file.name.replace("_QUARANTINED", "").replace("_1", "").replace("_2", "") 
            # This logic is a bit brittle if original had _QUARANTINED in logic but unlikely
            
            # Try to find model ID
            # Heuristic 1: Exact match in registry expected_files
            model_id = file_map.get(clean_name)
            if not model_id:
                model_id = file_map.get(clean_name.lower())
            
            if model_id:
                restore_file(q_file, model_id, clean_name)
            else:
                print(f"   ⚠️ Could not map {clean_name} to a Model ID. Leaving in quarantine.")
                # Fallback: check all folders for matching original filename?
                # skipping for safety.
                
        else:
            print(f"   🚨 STILL UNSAFE: {result.issues}")

    # cleanup
    if not any(QUARANTINE_DIR.iterdir()):
        print("\nCleaning up empty quarantine dir...")
        QUARANTINE_DIR.rmdir()

if __name__ == "__main__":
    main()
