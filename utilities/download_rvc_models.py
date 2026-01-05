"""
RVC Model Downloader (Robust)

Downloads RVC models defined in 'utilities/rvc_model_registry.yaml'.
Uses the centralized HFClient for robust downloads with integrity checks.

Usage:
    python utilities/download_rvc_models.py
"""

import sys
import yaml
import shutil
import zipfile
import requests
from pathlib import Path
from typing import Dict, List, Any

# Add repo root to path to import pipeline modules
REPO_ROOT = Path(__file__).parent.parent
sys.path.append(str(REPO_ROOT))

try:
    from pipeline.utils.huggingface_hub import HFClient, HFDownloadResult
    from pipeline.utils.pth_security import scan_pth_file, quarantine_file
except ImportError:
    print("❌ Critical: pipeline.utils not found. Run from repo root.")
    sys.exit(1)

REGISTRY_PATH = REPO_ROOT / "utilities" / "rvc_model_registry.yaml"
MODELS_ROOT = REPO_ROOT / "models" / "rvc"


def load_registry() -> Dict[str, Any]:
    if not REGISTRY_PATH.exists():
        print(f"❌ Registry not found at {REGISTRY_PATH}")
        sys.exit(1)
    with open(REGISTRY_PATH, 'r') as f:
        return yaml.safe_load(f)


def extract_zip(zip_path: Path, extract_to: Path) -> bool:
    """Extract zip file to destination."""
    try:
        with zipfile.ZipFile(zip_path, 'r') as zf:
            zf.extractall(extract_to)
        return True
    except Exception as e:
        print(f"   ❌ Extraction failed: {e}")
        return False


def process_model(client: HFClient, model: Dict[str, Any], group: str):
    """Download and install a single model."""
    model_id = model['id']
    url = model['url']
    
    # RVC models are usually hosted as direct file links or HF resolve links
    # We parse the URL to get repo_id and filename for HFClient
    # Expected format: https://huggingface.co/{repo_id}/resolve/main/{filename}
    
    print(f"\n📦 Processing [{group}] {model_id}...")
    
    if "huggingface.co" not in url:
        print(f"   ⚠️ Only HuggingFace URLs supported currently. Skipping {url}")
        return

    try:
        # naive parsing of HF url
        parts = url.split("huggingface.co/")[-1].split("/")
        repo_id = f"{parts[0]}/{parts[1]}"
        
        # Handle /resolve/main/ or /blob/main/
        if "resolve" in parts:
            idx = parts.index("resolve")
            revision = parts[idx+1]
            filename = "/".join(parts[idx+2:]).split("?")[0] # remove query params
        elif "blob" in parts:
            idx = parts.index("blob")
            revision = parts[idx+1]
            filename = "/".join(parts[idx+2:]).split("?")[0]
        else:
            print(f"   ⚠️ Could not parse HF URL: {url}")
            return
            
        unquoted_filename = requests.utils.unquote(filename)
        
    except Exception as e:
        print(f"   ⚠️ URL parse error: {e}")
        return

    # Target directory: models/rvc/{model_id}/
    target_dir = MODELS_ROOT / model_id
    if target_dir.exists() and any(target_dir.glob("*.pth")):
        print(f"   ✅ Already installed in {target_dir}")
        return

    # Download
    print(f"   ⬇️ Downloading {unquoted_filename} from {repo_id}...")
    result = client.download_file(
        repo_id=repo_id,
        filename=unquoted_filename,
        revision=revision,
        check_integrity=True
    )
    
    if not result.success:
        print(f"   ❌ Download failed: {result.error}")
        return
        
    # Extract
    print(f"   📂 Extracting to {target_dir}...")
    target_dir.mkdir(parents=True, exist_ok=True)
    
    if str(result.local_path).endswith('.zip'):
        success = extract_zip(result.local_path, target_dir)
        if not success:
            return
    else:
        # It's a direct .pth file?
        shutil.copy(result.local_path, target_dir / result.local_path.name)
        
    # Security Scan
    print("   🛡️ Scanning content...")
    pth_files = list(target_dir.glob("**/*.pth"))
    for pth in pth_files:
        scan = scan_pth_file(pth)
        if not scan.is_safe:
            print(f"   🚨 SECURITY WARNING: {pth.name} flagged!")
            print(f"      Issues: {scan.issues}")
            # Quarantine
            quarantine_file(pth, MODELS_ROOT / "quarantine")
        else:
            print(f"      ✅ {pth.name} passed scan.")


def main():
    import requests # needed for unquote
    
    print("="*60)
    print("RVC Model Manager")
    print("="*60)
    
    client = HFClient()
    if not client.is_available():
        print("❌ huggingface_hub not installed.")
        return

    registry = load_registry()
    
    # Process Baseline
    for model in registry.get('baseline', []):
        process_model(client, model, "BASELINE")
        
    # Process Verified Expansion
    for model in registry.get('expansion', []):
        if model.get('status') == 'verified':
            process_model(client, model, "EXPANSION")
        else:
            print(f"\n⏭️ Skipping unverified model: {model['id']}")

    print("\n" + "="*60)
    print("Done.")


if __name__ == "__main__":
    main()
