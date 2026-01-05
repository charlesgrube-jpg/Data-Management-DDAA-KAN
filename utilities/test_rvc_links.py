"""
RVC Model URL Sanity Checker

Verifies that RVC model URLs are reachable and return valid ZIP content.
Does NOT download full files - only checks HTTP headers.

Usage:
    python utilities/test_rvc_links.py [--download]
    
Options:
    --download: Actually download and extract 2-3 test models (slow, ~1GB)
"""

import requests
import sys
from pathlib import Path

# Test URLs (subset of expansion models)
TEST_MODELS = {
    "JoeBiden": "https://huggingface.co/0x3e9/Biden_RVC/resolve/main/Biden.zip",
    "BarackObama": "https://huggingface.co/0x3e9/Obama_RVC/resolve/main/Obama.zip",
    "Spongebob": "https://huggingface.co/QuickWick/Music-AI-Voices/resolve/main/SpongeBob%20SquarePants/SpongeBob%20SquarePants.zip",
}


def check_url_headers(name: str, url: str) -> dict:
    """Check HTTP headers for a URL without downloading content."""
    result = {
        "name": name,
        "url": url,
        "status": None,
        "content_type": None,
        "content_length": None,
        "is_valid": False,
        "error": None
    }
    
    try:
        response = requests.head(url, allow_redirects=True, timeout=30)
        result["status"] = response.status_code
        result["content_type"] = response.headers.get("Content-Type", "unknown")
        
        content_length = response.headers.get("Content-Length")
        if content_length:
            result["content_length"] = int(content_length)
            result["content_length_mb"] = round(int(content_length) / (1024 * 1024), 2)
        
        # Validity checks
        if response.status_code != 200:
            result["error"] = f"HTTP {response.status_code}"
        elif "text/html" in result["content_type"]:
            result["error"] = "Returned HTML (likely error page)"
        elif result["content_length"] and result["content_length"] < 1_000_000:  # < 1MB
            result["error"] = f"Suspiciously small ({result['content_length_mb']} MB)"
        else:
            result["is_valid"] = True
            
    except requests.RequestException as e:
        result["error"] = str(e)
    
    return result


def download_and_extract(name: str, url: str, output_dir: Path) -> dict:
    """Download and extract a single model for verification."""
    import zipfile
    import shutil
    
    result = {
        "name": name,
        "downloaded": False,
        "extracted": False,
        "pth_found": False,
        "index_found": False,
        "files": [],
        "output_path": None,
        "error": None
    }
    
    model_dir = output_dir / name
    zip_path = output_dir / f"{name}.zip"
    
    try:
        # Download
        print(f"  Downloading {name}...")
        response = requests.get(url, stream=True, timeout=120)
        response.raise_for_status()
        
        with open(zip_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        result["downloaded"] = True
        
        # Extract
        print(f"  Extracting {name}...")
        model_dir.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(model_dir)
        result["extracted"] = True
        
        # Find files
        pth_files = list(model_dir.rglob("*.pth"))
        index_files = list(model_dir.rglob("*.index"))
        
        result["files"] = [str(f.relative_to(model_dir)) for f in pth_files + index_files]
        result["pth_found"] = len(pth_files) > 0
        result["index_found"] = len(index_files) > 0
        result["output_path"] = str(model_dir)
        
        # Cleanup zip
        zip_path.unlink()
        
    except Exception as e:
        result["error"] = str(e)
        if zip_path.exists():
            zip_path.unlink()
    
    return result


def main():
    download_mode = "--download" in sys.argv
    
    print("=" * 60)
    print("RVC Model URL Sanity Check")
    print("=" * 60)
    
    if not download_mode:
        print("\nMode: Header Check Only (no downloads)")
        print("-" * 40)
        
        all_valid = True
        for name, url in TEST_MODELS.items():
            result = check_url_headers(name, url)
            
            if result["is_valid"]:
                print(f"✅ {name}:")
                print(f"   Status: {result['status']}")
                print(f"   Content-Type: {result['content_type']}")
                print(f"   Size: {result.get('content_length_mb', 'unknown')} MB")
            else:
                print(f"❌ {name}:")
                print(f"   Error: {result['error']}")
                all_valid = False
        
        print("\n" + "=" * 60)
        if all_valid:
            print("All URLs validated successfully!")
        else:
            print("Some URLs failed validation.")
        
    else:
        print("\nMode: Full Download + Extract (TEST ONLY)")
        print("-" * 40)
        
        output_dir = Path("models/rvc_test")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        for name, url in list(TEST_MODELS.items())[:2]:  # Only first 2
            result = download_and_extract(name, url, output_dir)
            
            if result["pth_found"]:
                print(f"✅ {name}:")
                print(f"   Path: {result['output_path']}")
                print(f"   Files: {result['files']}")
            else:
                print(f"❌ {name}:")
                print(f"   Error: {result['error']}")
        
        print(f"\nTest models saved to: {output_dir.absolute()}")


if __name__ == "__main__":
    main()
