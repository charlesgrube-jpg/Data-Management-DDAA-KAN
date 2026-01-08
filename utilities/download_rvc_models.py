"""
RVC Model Downloader

Downloads a standard set of community RVC models using system curl.
"""

import os
import zipfile
import shutil
import subprocess
from pathlib import Path

# Format: "ModelName": "DirectDownloadURL"
MODELS = {
    "TaylorSwift": "https://huggingface.co/bennetJL/TaylorSwift/resolve/main/TaylorSwift2024.zip?download=true",
    "EdSheeran": "https://huggingface.co/SUP3RMASS1VE/Ed-Sheeran/resolve/main/Ed%20Sheeran.zip?download=true",
    "KanyeWest": "https://huggingface.co/TheRealheavy/KanyeWestGraduationEra/resolve/main/KanyeWestGraduation.zip?download=true"
}

MODELS_DIR = Path(__file__).parent.parent / "models" / "rvc"

def setup_models():
    print(f"🚀 Setting up {len(MODELS)} RVC Models in {MODELS_DIR}...")
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    
    for name, url in MODELS.items():
        model_folder = MODELS_DIR / name
        # Check if already installed (heuristic: contains a .pth file)
        if model_folder.exists() and list(model_folder.glob("*.pth")):
            print(f"✅ {name} already exists. Skipping.")
            continue
            
        print(f"\n📥 Downloading {name}...")
        zip_path = MODELS_DIR / f"{name}.zip"
        
        try:
            # Use curl
            # -L: Follow redirects
            # -f: Fail silently on errors (exit code > 0)
            subprocess.run(["curl", "-L", "-o", str(zip_path), url], check=True)
            
            # Check size (if it's < 1KB, it's likely still a pointer or error page)
            if zip_path.stat().st_size < 1000:
                print(f"❌ Error: Downloaded file too small ({zip_path.stat().st_size} bytes). Likely an LFS pointer or error.")
                with open(zip_path, 'r') as f:
                    print(f"Content: {f.read()}")
                continue
            
            print(f"📦 Extracting {name}...")
            
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                # Extract to a temp extracted folder
                extract_root = MODELS_DIR / f"{name}_extracted"
                if extract_root.exists():
                    shutil.rmtree(extract_root)
                zip_ref.extractall(extract_root)
                
                # Find .pth and .index
                pth_files = list(extract_root.rglob("*.pth"))
                index_files = list(extract_root.rglob("*.index"))
                
                if not pth_files:
                    print(f"❌ Error: No .pth file found in {name} zip!")
                    pass # Don't return, try next
                else:
                    # Move to final folder
                    model_folder.mkdir(exist_ok=True)
                    
                    # Select largest .pth
                    source_pth = max(pth_files, key=lambda p: p.stat().st_size)
                    shutil.move(str(source_pth), str(model_folder / f"{name}.pth"))
                    print(f"   -> Secured model: {name}.pth")
                    
                    # Move index if exists
                    if index_files:
                        source_index = max(index_files, key=lambda p: p.stat().st_size)
                        shutil.move(str(source_index), str(model_folder / f"{name}.index"))
                        print(f"   -> Secured index: {name}.index")
                        
                    print(f"✅ Installed {name}")

            # Cleanup zip and temp dir
            if zip_path.exists():
                zip_path.unlink()
            if extract_root.exists():
                shutil.rmtree(extract_root)
                
        except Exception as e:
            print(f"❌ Failed to install {name}: {e}")

    print("\n✨ RVC Model Setup Complete!")

if __name__ == "__main__":
    setup_models()
