"""
RVC Model Downloader

Downloads a standard set of community RVC models for research/testing purposes.
Models selected to represent diverse vocal characteristics (Gender/Pitch/Timbre).

Models:
1. Dua Lipa (Female, Pop)
2. Taylor Swift (Female, Clear)
3. Ed Sheeran (Male, Soft)
4. Kanye West (Male, Deep)
"""

import os
import requests
import zipfile
import shutil
from pathlib import Path
from tqdm import tqdm

# Mapping: Model Name -> Download URL
MODELS = {
    "DuaLipa": "https://huggingface.co/gilliaaan/DuaLipa/resolve/main/DuaLipa.zip",
    "TaylorSwift": "https://huggingface.co/bennetJL/TaylorSwift/resolve/main/TaylorSwift2024.zip?download=true",
    "EdSheeran": "https://huggingface.co/SUP3RMASS1VE/Ed-Sheeran/resolve/main/Ed%20Sheeran.zip?download=true",
    "KanyeWest": "https://huggingface.co/TheRealheavy/KanyeWestGraduationEra/resolve/main/KanyeWestGraduation.zip?download=true"
}

MODELS_DIR = Path(__file__).parent.parent / "models" / "rvc"

def download_file(url, dest_path):
    """Download file with progress bar."""
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    block_size = 1024 * 1024  # 1MB
    
    with open(dest_path, "wb") as file, tqdm(
        desc=dest_path.name,
        total=total_size,
        unit='iB',
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for data in response.iter_content(block_size):
            size = file.write(data)
            bar.update(size)

def setup_models():
    print(f"🚀 Setting up RVC Models in {MODELS_DIR}...")
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    
    for name, url in MODELS.items():
        model_folder = MODELS_DIR / name
        if model_folder.exists() and (model_folder / f"{name}.pth").exists():
            print(f"✅ {name} already exists. Skipping.")
            continue
            
        print(f"\n📥 Downloading {name}...")
        zip_path = MODELS_DIR / f"{name}.zip"
        
        try:
            download_file(url, zip_path)
            
            print(f"📦 Extracting {name}...")
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                # Extract to a temp extracted folder first to handle weird zip structures
                extract_root = MODELS_DIR / f"{name}_extracted"
                zip_ref.extractall(extract_root)
                
                # Find the .pth and .index files
                pth_files = list(extract_root.rglob("*.pth"))
                index_files = list(extract_root.rglob("*.index"))
                
                if not pth_files:
                    print(f"❌ Error: No .pth file found in {name} zip!")
                    continue
                
                # Move files to final folder
                model_folder.mkdir(exist_ok=True)
                
                # Move main model (rename to match folder for simplicity)
                source_pth = pth_files[0] # Take largest if multiple? Usually just one.
                # Heuristic: Take the largest pth file (often there are small G/D files for training)
                if len(pth_files) > 1:
                     source_pth = max(pth_files, key=lambda p: p.stat().st_size)
                
                shutil.move(str(source_pth), str(model_folder / f"{name}.pth"))
                print(f"   -> Secured model: {name}.pth")
                
                # Move index if exists
                if index_files:
                    source_index = max(index_files, key=lambda p: p.stat().st_size) # Best index
                    shutil.move(str(source_index), str(model_folder / f"{name}.index"))
                    print(f"   -> Secured index: {name}.index")
                
            # Cleanup
            zip_path.unlink()
            shutil.rmtree(extract_root)
            print(f"✅ Installed {name}")
            
        except Exception as e:
            print(f"❌ Failed to install {name}: {e}")
            if zip_path.exists():
                zip_path.unlink()
            if (MODELS_DIR / f"{name}_extracted").exists():
                shutil.rmtree(MODELS_DIR / f"{name}_extracted")

    print("\n✨ All RVC models setup complete!")

if __name__ == "__main__":
    setup_models()
