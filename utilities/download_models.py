
import os
import requests
from tqdm import tqdm

def download_file(url, filename, desc=None):
    """Download a file with progress bar."""
    if os.path.exists(filename):
        print(f"✅ {filename} already exists")
        return

    print(f"⬇️ Downloading {desc or filename}...")
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    
    with open(filename, 'wb') as file, tqdm(
        desc=filename,
        total=total_size,
        unit='iB',
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for data in response.iter_content(chunk_size=1024):
            size = file.write(data)
            bar.update(size)

def download_rvc_models():
    """Download RVC base models."""
    models = {
        "hubert_base.pt": "https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/hubert_base.pt",
        "rmvpe.pt": "https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/rmvpe.pt"
    }
    
    os.makedirs("rvc_models", exist_ok=True)
    
    for name, url in models.items():
        # Download strictly to root (where rvc-python expects them usually) or rvc_models dir
        # rvc-python often looks in local dir. 
        # Making sure to put them where logic anticipates.
        # Synthesizer config points to literal "hubert_base.pt" or "rvc_models/hubert_base.pt"
        # Let's put in root for maximum compatibility with default rvc-python, 
        # and also copy/link as needed.
        
        download_file(url, name, f"RVC Model: {name}")

if __name__ == "__main__":
    print("🚀 Downloading required models for Colab/Linux...")
    try:
        download_rvc_models()
        print("✅ Model download complete!")
    except Exception as e:
        print(f"❌ Download failed: {e}")
