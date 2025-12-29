
import os
import sys
import numpy as np
import scipy.io.wavfile as wav

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from pipeline.synthesizer.huggingface_synthesizer import HuggingFaceSynthesizer

def test_synthesis():
    print("🧪 Testing Hugging Face Synthesis Module...")
    print("⏳ This may take a while as models download (Bark ~4GB, MMS ~1GB)...")
    
    # Models to test
    models = [
        "microsoft/speecht5_tts", # Vocoder (Light)
        "facebook/mms-tts-eng",   # Flow (Medium)
        "suno/bark"             # Codec (Heavy)
    ]
    
    text = "This is a synthetic voice test."
    output_dir = "test_synthesis_output"
    os.makedirs(output_dir, exist_ok=True)
    
    # Check for huggingface_hub for explicit progress bars
    try:
        from huggingface_hub import snapshot_download
        HAS_HF_HUB = True
    except ImportError:
        HAS_HF_HUB = False
        print("⚠️ huggingface_hub not found, download progress might be hidden.")

    for model_name in models:
        print(f"\n────────────────────────────────────────")
        print(f"🔄 Processing Model: {model_name}")
        
        # Explicit download step for UX
        if HAS_HF_HUB:
            print(f"⬇️  Checking/Downloading weights for {model_name}...")
            # snapshot_download automatically shows a tqdm progress bar
            try:
                snapshot_download(repo_id=model_name)
            except Exception as e:
                print(f"⚠️ Download warning: {e}")

        synth = HuggingFaceSynthesizer(model_name)
        
        try:
            print(f"  generating with {model_name}...")
            audio = synth.synthesize(text)
            
            if audio is not None:
                duration = len(audio) / 16000 # Estimate assuming 16k
                print(f"  ✅ Success! Generated {duration:.2f}s of audio.")
                print(f"  Shape: {audio.shape}, Type: {audio.dtype}")
                
                # Save just to prove it works
                filename = model_name.replace("/", "_") + ".wav"
                path = os.path.join(output_dir, filename)
                # Normalize just for listening check
                if audio.dtype == np.float32:
                     audio = (audio * 32767).astype(np.int16)
                
                # Bark is 24k, others 16k usually. But for a quick test 16k write is fine or we can guess.
                # Actually HuggingFaceSynthesizer usually returns numpy arrays that are float.
                wav.write(path, 24000 if "bark" in model_name else 16000, audio)
                print(f"  Saved to {path}")
            else:
                print("  ❌ Failed (Returned None)")
                
        except Exception as e:
            print(f"  🔥 Exception: {e}")

if __name__ == "__main__":
    test_synthesis()
