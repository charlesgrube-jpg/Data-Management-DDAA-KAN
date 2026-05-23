"""
RVC (Retrieval-based Voice Conversion) Synthesizer

Uses rvc-python to convert voice in audio while preserving content.
Alternative to TTS for generating synthetic deepfake audio.

Usage:
    from pipeline.synthesizer.rvc_synthesizer import synthesize_vc
    converted = synthesize_vc(audio, sr=16000, model_path="path/to/model.pth")
"""

import numpy as np
from pathlib import Path
from typing import Optional


def synthesize_vc(
    audio: np.ndarray,
    sr: int = 16000,
    model_path: Optional[str] = None,
    device: str = "cpu"
) -> Optional[np.ndarray]:
    """
    Convert voice in audio using RVC.
    
    Args:
        audio: Input audio array
        sr: Sample rate
        model_path: Path to RVC model (.pth file)
        device: Device to use ("cpu" or "cuda:0")
        
    Returns:
        Voice-converted audio array, or None if failed
    """
    # Check if rvc-python is available
    try:
        from rvc_python.infer import RVCInference
    except ImportError:
        print("[RVC] rvc-python not installed. Install with: pip install rvc-python")
        return None
    
    # Check model path
    if model_path is None:
        print("[RVC] No model path specified, skipping VC")
        return None
    
    if not Path(model_path).exists():
        print(f"[RVC] Model not found at {model_path}")
        return None
    
    try:
        # Initialize RVC inference
        rvc = RVCInference(device=device)
        
        # Load model
        print(f"[RVC] Loading model: {model_path}")
        rvc.load_model(model_path)
        
        # Save to temp file (RVC requires file input)
        import tempfile
        import soundfile as sf
        
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f_in:
            temp_in = f_in.name
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f_out:
            temp_out = f_out.name
        
        # Write input audio
        sf.write(temp_in, audio, sr)
        
        # Perform voice conversion
        rvc.infer_file(temp_in, temp_out)
        
        # Load converted audio
        import librosa
        converted, _ = librosa.load(temp_out, sr=sr)
        
        # Cleanup temp files
        Path(temp_in).unlink()
        Path(temp_out).unlink()
        
        print(f"[RVC] Conversion successful: {len(converted)/sr:.2f}s")
        return converted
        
    except Exception as e:
        print(f"[RVC] Voice conversion failed: {e}")
        return None


def synthesize_vc_batch(
    audios: list,
    sr: int = 16000,
    model_path: Optional[str] = None,
    device: str = "cpu"
) -> list:
    """
    Convert voice for multiple audio samples.
    
    Args:
        audios: List of audio arrays
        sr: Sample rate
        model_path: Path to RVC model
        device: Device to use
        
    Returns:
        List of converted audio arrays (None for failed items)
    """
    results = []
    for i, audio in enumerate(audios):
        converted = synthesize_vc(audio, sr, model_path, device)
        results.append(converted)
        if converted is not None:
            duration = len(converted) / sr
            print(f"  [{i+1}/{len(audios)}] Converted: {duration:.2f}s")
    return results


if __name__ == "__main__":
    # Test (requires a model)
    print("Testing RVC synthesizer...")
    
    # Generate test audio (3 seconds of random noise as placeholder)
    sr = 16000
    test_audio = np.random.randn(3 * sr) * 0.1
    
    # Try conversion (will fail without model, but tests error handling)
    model_path = "./rvc_models/test_model.pth"
    converted = synthesize_vc(test_audio, sr, model_path)
    
    if converted is not None:
        print(f"  ✓ Conversion successful: {len(converted)/sr:.2f}s")
    else:
        print(f"  ✗ Conversion failed (expected if no model available)")
        print(f"  To test properly: download an RVC model and update model_path")
