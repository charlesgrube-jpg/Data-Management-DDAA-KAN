"""
Simple TTS Synthesizer using gTTS (Google Text-to-Speech)

Alternative to Coqui TTS for Python 3.14 compatibility.
Uses Google's TTS API to generate speech from text.

Usage:
    from pipeline.gtts_synthesizer import synthesize_tts
    audio = synthesize_tts("Hello world", sample_rate=16000)
"""

import numpy as np
from pathlib import Path
import tempfile
from typing import Optional


def synthesize_tts(
    text: str,
    sample_rate: int = 16000,
    language: str = "en"
) -> Optional[np.ndarray]:
    """
    Generate speech from text using Google TTS.
    
    Args:
        text: Input transcript
        sample_rate: Target sample rate for output
        language: Language code (e.g., "en", "en-us", "en-uk")
        
    Returns:
        Synthesized audio as numpy array, or None if failed
    """
    try:
        from gtts import gTTS
        import librosa
        import soundfile as sf
    except ImportError as e:
        print(f"[gTTS] Missing dependency: {e}")
        return None
    
    if not text or not text.strip():
        print("[gTTS] Empty text, skipping")
        return None
    
    try:
        # Generate speech
        tts = gTTS(text=text, lang=language, slow=False)
        
        # Save to temp file (gTTS can only save to file)
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
            temp_path = f.name
        
        tts.save(temp_path)
        
        # Load and resample
        audio, sr = librosa.load(temp_path, sr=sample_rate)
        
        # Cleanup temp file
        Path(temp_path).unlink()
        
        return audio
        
    except Exception as e:
        print(f"[gTTS] Synthesis failed: {e}")
        return None


def synthesize_batch(
    texts: list,
    sample_rate: int = 16000,
    language: str = "en"
) -> list:
    """
    Generate speech for multiple texts.
    
    Args:
        texts: List of transcripts
        sample_rate: Target sample rate
        language: Language code
        
    Returns:
        List of audio arrays (None for failed items)
    """
    results = []
    for i, text in enumerate(texts):
        audio = synthesize_tts(text, sample_rate, language)
        results.append(audio)
        if audio is not None:
            duration = len(audio) / sample_rate
            print(f"  [{i+1}/{len(texts)}] Synthesized: {duration:.2f}s")
    return results


if __name__ == "__main__":
    # Test
    test_texts = [
        "Hello, this is a test of the text to speech system.",
        "The quick brown fox jumps over the lazy dog.",
    ]
    
    print("Testing gTTS synthesizer...")
    for text in test_texts:
        audio = synthesize_tts(text)
        if audio is not None:
            print(f"  ✓ '{text[:30]}...' -> {len(audio)/16000:.2f}s")
        else:
            print(f"  ✗ '{text[:30]}...' failed")
