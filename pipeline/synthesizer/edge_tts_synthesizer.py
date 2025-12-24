"""
Edge-TTS Synthesizer - Pure Python, no DLL conflicts.
Uses Microsoft's neural TTS voices.
"""
import asyncio
import io
import numpy as np
import tempfile
import os
from typing import Optional, List

# Available voices for variety
EDGE_TTS_VOICES = [
    "en-US-JennyNeural",      # Female, American
    "en-US-GuyNeural",        # Male, American
    "en-US-AriaNeural",       # Female, American
    "en-US-DavisNeural",      # Male, American
    "en-GB-SoniaNeural",      # Female, British
    "en-GB-RyanNeural",       # Male, British
    "en-AU-NatashaNeural",    # Female, Australian
    "en-AU-WilliamNeural",    # Male, Australian
    "en-IN-NeerjaNeural",     # Female, Indian
    "en-IN-PrabhatNeural",    # Male, Indian
]


async def _synthesize_async(text: str, voice: str, sample_rate: int = 16000) -> Optional[np.ndarray]:
    """Async synthesis using edge-tts."""
    try:
        import edge_tts
        import soundfile as sf
        import re
        
        # Sanitize text - remove problematic characters
        text = re.sub(r'[^\w\s.,!?;:\'"()-]', '', text)  # Keep basic punctuation
        text = text.strip()
        
        if not text or len(text) < 2:
            print(f"[Edge-TTS] Text too short or empty after sanitization")
            return None
        
        # Create temp file for output
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
            temp_path = f.name
        
        try:
            # Generate speech
            communicate = edge_tts.Communicate(text, voice)
            await communicate.save(temp_path)
            
            # Load and convert to numpy
            audio, sr = sf.read(temp_path)
            
            # Resample if needed
            if sr != sample_rate:
                import librosa
                audio = librosa.resample(audio, orig_sr=sr, target_sr=sample_rate)
            
            # Convert to mono if stereo
            if len(audio.shape) > 1:
                audio = audio.mean(axis=1)
            
            return audio.astype(np.float32)
            
        finally:
            # Clean up temp file
            if os.path.exists(temp_path):
                os.remove(temp_path)
                
    except Exception as e:
        print(f"[Edge-TTS] Synthesis failed: {e}")
        return None


def synthesize_edge_tts(
    text: str, 
    voice: Optional[str] = None,
    sample_rate: int = 16000
) -> Optional[np.ndarray]:
    """
    Synthesize text to speech using Microsoft Edge TTS.
    
    Args:
        text: Text to synthesize
        voice: Voice name (random if None)
        sample_rate: Target sample rate
        
    Returns:
        Audio as numpy array, or None if failed
    """
    import random
    
    if voice is None:
        voice = random.choice(EDGE_TTS_VOICES)
    
    # Run async function
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        result = loop.run_until_complete(_synthesize_async(text, voice, sample_rate))
        loop.close()
        return result
    except Exception as e:
        print(f"[Edge-TTS] Error: {e}")
        return None


def get_available_voices() -> List[str]:
    """Return list of available voices."""
    return EDGE_TTS_VOICES.copy()


if __name__ == "__main__":
    # Test
    print("Testing Edge-TTS...")
    audio = synthesize_edge_tts("Hello, this is a test of Edge TTS synthesis.")
    if audio is not None:
        print(f"Success! Generated {len(audio)} samples ({len(audio)/16000:.2f}s)")
    else:
        print("Failed!")
