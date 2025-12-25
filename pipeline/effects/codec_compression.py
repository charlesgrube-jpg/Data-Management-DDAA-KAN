"""
Codec Compression Effects

Simulates lossy codec compression (MP3, Opus, AAC) that occurs during
digital transmission (messaging apps, social media, VoIP).

Requires:
- pydub for audio encoding/decoding
- ffmpeg installed on system

Usage:
    from pipeline.effects.codec_compression import apply_codec_compression
    compressed = apply_codec_compression(audio, sr=16000, codec='mp3', bitrate='64k')
"""

import numpy as np
from typing import Optional
import random


# Global variable to cache ffmpeg availability
_FFMPEG_AVAILABLE = None

def is_ffmpeg_available() -> bool:
    """Check if ffmpeg is available on the system."""
    global _FFMPEG_AVAILABLE
    if _FFMPEG_AVAILABLE is not None:
        return _FFMPEG_AVAILABLE

    try:
        import subprocess
        result = subprocess.run(
            ['ffmpeg', '-version'],
            capture_output=True,
            timeout=5
        )
        _FFMPEG_AVAILABLE = (result.returncode == 0)
    except (FileNotFoundError, subprocess.TimeoutExpired):
        _FFMPEG_AVAILABLE = False
    
    return _FFMPEG_AVAILABLE


def apply_codec_compression(
    audio: np.ndarray,
    sr: int,
    codec: str = "mp3",
    bitrate: str = "64k"
) -> Optional[np.ndarray]:
    """
    Apply lossy codec compression to audio.
    
    Args:
        audio: Input audio array
        sr: Sample rate
        codec: Codec type ("mp3", "opus", "aac")
        bitrate: Target bitrate (e.g., "64k", "128k")
        
    Returns:
        Compressed audio array, or None if failed
    """
    # Check dependencies
    try:
        from pydub import AudioSegment
        import soundfile as sf
    except ImportError as e:
        print(f"[Codec] Missing dependency: {e}")
        print("[Codec] Install with: pip install pydub soundfile")
        return None
    
    if not is_ffmpeg_available():
        print("[Codec] ffmpeg not found. Skipping codec compression.")
        print("[Codec] Install ffmpeg: conda install -c conda-forge ffmpeg")
        return None
    
    try:
        import tempfile
        
        # Convert numpy to temp WAV file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f_in:
            temp_in = f_in.name
        
        sf.write(temp_in, audio, sr)
        audio_segment = AudioSegment.from_wav(temp_in)
        
        # Compress with codec
        codec_format = codec.lower()
        if codec_format == "opus":
            # Opus requires special handling
            codec_format = "opus"
            codec_ext = "opus"
        elif codec_format == "aac":
            codec_format = "adts"  # AAC format for pydub
            codec_ext = "aac"
        else:
            codec_ext = codec_format
        
        with tempfile.NamedTemporaryFile(suffix=f".{codec_ext}", delete=False) as f_compressed:
            temp_compressed = f_compressed.name
        
        # Export with compression
        audio_segment.export(
            temp_compressed,
            format=codec_format,
            bitrate=bitrate,
            codec=codec if codec != "opus" else "libopus"
        )
        
        # Load compressed audio back
        compressed_segment = AudioSegment.from_file(temp_compressed)
        
        # Convert back to numpy array
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f_out:
            temp_out = f_out.name
        
        compressed_segment.export(temp_out, format="wav")
        
        # Load as numpy array
        import librosa
        result, _ = librosa.load(temp_out, sr=sr)
        
        # Cleanup temp files
        import os
        for temp_file in [temp_in, temp_compressed, temp_out]:
            try:
                os.unlink(temp_file)
            except:
                pass
        
        return result
        
    except Exception as e:
        print(f"[Codec] Compression failed: {e}")
        return None


def select_codec_tier(config) -> str:
    """
    Select a codec tier based on distribution in config.
    
    Returns:
        Codec tier string (e.g., "mp3_64k", "none")
    """
    if not hasattr(config, 'codec_compression') or not config.codec_compression.enabled:
        return "none"
    
    distribution = config.codec_compression.codec_distribution
    
    # Convert distribution to choices
    choices = []
    weights = []
    for tier, weight in distribution.items():
        choices.append(tier)
        weights.append(weight)
    
    # Normalize weights
    total = sum(weights)
    weights = [w / total for w in weights]
    
    return random.choices(choices, weights=weights)[0]


def parse_codec_tier(tier: str) -> tuple:
    """
    Parse codec tier string into (codec, bitrate).
    
    Args:
        tier: e.g., "mp3_64k", "opus_48k", "none"
        
    Returns:
        (codec, bitrate) tuple, or (None, None) if "none"
    """
    if tier == "none":
        return None, None
    
    parts = tier.split("_")
    if len(parts) != 2:
        return None, None
    
    codec = parts[0]
    bitrate = parts[1]
    
    return codec, bitrate


def apply_codec_if_enabled(
    audio: np.ndarray,
    sr: int,
    config,
    apply_to_synthetic_only: bool = False,
    is_synthetic: bool = False
) -> tuple:
    """
    Apply codec compression if enabled in config.
    
    Args:
        audio: Input audio
        sr: Sample rate
        config: Pipeline config
        apply_to_synthetic_only: If True, only apply to synthetic samples
        is_synthetic: Whether this is a synthetic sample
        
    Returns:
        (processed_audio, codec_tier) tuple
    """
    # Check if codec is enabled
    if not hasattr(config, 'codec_compression') or not config.codec_compression.enabled:
        return audio, "none"
    
    # Check apply_to setting
    apply_to = config.codec_compression.apply_to
    if apply_to == "none":
        return audio, "none"
    elif apply_to == "synthetic_only" and not is_synthetic:
        return audio, "none"
    
    # Select codec tier
    tier = select_codec_tier(config)
    
    if tier == "none":
        return audio, "none"
    
    # Parse tier
    codec, bitrate = parse_codec_tier(tier)
    
    if codec is None:
        return audio, "none"
    
    # Apply compression
    compressed = apply_codec_compression(audio, sr, codec, bitrate)
    
    if compressed is None:
        # Fallback to original if compression failed
        print(f"[Codec] Compression failed, using original audio")
        return audio, "none"
    
    return compressed, tier


if __name__ == "__main__":
    # Test codec compression
    print("Testing codec compression module...")
    
    # Check ffmpeg
    if is_ffmpeg_available():
        print("✓ ffmpeg is available")
    else:
        print("✗ ffmpeg not found (codec compression will be skipped)")
    
    # Generate test audio
    sr = 16000
    duration = 3
    test_audio = np.random.randn(duration * sr) * 0.1
    
    # Test different codecs
    for codec, bitrate in [("mp3", "64k"), ("opus", "48k"), ("aac", "96k")]:
        print(f"\nTesting {codec} at {bitrate}...")
        result = apply_codec_compression(test_audio, sr, codec, bitrate)
        if result is not None:
            print(f"  ✓ {codec} compression successful: {len(result)/sr:.2f}s")
        else:
            print(f"  ✗ {codec} compression failed")
