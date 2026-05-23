"""
Audio Preprocessor Module

Handles: resampling, RMS normalization, silence trimming.
All operations preserve audio quality while standardizing format.
"""

import numpy as np
import librosa
from typing import Tuple, Optional
from pipeline.config import Config


def preprocess_audio(
    audio: np.ndarray,
    sr: int,
    config: Config
) -> Tuple[Optional[np.ndarray], dict]:
    """
    Full preprocessing pipeline: resample, normalize, trim.
    
    Args:
        audio: Raw audio array
        sr: Current sample rate
        config: Pipeline configuration
        
    Returns:
        Tuple of (processed_audio, metadata_dict)
        Returns (None, metadata) if audio should be skipped
    """
    metadata = {"preprocessing": {}}
    
    # Step 1: Resample if needed
    if sr != config.audio.target_sr:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=config.audio.target_sr)
        metadata["preprocessing"]["resampled"] = True
    
    # Step 2: Convert to mono if stereo
    if len(audio.shape) > 1:
        audio = np.mean(audio, axis=0)
        metadata["preprocessing"]["converted_mono"] = True
    
    # Step 3: Check duration (skip if too long or too short)
    duration = len(audio) / config.audio.target_sr
    if duration < config.audio.min_duration:
        metadata["skip_reason"] = f"too_short ({duration:.2f}s < {config.audio.min_duration}s)"
        return None, metadata
    if duration > config.audio.max_duration:
        metadata["skip_reason"] = f"too_long ({duration:.2f}s > {config.audio.max_duration}s)"
        return None, metadata
    
    # Step 4: Trim silence
    audio, trim_idx = trim_silence(audio, config.audio.silence_threshold_db)
    metadata["preprocessing"]["trimmed"] = True
    metadata["preprocessing"]["trim_samples"] = int(trim_idx[0])
    
    # Check duration after trimming
    duration_after = len(audio) / config.audio.target_sr
    if duration_after < config.audio.min_duration:
        metadata["skip_reason"] = f"too_short_after_trim ({duration_after:.2f}s)"
        return None, metadata
    
    # Step 5: RMS Normalize
    audio = rms_normalize(audio, config.audio.normalize_db)
    metadata["preprocessing"]["normalized"] = True
    metadata["preprocessing"]["target_db"] = config.audio.normalize_db
    
    # Record final duration
    metadata["duration"] = duration_after
    
    return audio, metadata


def resample(audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    """Resample audio to target sample rate."""
    if orig_sr == target_sr:
        return audio
    return librosa.resample(audio, orig_sr=orig_sr, target_sr=target_sr)


def rms_normalize(audio: np.ndarray, target_db: float = -20.0) -> np.ndarray:
    """
    Normalize audio to target RMS level in dBFS.
    
    Args:
        audio: Input audio array
        target_db: Target RMS level in dBFS (e.g., -20)
        
    Returns:
        Normalized audio array clipped to [-1, 1]
    """
    # Calculate current RMS
    rms = np.sqrt(np.mean(audio ** 2))
    
    # Avoid log(0)
    if rms < 1e-10:
        return audio
    
    # Calculate current dB
    current_db = 20 * np.log10(rms)
    
    # Calculate gain needed
    gain_db = target_db - current_db
    gain_linear = 10 ** (gain_db / 20)
    
    # Apply gain and clip
    normalized = audio * gain_linear
    return np.clip(normalized, -1.0, 1.0)


def trim_silence(
    audio: np.ndarray,
    threshold_db: float = 40
) -> Tuple[np.ndarray, Tuple[int, int]]:
    """
    Trim leading and trailing silence.
    
    Args:
        audio: Input audio array
        threshold_db: Threshold below reference to consider silence
        
    Returns:
        Tuple of (trimmed_audio, (start_idx, end_idx))
    """
    trimmed, index = librosa.effects.trim(audio, top_db=threshold_db)
    return trimmed, index


def compute_snr(audio: np.ndarray, sr: int) -> float:
    """
    Estimate Signal-to-Noise Ratio.
    
    Simple estimation based on signal vs noise floor.
    """
    # Use librosa's built-in RMS
    rms = librosa.feature.rms(y=audio)[0]
    
    # Estimate noise as bottom 10% of RMS values
    noise_floor = np.percentile(rms, 10)
    signal_level = np.percentile(rms, 90)
    
    if noise_floor < 1e-10:
        return 60.0  # Very clean signal
    
    snr_db = 20 * np.log10(signal_level / noise_floor)
    return snr_db


def detect_clipping(audio: np.ndarray, threshold: float = 0.99) -> bool:
    """Check if audio has clipping artifacts."""
    return np.any(np.abs(audio) > threshold)


if __name__ == "__main__":
    # Test preprocessing
    import soundfile as sf
    
    # Generate test tone
    sr = 16000
    t = np.linspace(0, 3, 3 * sr)
    test_audio = 0.1 * np.sin(2 * np.pi * 440 * t)  # Quiet 440Hz tone
    
    # Add silence
    test_audio = np.concatenate([np.zeros(sr), test_audio, np.zeros(sr)])
    
    from pipeline.config import load_config
    cfg = load_config("../config.yaml")
    
    processed, meta = preprocess_audio(test_audio, sr, cfg)
    
    if processed is not None:
        print(f"Original duration: {len(test_audio)/sr:.2f}s")
        print(f"Processed duration: {meta['duration']:.2f}s")
        print(f"Metadata: {meta}")
    else:
        print(f"Skipped: {meta['skip_reason']}")
