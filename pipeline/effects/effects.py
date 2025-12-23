"""
Audio Effects Module

Applies channel degradation effects to simulate real-world conditions.
Same effects applied to both real and synthetic audio for fairness.

Quality tiers:
- clean: No degradation
- mobile: Compression + mild background noise
- noisy: Heavy noise + reverb
"""

import numpy as np
import random
from typing import Tuple
from pipeline.config import Config


def apply_effects(
    audio: np.ndarray,
    sr: int,
    quality_tier: str
) -> np.ndarray:
    """
    Apply quality degradation based on tier.
    
    Args:
        audio: Input audio array
        sr: Sample rate
        quality_tier: "clean", "mobile", or "noisy"
        
    Returns:
        Processed audio array
    """
    if quality_tier == "clean":
        return audio
    
    if quality_tier == "mobile":
        # Mild degradation typical of mobile recordings
        audio = add_lowpass_filter(audio, sr, cutoff=7000)
        audio = add_noise(audio, snr_db=30)
        audio = add_compression_artifacts(audio, quality=0.8)
        return audio
    
    if quality_tier == "noisy":
        # Heavy degradation
        audio = add_lowpass_filter(audio, sr, cutoff=5000)
        audio = add_noise(audio, snr_db=15)
        audio = add_reverb(audio, sr, room_size=0.5)
        return audio
    
    return audio


def select_quality_tier(config: Config) -> str:
    """
    Randomly select quality tier based on configured distribution.
    
    Returns:
        Quality tier string
    """
    distribution = config.effects.quality_distribution
    tiers = list(distribution.keys())
    weights = list(distribution.values())
    
    return random.choices(tiers, weights=weights, k=1)[0]


def add_noise(audio: np.ndarray, snr_db: float = 20) -> np.ndarray:
    """
    Add white noise at specified SNR level.
    
    Args:
        audio: Input audio
        snr_db: Signal-to-noise ratio in dB
        
    Returns:
        Noisy audio
    """
    # Calculate signal power
    signal_power = np.mean(audio ** 2)
    
    # Calculate noise power for target SNR
    noise_power = signal_power / (10 ** (snr_db / 10))
    
    # Generate noise
    noise = np.random.randn(len(audio)) * np.sqrt(noise_power)
    
    return audio + noise


def add_reverb(
    audio: np.ndarray,
    sr: int,
    room_size: float = 0.3,
    damping: float = 0.5
) -> np.ndarray:
    """
    Add simple reverb effect using convolution.
    
    Args:
        audio: Input audio
        sr: Sample rate
        room_size: Room size parameter (0-1)
        damping: High frequency damping (0-1)
        
    Returns:
        Reverberant audio
    """
    # Simple impulse response simulation
    decay_samples = int(room_size * sr)
    ir_length = min(decay_samples, sr // 2)  # Max 0.5 seconds
    
    # Generate exponentially decaying impulse response
    t = np.arange(ir_length)
    decay = np.exp(-t / (ir_length / 3))
    
    # Add some randomness for realism
    ir = np.random.randn(ir_length) * decay
    ir[0] = 1.0  # Direct sound
    
    # Apply damping (reduce high frequencies)
    if damping > 0:
        from scipy.ndimage import gaussian_filter1d
        ir = gaussian_filter1d(ir, sigma=damping * 5)
    
    # Normalize IR
    ir = ir / np.max(np.abs(ir))
    
    # Convolve with audio
    reverbed = np.convolve(audio, ir, mode='same')
    
    # Mix with original (80% dry, 20% wet)
    mix = 0.8 * audio + 0.2 * reverbed
    
    return np.clip(mix, -1.0, 1.0)


def add_lowpass_filter(
    audio: np.ndarray,
    sr: int,
    cutoff: int = 7000
) -> np.ndarray:
    """
    Apply lowpass filter to simulate telephone/mobile quality.
    
    Args:
        audio: Input audio
        sr: Sample rate
        cutoff: Cutoff frequency in Hz
        
    Returns:
        Filtered audio
    """
    try:
        from scipy.signal import butter, filtfilt
        
        nyquist = sr / 2
        normalized_cutoff = cutoff / nyquist
        
        # 4th order Butterworth filter
        b, a = butter(4, normalized_cutoff, btype='low')
        filtered = filtfilt(b, a, audio)
        
        return filtered
    except ImportError:
        # Fallback: simple moving average
        window_size = max(1, sr // cutoff)
        return np.convolve(audio, np.ones(window_size)/window_size, mode='same')


def add_compression_artifacts(
    audio: np.ndarray,
    quality: float = 0.8
) -> np.ndarray:
    """
    Simulate lossy compression artifacts.
    
    Args:
        audio: Input audio
        quality: Quality factor (0-1, higher = less degradation)
        
    Returns:
        Audio with compression-like artifacts
    """
    # Simple quantization to simulate compression
    levels = int(256 * quality)
    levels = max(16, levels)  # Minimum 16 levels
    
    # Quantize
    quantized = np.round(audio * (levels / 2)) / (levels / 2)
    
    return np.clip(quantized, -1.0, 1.0)


def apply_effects_to_pair(
    real_audio: np.ndarray,
    synthetic_audio: np.ndarray,
    sr: int,
    config: Config
) -> Tuple[np.ndarray, np.ndarray, str]:
    """
    Apply same effects to both real and synthetic audio.
    
    Critical for fairness: both samples get identical degradation.
    
    Args:
        real_audio: Real audio array
        synthetic_audio: Synthetic audio array
        sr: Sample rate
        config: Pipeline configuration
        
    Returns:
        Tuple of (processed_real, processed_synthetic, quality_tier)
    """
    if not config.effects.enabled:
        return real_audio, synthetic_audio, "clean"
    
    # Select quality tier
    tier = select_quality_tier(config)
    
    if config.effects.apply_same_to_pair:
        # Apply exact same effects (important for fairness)
        # Use same random seed for both
        state = np.random.get_state()
        
        real_processed = apply_effects(real_audio, sr, tier)
        
        np.random.set_state(state)  # Reset to same state
        synthetic_processed = apply_effects(synthetic_audio, sr, tier)
    else:
        # Independent effects (not recommended)
        real_processed = apply_effects(real_audio, sr, tier)
        synthetic_processed = apply_effects(synthetic_audio, sr, tier)
    
    return real_processed, synthetic_processed, tier


if __name__ == "__main__":
    from pipeline.config import load_config
    
    cfg = load_config("../config.yaml")
    
    # Test effects
    sr = 16000
    t = np.linspace(0, 3, 3 * sr)
    test_audio = 0.5 * np.sin(2 * np.pi * 440 * t)  # 440Hz tone
    
    for tier in ["clean", "mobile", "noisy"]:
        processed = apply_effects(test_audio, sr, tier)
        print(f"{tier}: max={np.max(np.abs(processed)):.3f}, "
              f"rms={np.sqrt(np.mean(processed**2)):.3f}")
