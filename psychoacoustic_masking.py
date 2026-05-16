"""
Psychoacoustic Masking Threshold Computation

Computes frequency-domain masking thresholds based on human auditory perception.
Perturbations below this threshold are imperceptible to humans.

Based on MPEG-1 Audio psychoacoustic model and ISO 226 loudness contours.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple


def compute_masking_threshold(
    audio: torch.Tensor,
    sample_rate: int = 16000,
    n_fft: int = 2048,
    hop_length: int = 512,
    absolute_threshold_db: float = -60.0
) -> torch.Tensor:
    """
    Compute the psychoacoustic masking threshold for an audio signal.
    
    The masking threshold defines the level below which added noise
    is imperceptible to humans.
    
    Args:
        audio: Audio waveform tensor of shape (batch, samples) or (samples,)
        sample_rate: Sample rate in Hz
        n_fft: FFT size
        hop_length: STFT hop length
        absolute_threshold_db: Minimum absolute threshold in dB
        
    Returns:
        Masking threshold in dB, shape (batch, n_fft//2+1, num_frames)
    """
    if audio.dim() == 1:
        audio = audio.unsqueeze(0)
    
    # Compute STFT
    window = torch.hann_window(n_fft, device=audio.device)
    stft = torch.stft(
        audio,
        n_fft=n_fft,
        hop_length=hop_length,
        window=window,
        return_complex=True,
        pad_mode='reflect'
    )
    
    # Power spectral density (dB)
    psd = torch.abs(stft) ** 2
    psd_db = 10 * torch.log10(psd + 1e-10)
    
    # Compute masking threshold using simplified psychoacoustic model
    # Based on spreading function and absolute threshold of hearing
    masking = PsychoacousticMasking(
        n_fft=n_fft,
        sample_rate=sample_rate,
        device=audio.device
    )
    
    threshold = masking.compute_threshold(psd_db)
    
    # Apply absolute threshold floor
    threshold = torch.maximum(
        threshold, 
        torch.tensor(absolute_threshold_db, device=audio.device)
    )
    
    return threshold


class PsychoacousticMasking(nn.Module):
    """
    Psychoacoustic masking model based on MPEG-1 Audio Layer III.
    
    Computes the masking threshold from the power spectral density.
    Perturbations below this threshold are imperceptible.
    """
    
    def __init__(
        self,
        n_fft: int = 2048,
        sample_rate: int = 16000,
        bark_bands: int = 25,
        device: str = "cpu"
    ):
        super().__init__()
        
        self.n_fft = n_fft
        self.n_bins = n_fft // 2 + 1
        self.sample_rate = sample_rate
        self.bark_bands = bark_bands
        
        # Precompute frequency to Bark scale mapping
        freqs = torch.linspace(0, sample_rate / 2, self.n_bins)
        bark = self._hz_to_bark(freqs)
        self.register_buffer('freqs', freqs.to(device))
        self.register_buffer('bark', bark.to(device))
        
        # Absolute threshold of hearing (ATH) - ISO 226
        ath = self._absolute_threshold_of_hearing(freqs)
        self.register_buffer('ath', ath.to(device))
        
        # Spreading function (simplified)
        spread = self._create_spreading_function(bark_bands)
        self.register_buffer('spreading', spread.to(device))
    
    def _hz_to_bark(self, freq: torch.Tensor) -> torch.Tensor:
        """Convert frequency (Hz) to Bark scale."""
        return 13 * torch.atan(0.76 * freq / 1000) + 3.5 * torch.atan((freq / 7500) ** 2)
    
    def _absolute_threshold_of_hearing(self, freq: torch.Tensor) -> torch.Tensor:
        """
        Compute absolute threshold of hearing in dB SPL.
        Based on ISO 226 equal-loudness contours.
        """
        # Simplified ATH formula
        f_khz = freq / 1000.0 + 1e-10
        ath = (
            3.64 * torch.pow(f_khz, -0.8)
            - 6.5 * torch.exp(-0.6 * (f_khz - 3.3) ** 2)
            + 1e-3 * torch.pow(f_khz, 4)
        )
        # Clamp to reasonable range
        return torch.clamp(ath, -20, 80)
    
    def _create_spreading_function(self, num_bands: int) -> torch.Tensor:
        """
        Create spreading function matrix.
        
        Models how energy in one critical band masks neighboring bands.
        """
        # Simplified triangular spreading function
        spread = torch.zeros(num_bands, num_bands)
        
        for i in range(num_bands):
            for j in range(num_bands):
                diff = j - i
                if diff == 0:
                    spread[i, j] = 0.0  # dB
                elif diff > 0:
                    # Upper slope (masking higher frequencies)
                    spread[i, j] = -25 * diff
                else:
                    # Lower slope (masking lower frequencies)
                    spread[i, j] = -10 * abs(diff)
        
        return spread
    
    def compute_threshold(self, psd_db: torch.Tensor) -> torch.Tensor:
        """
        Compute masking threshold from power spectral density.
        
        Args:
            psd_db: Power spectral density in dB, shape (batch, n_bins, frames)
            
        Returns:
            Masking threshold in dB, same shape as input
        """
        batch, n_bins, n_frames = psd_db.shape
        
        # Find tonal and noise maskers (simplified: just use peak detection)
        # Group into critical bands (simplified: use max in each band)
        bark_edges = torch.linspace(0, self.bark[-1], self.bark_bands + 1, device=psd_db.device)
        
        band_power = torch.zeros(batch, self.bark_bands, n_frames, device=psd_db.device)
        
        for b in range(self.bark_bands):
            mask = (self.bark >= bark_edges[b]) & (self.bark < bark_edges[b + 1])
            if mask.any():
                band_power[:, b, :] = psd_db[:, mask, :].max(dim=1)[0]
        
        # Apply spreading function to get global masking threshold per band
        # spread: (bands, bands), band_power: (batch, bands, frames)
        # Result: (batch, bands, frames)
        spread_db = self.spreading.unsqueeze(0).unsqueeze(-1)  # (1, bands, bands, 1)
        band_power_exp = band_power.unsqueeze(2)  # (batch, bands, 1, frames)
        
        # Masking contribution from each band to every other band
        masked = band_power_exp + spread_db  # (batch, bands, bands, frames)
        
        # Sum contributions (in power domain, then back to dB)
        masked_power = torch.pow(10, masked / 10)
        total_masked = masked_power.sum(dim=2)  # (batch, bands, frames)
        threshold_db = 10 * torch.log10(total_masked + 1e-10)
        
        # Interpolate back to FFT bins
        # Map each FFT bin to its corresponding band
        bin_to_band = torch.zeros(n_bins, device=psd_db.device, dtype=torch.long)
        for i, b in enumerate(self.bark):
            band_idx = torch.searchsorted(bark_edges[1:], b)
            bin_to_band[i] = min(band_idx, self.bark_bands - 1)
        
        # Expand threshold to full frequency resolution
        threshold_full = torch.zeros(batch, n_bins, n_frames, device=psd_db.device)
        for i in range(n_bins):
            threshold_full[:, i, :] = threshold_db[:, bin_to_band[i], :]
        
        # Combine with absolute threshold of hearing
        ath_expanded = self.ath.view(1, -1, 1).expand(batch, -1, n_frames)
        threshold_full = torch.maximum(threshold_full, ath_expanded)
        
        return threshold_full
    
    def forward(self, audio: torch.Tensor, n_fft: Optional[int] = None) -> torch.Tensor:
        """
        Compute masking threshold for audio.
        
        Args:
            audio: Waveform tensor (batch, samples) or (samples,)
            n_fft: Optional FFT size override
            
        Returns:
            Masking threshold in dB
        """
        return compute_masking_threshold(
            audio,
            sample_rate=self.sample_rate,
            n_fft=n_fft or self.n_fft
        )


class ImperceptibilityLoss(nn.Module):
    """
    Loss function for imperceptible adversarial perturbations.
    
    Penalizes perturbations that exceed the psychoacoustic masking threshold.
    This is ℓ_θ(x, δ) from the paper.
    """
    
    def __init__(
        self,
        sample_rate: int = 16000,
        n_fft: int = 2048,
        hop_length: int = 512
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
    
    def forward(
        self,
        clean_audio: torch.Tensor,
        perturbation: torch.Tensor,
        masking_threshold: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute imperceptibility loss.
        
        Args:
            clean_audio: Original audio (batch, samples)
            perturbation: Adversarial perturbation δ (batch, samples)
            masking_threshold: Precomputed threshold (optional)
            
        Returns:
            Scalar loss value
        """
        # Compute masking threshold of clean audio if not provided
        if masking_threshold is None:
            masking_threshold = compute_masking_threshold(
                clean_audio,
                self.sample_rate,
                self.n_fft,
                self.hop_length
            )
        
        # Compute power spectral density of perturbation
        window = torch.hann_window(self.n_fft, device=perturbation.device)
        
        if perturbation.dim() == 1:
            perturbation = perturbation.unsqueeze(0)
            
        stft = torch.stft(
            perturbation,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            window=window,
            return_complex=True,
            pad_mode='reflect'
        )
        
        psd_delta = torch.abs(stft) ** 2
        psd_delta_db = 10 * torch.log10(psd_delta + 1e-10)
        
        # Hinge loss: max(psd_delta - threshold, 0)
        # Only penalize when perturbation exceeds the masking threshold
        exceed = psd_delta_db - masking_threshold
        loss = F.relu(exceed).mean()
        
        return loss


if __name__ == "__main__":
    # Test psychoacoustic masking
    import torch
    
    # Generate test audio (1 second)
    sr = 16000
    duration = 1.0
    t = torch.linspace(0, duration, int(sr * duration))
    
    # Mix of frequencies
    audio = (
        0.5 * torch.sin(2 * torch.pi * 440 * t) +  # A4
        0.3 * torch.sin(2 * torch.pi * 880 * t) +  # A5
        0.2 * torch.sin(2 * torch.pi * 1760 * t)   # A6
    )
    
    # Compute masking threshold
    threshold = compute_masking_threshold(audio, sample_rate=sr)
    
    print(f"Audio shape: {audio.shape}")
    print(f"Threshold shape: {threshold.shape}")
    print(f"Threshold range: [{threshold.min():.1f}, {threshold.max():.1f}] dB")
    
    # Test imperceptibility loss
    perturbation = 0.01 * torch.randn_like(audio)
    loss_fn = ImperceptibilityLoss(sample_rate=sr)
    loss = loss_fn(audio.unsqueeze(0), perturbation.unsqueeze(0))
    print(f"Imperceptibility loss: {loss.item():.4f}")
