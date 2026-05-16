"""
Room Impulse Response Simulation for EOT (Expectation over Transformation)

Simulates room acoustics to make adversarial attacks robust to physical playback.
Uses pyroomacoustics if available, otherwise provides a simple fallback.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple, List


class RoomSimulator(nn.Module):
    """
    Simulates room acoustics by convolving audio with room impulse responses.
    
    Used for Expectation over Transformation (EOT) to make adversarial
    examples robust to physical-world conditions.
    """
    
    def __init__(
        self,
        sample_rate: int = 16000,
        room_dims: Tuple[float, float, float] = (8.0, 6.0, 3.0),
        rt60_range: Tuple[float, float] = (0.2, 0.8),
        use_pyroomacoustics: bool = True
    ):
        """
        Args:
            sample_rate: Audio sample rate
            room_dims: Default room dimensions (width, length, height) in meters
            rt60_range: Range of RT60 (reverberation time) in seconds
            use_pyroomacoustics: Whether to try using pyroomacoustics
        """
        super().__init__()
        
        self.sample_rate = sample_rate
        self.room_dims = room_dims
        self.rt60_range = rt60_range
        
        # Check for pyroomacoustics
        self._use_pra = False
        if use_pyroomacoustics:
            try:
                import pyroomacoustics as pra
                self._pra = pra
                self._use_pra = True
                print("[RoomSimulator] Using pyroomacoustics for realistic RIR")
            except ImportError:
                print("[RoomSimulator] pyroomacoustics not available, using synthetic RIR")
    
    def generate_rir(
        self,
        room_dim: Optional[Tuple[float, float, float]] = None,
        rt60: Optional[float] = None,
        src_pos: Optional[Tuple[float, float, float]] = None,
        mic_pos: Optional[Tuple[float, float, float]] = None,
        length: int = 8000
    ) -> torch.Tensor:
        """
        Generate a room impulse response.
        
        Args:
            room_dim: Room dimensions (w, l, h) in meters
            rt60: Reverberation time T60 in seconds
            src_pos: Source position (x, y, z)
            mic_pos: Microphone position (x, y, z)
            length: RIR length in samples
            
        Returns:
            Room impulse response tensor (length,)
        """
        if room_dim is None:
            room_dim = self.room_dims
        if rt60 is None:
            rt60 = np.random.uniform(*self.rt60_range)
        
        if self._use_pra:
            return self._generate_rir_pra(room_dim, rt60, src_pos, mic_pos, length)
        else:
            return self._generate_rir_synthetic(rt60, length)
    
    def _generate_rir_pra(
        self,
        room_dim: Tuple[float, float, float],
        rt60: float,
        src_pos: Optional[Tuple[float, float, float]],
        mic_pos: Optional[Tuple[float, float, float]],
        length: int
    ) -> torch.Tensor:
        """Generate RIR using pyroomacoustics."""
        pra = self._pra
        
        # Random positions if not specified
        if src_pos is None:
            src_pos = [
                np.random.uniform(0.5, room_dim[0] - 0.5),
                np.random.uniform(0.5, room_dim[1] - 0.5),
                np.random.uniform(1.0, min(2.0, room_dim[2] - 0.5))
            ]
        if mic_pos is None:
            mic_pos = [
                np.random.uniform(0.5, room_dim[0] - 0.5),
                np.random.uniform(0.5, room_dim[1] - 0.5),
                np.random.uniform(1.0, min(2.0, room_dim[2] - 0.5))
            ]
        
        # Compute absorption coefficient from RT60
        e_absorption, max_order = pra.inverse_sabine(rt60, list(room_dim))
        
        # Create room
        room = pra.ShoeBox(
            list(room_dim),
            fs=self.sample_rate,
            materials=pra.Material(e_absorption),
            max_order=max_order
        )
        
        # Add source and microphone
        room.add_source(src_pos)
        room.add_microphone(mic_pos)
        
        # Compute RIR
        room.compute_rir()
        rir = room.rir[0][0]
        
        # Pad or truncate to desired length
        if len(rir) < length:
            rir = np.pad(rir, (0, length - len(rir)))
        else:
            rir = rir[:length]
        
        return torch.from_numpy(rir.astype(np.float32))
    
    def _generate_rir_synthetic(
        self,
        rt60: float,
        length: int
    ) -> torch.Tensor:
        """
        Generate synthetic RIR using exponential decay model.
        
        This is a simplified model when pyroomacoustics is not available.
        """
        t = torch.linspace(0, length / self.sample_rate, length)
        
        # Direct sound (delta at start)
        rir = torch.zeros(length)
        rir[0] = 1.0
        
        # Early reflections (random sparse impulses)
        num_early = int(self.sample_rate * 0.05)  # First 50ms
        early_times = torch.randint(1, num_early, (20,))
        early_gains = torch.rand(20) * 0.5
        for time, gain in zip(early_times, early_gains):
            if time < length:
                rir[time] += gain * (1 if torch.rand(1) > 0.5 else -1)
        
        # Late reverberation (exponentially decaying noise)
        decay_rate = 6.91 / rt60  # -60dB decay in rt60 seconds
        late_start = int(self.sample_rate * 0.05)
        late_length = length - late_start
        
        if late_length > 0:
            late_noise = torch.randn(late_length) * 0.1
            late_decay = torch.exp(-decay_rate * t[late_start:])
            rir[late_start:] += late_noise * late_decay
        
        # Normalize
        rir = rir / (torch.abs(rir).max() + 1e-8)
        
        return rir
    
    def apply_rir(
        self,
        audio: torch.Tensor,
        rir: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Apply room impulse response to audio via convolution.
        
        Args:
            audio: Audio waveform (batch, samples) or (samples,)
            rir: Room impulse response (optional, generates random if None)
            
        Returns:
            Reverberant audio of same shape as input
        """
        squeeze = False
        if audio.dim() == 1:
            audio = audio.unsqueeze(0)
            squeeze = True
        
        batch_size = audio.shape[0]
        
        if rir is None:
            rir = self.generate_rir()
        
        rir = rir.to(audio.device)
        
        # Ensure RIR has correct shape for conv1d
        if rir.dim() == 1:
            rir = rir.unsqueeze(0).unsqueeze(0)  # (1, 1, rir_length)
        
        # Pad audio for same-length output
        pad_length = rir.shape[-1] - 1
        audio_padded = F.pad(audio.unsqueeze(1), (pad_length, 0))  # (batch, 1, samples+pad)
        
        # Convolve
        reverberant = F.conv1d(audio_padded, rir)[:, 0, :]  # (batch, samples)
        
        # Normalize to prevent clipping
        reverberant = reverberant / (reverberant.abs().max(dim=-1, keepdim=True)[0] + 1e-8)
        reverberant = reverberant * audio.abs().max(dim=-1, keepdim=True)[0]
        
        if squeeze:
            reverberant = reverberant.squeeze(0)
        
        return reverberant
    
    def sample_transformation(self) -> callable:
        """
        Sample a random room transformation function.
        
        Returns:
            A function that applies a random room configuration to audio
        """
        # Sample random room parameters
        room_dim = (
            np.random.uniform(4, 12),
            np.random.uniform(4, 10),
            np.random.uniform(2.5, 4)
        )
        rt60 = np.random.uniform(*self.rt60_range)
        
        # Generate RIR for this configuration
        rir = self.generate_rir(room_dim, rt60)
        
        def transform(audio: torch.Tensor) -> torch.Tensor:
            return self.apply_rir(audio, rir)
        
        return transform
    
    def forward(
        self,
        audio: torch.Tensor,
        rir: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Apply room transformation."""
        return self.apply_rir(audio, rir)


class EOTTransformations(nn.Module):
    """
    Expectation over Transformation (EOT) for robust adversarial attacks.
    
    Combines multiple random transformations (room reverb, noise, etc.)
    to make adversarial examples robust to physical-world conditions.
    """
    
    def __init__(
        self,
        sample_rate: int = 16000,
        include_room: bool = True,
        include_noise: bool = True,
        include_gain: bool = True,
        noise_snr_range: Tuple[float, float] = (20, 40),
        gain_range: Tuple[float, float] = (0.8, 1.2)
    ):
        super().__init__()
        
        self.sample_rate = sample_rate
        self.include_room = include_room
        self.include_noise = include_noise
        self.include_gain = include_gain
        self.noise_snr_range = noise_snr_range
        self.gain_range = gain_range
        
        if include_room:
            self.room_sim = RoomSimulator(sample_rate)
    
    def sample_transformations(self, num_samples: int = 1) -> List[callable]:
        """
        Sample multiple random transformation functions.
        
        Args:
            num_samples: Number of transformations to sample
            
        Returns:
            List of transformation functions
        """
        transforms = []
        
        for _ in range(num_samples):
            def make_transform():
                # Room transformation
                room_t = None
                if self.include_room:
                    room_t = self.room_sim.sample_transformation()
                
                # Random gain
                gain = np.random.uniform(*self.gain_range) if self.include_gain else 1.0
                
                # Random noise SNR
                snr = np.random.uniform(*self.noise_snr_range) if self.include_noise else float('inf')
                
                def transform(audio: torch.Tensor) -> torch.Tensor:
                    x = audio
                    
                    # Apply gain
                    if gain != 1.0:
                        x = x * gain
                    
                    # Apply room
                    if room_t is not None:
                        x = room_t(x)
                    
                    # Add noise
                    if snr < float('inf'):
                        signal_power = (x ** 2).mean()
                        noise_power = signal_power / (10 ** (snr / 10))
                        noise = torch.randn_like(x) * torch.sqrt(noise_power)
                        x = x + noise
                    
                    return x
                
                return transform
            
            transforms.append(make_transform())
        
        return transforms
    
    def forward(
        self,
        audio: torch.Tensor,
        num_transforms: int = 1
    ) -> List[torch.Tensor]:
        """
        Apply multiple random transformations to audio.
        
        Args:
            audio: Input audio
            num_transforms: Number of transformed versions to return
            
        Returns:
            List of transformed audio tensors
        """
        transforms = self.sample_transformations(num_transforms)
        return [t(audio) for t in transforms]


if __name__ == "__main__":
    # Test room simulation
    import torch
    
    # Generate test audio (1 second)
    sr = 16000
    duration = 1.0
    t = torch.linspace(0, duration, int(sr * duration))
    audio = 0.5 * torch.sin(2 * torch.pi * 440 * t)
    
    # Room simulator
    room_sim = RoomSimulator(sample_rate=sr)
    
    # Generate and apply RIR
    rir = room_sim.generate_rir()
    reverberant = room_sim.apply_rir(audio)
    
    print(f"Audio shape: {audio.shape}")
    print(f"RIR shape: {rir.shape}")
    print(f"Reverberant shape: {reverberant.shape}")
    
    # Test EOT
    eot = EOTTransformations(sample_rate=sr)
    transformed = eot(audio, num_transforms=3)
    print(f"EOT produced {len(transformed)} transformations")
