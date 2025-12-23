"""
Audio Synthesizer Module

Generates synthetic versions using:
- TTS (Text-to-Speech): transcript -> audio
- VC (Voice Conversion): audio -> audio (different voice)

Wraps Coqui TTS and RVC/FreeVC for synthesis.
"""

import numpy as np
from typing import Dict, Any, Optional, List, Tuple
import random
from abc import ABC, abstractmethod
from pipeline.config import Config


class BaseSynthesizer(ABC):
    """Abstract base class for audio synthesizers."""
    
    @abstractmethod
    def synthesize(self, *args, **kwargs) -> np.ndarray:
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        pass


class TTSSynthesizer(BaseSynthesizer):
    """Text-to-Speech synthesis using Coqui TTS."""
    
    def __init__(self, model_name: str, device: str = "cpu"):
        self.model_name = model_name
        self.device = device
        self._tts = None
        self._available = None
    
    def _load_model(self):
        """Lazy load TTS model."""
        if self._tts is None:
            try:
                from TTS.api import TTS
                self._tts = TTS(model_name=self.model_name).to(self.device)
                self._available = True
                print(f"[TTS] Loaded model: {self.model_name}")
            except Exception as e:
                print(f"[TTS] Failed to load {self.model_name}: {e}")
                self._available = False
    
    def is_available(self) -> bool:
        if self._available is None:
            self._load_model()
        return self._available
    
    def synthesize(
        self,
        text: str,
        speaker: Optional[str] = None,
        language: Optional[str] = None
    ) -> Optional[np.ndarray]:
        """
        Generate speech from text.
        
        Args:
            text: Input transcript
            speaker: Speaker ID (for multi-speaker models)
            language: Language code (for multilingual models)
            
        Returns:
            Synthesized audio array or None if failed
        """
        self._load_model()
        
        if not self._available:
            return None
        
        try:
            # Coqui TTS returns audio as numpy array
            audio = self._tts.tts(
                text=text,
                speaker=speaker,
                language=language
            )
            return np.array(audio)
        except Exception as e:
            print(f"[TTS] Synthesis failed: {e}")
            return None


class VCSynthesizer(BaseSynthesizer):
    """Voice Conversion synthesis (placeholder for RVC/FreeVC)."""
    
    def __init__(self, model_name: str, device: str = "cpu"):
        self.model_name = model_name
        self.device = device
        self._model = None
        self._available = None
    
    def _load_model(self):
        """Lazy load VC model."""
        if self._model is None:
            # TODO: Implement actual VC model loading
            # This is a placeholder - real implementation depends on chosen VC library
            print(f"[VC] WARNING: VC model {self.model_name} not implemented yet")
            print(f"[VC] Using pitch-shift placeholder for development")
            self._available = True  # Use placeholder
    
    def is_available(self) -> bool:
        if self._available is None:
            self._load_model()
        return self._available
    
    def synthesize(
        self,
        audio: np.ndarray,
        sr: int,
        target_voice: Optional[str] = None
    ) -> Optional[np.ndarray]:
        """
        Convert voice in audio to target voice.
        
        Args:
            audio: Input audio array
            sr: Sample rate
            target_voice: Target voice/speaker ID
            
        Returns:
            Converted audio array or None if failed
        """
        self._load_model()
        
        if not self._available:
            return None
        
        try:
            # PLACEHOLDER: Simple pitch shift as stand-in for real VC
            # Replace with actual RVC/FreeVC implementation
            import librosa
            
            # Random pitch shift to simulate voice conversion
            n_steps = random.choice([-3, -2, 2, 3])  # Semitones
            converted = librosa.effects.pitch_shift(audio, sr=sr, n_steps=n_steps)
            
            return converted
        except Exception as e:
            print(f"[VC] Conversion failed: {e}")
            return None


class SynthesizerManager:
    """Manages multiple synthesizers and handles synthesis strategy."""
    
    def __init__(self, config: Config):
        self.config = config
        self.tts_synthesizers: List[TTSSynthesizer] = []
        self.vc_synthesizers: List[VCSynthesizer] = []
        self._initialized = False
    
    def initialize(self, device: str = "cpu"):
        """Initialize all configured synthesizers."""
        if self._initialized:
            return
        
        # Load TTS models
        for model_name in self.config.synthesis.tts_models:
            synth = TTSSynthesizer(model_name, device)
            if synth.is_available():
                self.tts_synthesizers.append(synth)
        
        # Load VC models
        for model_name in self.config.synthesis.vc_models:
            synth = VCSynthesizer(model_name, device)
            if synth.is_available():
                self.vc_synthesizers.append(synth)
        
        print(f"[Synthesizer] Initialized {len(self.tts_synthesizers)} TTS, "
              f"{len(self.vc_synthesizers)} VC models")
        
        self._initialized = True
    
    def synthesize(
        self,
        audio: np.ndarray,
        sr: int,
        transcript: str
    ) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
        """
        Generate synthetic versions based on config strategy.
        
        Args:
            audio: Source audio
            sr: Sample rate
            transcript: Text transcript
            
        Returns:
            Tuple of (tts_result, vc_result) - each is dict with audio + metadata
        """
        self.initialize()
        
        strategy = self.config.synthesis.pick_strategy
        tts_result = None
        vc_result = None
        
        # TTS synthesis
        if strategy in ["random", "both", "tts_only"] and self.tts_synthesizers:
            synth = random.choice(self.tts_synthesizers)
            tts_audio = synth.synthesize(transcript)
            
            if tts_audio is not None:
                tts_result = {
                    "audio": tts_audio,
                    "generator": synth.model_name,
                    "method": "tts"
                }
        
        # VC synthesis
        if strategy in ["random", "both", "vc_only"] and self.vc_synthesizers:
            synth = random.choice(self.vc_synthesizers)
            vc_audio = synth.synthesize(audio, sr)
            
            if vc_audio is not None:
                vc_result = {
                    "audio": vc_audio,
                    "generator": synth.model_name,
                    "method": "vc"
                }
        
        return tts_result, vc_result
    
    def pick_synthetic(
        self,
        tts_result: Optional[Dict],
        vc_result: Optional[Dict]
    ) -> Optional[Dict[str, Any]]:
        """
        Pick one synthetic result based on strategy.
        
        For 'random' strategy, randomly chooses between available results.
        For 'both', caller should handle both separately.
        """
        strategy = self.config.synthesis.pick_strategy
        
        if strategy == "both":
            # Return TTS, caller should also use VC separately
            return tts_result or vc_result
        
        if strategy == "tts_only":
            return tts_result
        
        if strategy == "vc_only":
            return vc_result
        
        # Random strategy
        available = [r for r in [tts_result, vc_result] if r is not None]
        if not available:
            return None
        return random.choice(available)


# Module-level singleton for easy access
_manager: Optional[SynthesizerManager] = None


def get_synthesizer_manager(config: Config) -> SynthesizerManager:
    """Get or create the synthesizer manager singleton."""
    global _manager
    if _manager is None:
        _manager = SynthesizerManager(config)
    return _manager


def synthesize_audio(
    audio: np.ndarray,
    sr: int,
    transcript: str,
    config: Config
) -> Optional[Dict[str, Any]]:
    """
    Convenience function to synthesize audio.
    
    Returns:
        Dict with synthetic audio and metadata, or None if failed
    """
    manager = get_synthesizer_manager(config)
    tts_result, vc_result = manager.synthesize(audio, sr, transcript)
    return manager.pick_synthetic(tts_result, vc_result)


if __name__ == "__main__":
    from pipeline.config import load_config
    
    cfg = load_config("../config.yaml")
    
    # Test synthesis
    sr = 16000
    test_audio = np.random.randn(3 * sr) * 0.1
    test_transcript = "Hello, this is a test sentence."
    
    result = synthesize_audio(test_audio, sr, test_transcript, cfg)
    
    if result:
        print(f"Generated synthetic audio:")
        print(f"  Method: {result['method']}")
        print(f"  Generator: {result['generator']}")
        print(f"  Duration: {len(result['audio'])/sr:.2f}s")
    else:
        print("Synthesis failed (models may not be installed)")
