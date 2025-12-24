"""
Synthesizer Module

Text-to-Speech and Voice Conversion:
- TTS: Generate speech from text (gTTS, Coqui XTTS)
- VC: Transform speaker identity using RVC
"""

from .gtts_synthesizer import synthesize_tts
from .rvc_synthesizer import synthesize_vc

__all__ = ["synthesize_tts", "synthesize_vc"]
