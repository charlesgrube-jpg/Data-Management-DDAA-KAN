"""
Synthesizer Module

Text-to-Speech and Voice Conversion:
- TTS: Generate speech from text (gTTS, Coqui XTTS)
- VC: Transform speaker identity (placeholder for now)
"""

from .gtts_synthesizer import synthesize_tts

__all__ = ["synthesize_tts"]
