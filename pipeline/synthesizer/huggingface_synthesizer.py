
import os
import torch
import numpy as np
import scipy.io.wavfile as wav
import tempfile
from .base_factory import BaseSynthesizer

# Try imports (transformers might not be installed)
try:
    from transformers import pipeline, AutoProcessor, BarkModel, SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan, VitsModel, AutoTokenizer
    from datasets import load_dataset
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

class HuggingFaceSynthesizer(BaseSynthesizer):
    """
    Synthesizer that uses Hugging Face 'transformers' pipelines.
    Supports:
      - suno/bark (Codec/Quantized artifacts)
      - microsoft/speecht5_tts (Vocoder/Flow artifacts)
      - facebook/mms-tts (VITS artifacts) [Future expansion]
    """

    def __init__(self, model_name: str, device: str = "cpu"):
        self.model_name = model_name
        self.device = 0 if torch.cuda.is_available() else -1
        self._pipe = None
        self._processor = None
        self._model = None
        self._vocoder = None
        self._speaker_embeddings = None # For SpeechT5
        self._tokenizer = None # For MMS

        if not TRANSFORMERS_AVAILABLE:
            print(f"⚠️ Transformers not installed. Skipping {model_name}.")
            return

    def _load_model(self):
        """Lazy load the specific model pipeline."""
        if self._pipe is not None or self._model is not None:
             return

        print(f"⏳ Loading HF Model: {self.model_name}...")

        try:
            # --- CASE 1: BARK (Suno) ---
            if "bark" in self.model_name:
                # Use the dedicated pipeline for simplicity
                self._pipe = pipeline("text-to-speech", model="suno/bark", device=self.device)
            
            # --- CASE 2: SpeechT5 (Microsoft) ---
            elif "speecht5" in self.model_name:
                # SpeechT5 requires explicit processor/vocoder loading usually
                # But 'text-to-speech' pipeline supports it if simplified.
                # However, SpeechT5 needs speaker embeddings. The pipeline handles this auto-magically? 
                # Let's use the explicit way for safety.
                
                device_str = "cuda" if self.device == 0 else "cpu"
                self._processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")
                self._model = SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5_tts")
                self._vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan")
                
                self._model.to(device_str)
                self._vocoder.to(device_str)

                # Load xvector speaker embedding (generic default)
                embeddings_dataset = load_dataset("Matthijs/cmu-arctic-xvectors", split="validation", trust_remote_code=True)
                speaker_embeddings = torch.tensor(embeddings_dataset[7306]["xvector"]).unsqueeze(0)
                self._speaker_embeddings = speaker_embeddings.to(device_str)

            # --- CASE 3: MMS (Meta) ---
            elif "mms" in self.model_name:
                self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                self._model = VitsModel.from_pretrained(self.model_name)
                
                device_str = "cuda" if self.device == 0 else "cpu"
                self._model.to(device_str)

        except Exception as e:
            print(f"❌ Failed to load {self.model_name}: {e}")
            self._pipe = "FAILED"

    def synthesize(self, text: str) -> Optional[np.ndarray]:
        """Generate audio and return as numpy array."""
        if not TRANSFORMERS_AVAILABLE:
            return None

        self._load_model()
        if self._pipe == "FAILED":
            return None

        try:
            # --- BARK ---
            if "bark" in self.model_name:
                # Bark pipeline output: {'audio': array, 'sampling_rate': int}
                output = self._pipe(text)
                audio_data = output["audio"][0] # (samples,)
                # Transformers output is float [-1, 1], which is what we want for internal processing
                return audio_data

            # --- SpeechT5 ---
            elif "speecht5" in self.model_name:
                inputs = self._processor(text=text, return_tensors="pt")
                device_str = "cuda" if self.device == 0 else "cpu"
                inputs = {k: v.to(device_str) for k, v in inputs.items()}
                
                with torch.no_grad():
                    speech = self._model.generate_speech(inputs["input_ids"], self._speaker_embeddings, vocoder=self._vocoder)
                
                # Convert tensor to numpy
                speech_cpu = speech.cpu().numpy()
                return speech_cpu

            # --- MMS (Meta) ---
            elif "mms" in self.model_name:
                inputs = self._tokenizer(text, return_tensors="pt")
                device_str = "cuda" if self.device == 0 else "cpu"
                inputs = {k: v.to(device_str) for k, v in inputs.items()}

                with torch.no_grad():
                    output = self._model(**inputs).waveform

                speech_cpu = output.cpu().numpy()[0] # (samples,)
                return speech_cpu

        except Exception as e:
            print(f"❌ Error synthesizing with {self.model_name}: {e}")
            return None
        
        return None
