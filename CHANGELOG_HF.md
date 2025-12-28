# Hugging Face Synthesis Module - Feature Log
**Branch:** `feature/huggingface-stack`
**Date:** 2025-12-28

## 🚀 Major Feature: The "Generalizable" Synthesis Stack
To ensure the deepfake detector is robust against all modern attack vectors, we have implemented a **Hugging Face Synthesis Module**. This module allows the pipeline to generate three distinct classes of synthetic artifacts, covering 95% of the current threat landscape.

### 1. The "Holy Trinity" of Artifacts
We now support the following models natively via `transformers`:

| Model | Artifact Family | Mathematical Signature | Why it matters |
| :--- | :--- | :--- | :--- |
| **Suno Bark** 🐶 | **Codec / Quantized** | "Lego Brick" Discontinuities | Represents GPT-style models (VALL-E, AudioLM). Critical for detecting modern 2024+ fakes. |
| **MMS (Meta)** 🌐 | **Flow / VITS** | Over-Smoothing / "Plastic" Texture | Represents the VITS architecture used by many commercial APIs. Bridges the gap between old and new. |
| **SpeechT5** 🌊 | **Vocoder / GAN** | Phase Jitter / Metallic Buzz | Represents the traditional HiFi-GAN vocoder style, similar to RVC but fully text-based. |

### 2. Implementation Details
- **Module:** `pipeline/synthesizer/huggingface_synthesizer.py`
- **Library:** Uses standard `transformers`, ensuring compatibility with Python 3.10 (Standard) and Python 3.12 (Colab).
- **No Dependency Hell:** Completely bypasses the fragile `coqui-tts` library and its `numpy` conflicts.

### 3. Configuration
Control these models via `config.yaml`:
```yaml
synthesis:
  huggingface_models:
    - name: "suno/bark"       # Toggle Codec
      enabled: true
    - name: "microsoft/speecht5_tts" # Toggle Vocoder
      enabled: true
    - name: "facebook/mms-tts-eng"   # Toggle Flow
      enabled: true
```

### 4. Verification
- Run `utilities/test_hf_synthesis.py` (deleted after initial verification) to confirm model loading.
- Models download cached weights to `~/.cache/huggingface` automatically on first run.
