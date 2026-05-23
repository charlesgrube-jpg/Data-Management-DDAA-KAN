# Environment Setup

## Quick Start

```bash
# Create environment
conda create -n ddaa-clean python=3.10 -y
conda activate ddaa-clean

# Install dependencies
pip install -r requirements.txt

# Run pipeline
python run_pipeline.py

# Extract features
python -m pipeline.features.extract_features --type cqt
```

## Verify Installation

```bash
python -c "import edge_tts; print('edge-tts OK')"
python -c "import librosa; print('librosa OK')"
python -c "import torchaudio; print('torchaudio OK')"
```

## Google Colab

For full pipeline (RVC, Coqui, GPU features), use Google Colab:
1. Upload `DDAA_Pipeline_Colab.ipynb`
2. Uses `requirements_colab.txt` automatically

## requirements.txt vs requirements_colab.txt

| File | Platform | Includes |
|------|----------|----------|
| `requirements.txt` | Windows/Mac | Edge-TTS, gTTS, librosa |
| `requirements_colab.txt` | Linux/Colab | + RVC, Coqui TTS, nnAudio (GPU) |

Windows has DLL conflicts with RVC/Coqui. Use Colab for full synthesis.
