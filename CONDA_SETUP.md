# Environment Setup

## Quick Start (Recommended)

```bash
# Create clean environment
conda create -n ddaa-clean python=3.10 -y
conda activate ddaa-clean

# Install dependencies
pip install -r requirements.txt

# Run pipeline
python run_pipeline.py
```

## Verify Installation

```bash
python -c "import edge_tts; print('edge-tts OK')"
python -c "import librosa; print('librosa OK')"
python -c "import soundfile; print('soundfile OK')"
```

## Optional: FFmpeg for Codec Compression

To enable codec compression (MP3/AAC artifacts):

**Linux:**
```bash
apt install ffmpeg
```

**macOS:**
```bash
brew install ffmpeg
```

**Windows:**
Download from https://ffmpeg.org/download.html and add to PATH.

## Notes

- Python 3.10 recommended
- RVC/Coqui TTS require Docker or Linux (Windows has DLL conflicts)
- See `PIPELINE_SUMMARY.md` for full details
