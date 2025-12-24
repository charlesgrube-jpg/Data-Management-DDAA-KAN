---
description: Setup the development environment
---

# Setup Development Environment

This workflow sets up the Python environment and dependencies for the TTS deepfake detection pipeline.

## Steps

1. Create a virtual environment (recommended):
```bash
python -m venv venv
```

2. Activate the virtual environment:
   - Windows: `venv\Scripts\activate`
   - Linux/Mac: `source venv/bin/activate`

// turbo
3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Verify installation:
```bash
python -c "import numpy, librosa, gtts, soundfile; print('All core dependencies installed!')"
```

5. Create necessary directories:
```bash
mkdir -p mozilla_cv_data cache processed_dataset
```

## Dependencies
Key packages installed:
- `numpy`: Numerical operations
- `librosa`: Audio processing
- `soundfile`: Audio I/O
- `gtts`: Google Text-to-Speech synthesis
- `pydub`: Audio manipulation
- `datasets`: HuggingFace datasets (for Mozilla CV)
- `pyyaml`: Configuration parsing
