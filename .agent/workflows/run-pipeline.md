---
description: Run the main TTS deepfake detection pipeline
---

# Run TTS Pipeline

This workflow runs the complete pipeline that generates paired real/synthetic audio dataset for deepfake detection.

## Steps

1. Ensure dependencies are installed:
```bash
pip install -r requirements.txt
```

2. Verify Mozilla Common Voice data exists:
   - Check that `mozilla_cv_data/extracted/cv-corpus-24.0-2025-12-05/en` exists
   - If not, run the download utility first

// turbo
3. Run the pipeline:
```bash
python run_pipeline.py
```

4. Check output in the timestamped `processed_dataset/` directory

## What the pipeline does:
- Loads Mozilla Common Voice samples
- Preprocesses and segments audio
- Synthesizes fake audio using gTTS
- Applies channel effects (clean/mobile/noisy)
- Creates speaker-disjoint train/val/test splits
- Validates dataset balance
- Exports to WAV files with metadata.csv
