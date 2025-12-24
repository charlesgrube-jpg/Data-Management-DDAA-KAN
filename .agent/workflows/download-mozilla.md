---
description: Download Mozilla Common Voice dataset
---

# Download Mozilla Common Voice

This workflow downloads a sample of the Mozilla Common Voice dataset for testing the pipeline.

## Steps

1. Ensure you have the download utility:
   - Check `utilities/download_mozilla_api.py` exists

2. Run the download script:
```bash
python utilities/download_mozilla_api.py
```

3. The script will download to `mozilla_cv_data/extracted/`

4. Verify the data was extracted properly:
   - Should have `cv-corpus-24.0-2025-12-05/en/` directory
   - Contains `clips/` subdirectory with audio files
   - Contains TSV metadata files

## Configuration
Edit `config.yaml` to adjust:
- `source.max_samples`: Number of samples to download (start with 100)
- `source.language`: Language code (default: "en")
