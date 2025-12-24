---
description: Validate pipeline output and dataset quality
---

# Validate Pipeline Output

This workflow helps you validate the output from a pipeline run.

## Steps

1. Find the latest output directory:
```bash
ls -la processed_dataset/
```

2. Check the metadata file:
```bash
python -c "import pandas as pd; df = pd.read_csv('processed_dataset/run_YYYYMMDD_HHMMSS/metadata.csv'); print(df.info()); print(df.head())"
```
   (Replace `run_YYYYMMDD_HHMMSS` with actual timestamp)

3. Verify dataset balance:
```bash
python -c "import pandas as pd; df = pd.read_csv('processed_dataset/run_YYYYMMDD_HHMMSS/metadata.csv'); print('Real vs Synthetic:'); print(df['is_synthetic'].value_counts()); print('\nSplit distribution:'); print(df['split'].value_counts())"
```

4. Check audio file counts:
```bash
ls processed_dataset/run_YYYYMMDD_HHMMSS/train/real/*.wav | wc -l
ls processed_dataset/run_YYYYMMDD_HHMMSS/train/synthetic/*.wav | wc -l
```

5. Listen to sample outputs:
   - Open audio files in your preferred audio player
   - Check quality tiers: clean vs mobile vs noisy
   - Compare real vs synthetic pairs

## What to Check
- [ ] Metadata CSV has correct columns
- [ ] Real/synthetic balance is ~50/50
- [ ] Train/val/test splits follow config ratios (70/15/15)
- [ ] No speaker leakage between splits
- [ ] Audio files match metadata entries
- [ ] Quality tiers distributed correctly
- [ ] Transcripts are preserved
