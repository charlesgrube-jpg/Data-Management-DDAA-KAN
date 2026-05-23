# Deepfake Detection Pipeline

## Quick Start

**Windows (Local Testing):**
```bash
conda create -n ddaa-clean python=3.10 -y
conda activate ddaa-clean
pip install -r requirements.txt
python run_pipeline.py
# Feature extraction now runs automatically as Phase 9 inside run_pipeline.py
# No separate extract_features command needed.
```

**Google Colab (Full Pipeline):**
1. Upload `DDAA_Pipeline_Colab.ipynb` to Colab
2. Set your GitHub repo URL
3. Run All

---

## Pipeline Flow

```
[Mozilla CV Audio] → [Preprocess] → [Real Audio]
        ↓
    [Transcript]
        ↓
[TTS Synthesis] → [Normalize] → [Synthetic Audio]
        ↓
[Apply Effects to BOTH] → [Codec Compress]
        ↓
[Validate & Export WAV]
        ↓
[Feature Extraction (CQT/LFCC)] → [PyTorch Tensors]
```

---

## Files

| File | Purpose |
|------|---------|
| `run_pipeline.py` | Generates audio dataset AND runs feature extraction (Phase 9 inline) |
| `scripts/preextract_features.py` | Standalone CQT/LFCC feature extraction (re-run only) |
| `scripts/preextract_ssl_features.py` | Standalone SSL feature extraction (re-run only) |
| `config.yaml` | All settings |
| `DDAA_Pipeline_Colab.ipynb` | One-click Colab notebook |

---

## Requirements

| File | Use Case |
|------|----------|
| `requirements.txt` | Windows/macOS local dev (no RVC/Coqui) |
| `requirements_colab.txt` | Google Colab (full: RVC, Coqui, GPU features) |

**Key Differences:**
- `requirements.txt`: Minimal, avoids DLL conflicts on Windows
- `requirements_colab.txt`: Includes `rvc-python`, `TTS`, `nnAudio` (Linux-only)

---

## ⚠️ Important Colab Notes (Python 3.12)

Due to incompatibilities between `fairseq` (required by RVC) and Python 3.12, standard `pip install` may fail.
If RVC models fail to load, use this manual installation snippet in a Colab cell:

```python
# Nuclear Option for RVC/Fairseq on Python 3.12
!pip uninstall -y fairseq rvc-python
!git clone https://github.com/facebookresearch/fairseq.git
!cd fairseq && pip install --no-deps .
!pip install --no-deps rvc-python
```
