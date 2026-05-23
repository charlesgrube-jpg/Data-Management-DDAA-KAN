# Changelog & Handoff Notes
**Date:** 2025-12-26

## 🚀 Key Improvements

### 1. Data Ingestion (Fixed)
- **Resolved `FileNotFoundError`:**
  - Added repository auto-detection and directory switching in `DDAA_Pipeline_Colab.ipynb`.
  - Implemented `config.source.data_path` parameter to override default paths.
- **Interactive API Download:**
  - Updated `utilities/download_mozilla_api.py` to auto-detect the API Key and generate a fresh Download Token automatically (user leaves token blank).
- **Nested Folder Auto-Fix:**
  - Added logic to `setup_data()` in the notebook that detects deeply nested `clips` directories (common in Mozilla tarballs) and moves them to the root `mozilla_cv_data/` folder.

### 2. Pipeline Enhancements
- **Data Size Limiting:**
  - Added `config.source.max_size_mb` (default 100MB in Colab) to strictly limit processing volume for rapid testing.
  - Implemented tracking in `mozilla_cv_loader.py`.
- **Throughput Estimation:**
  - Updated `run_pipeline.py` to calculate MB/s processing speed and estimate time remaining for the full 3.5GB dataset.
- **Data Purity:**
  - **CRITICAL:** Disabled `fallback_on_error` in `config.yaml` (`true` -> `false`).
  - Previously, if TTS failed, the pipeline used Real audio as a placeholder for Fake audio. This is now disabled to prevent label noise in deepfake detection training.

### 3. Environment & Dependencies (Python 3.12 / Colab)
- **Repo Compatibility:**
  - Added `%cd Data-Management-DDAA-KAN` logic to ensure scripts run from the correct root.
- **Dependency Conflict Workarounds:**
  - Identified `fairseq` incompatibility with Python 3.12 (`mutable default` error).
  - Identified `rvc-python` incompatibility due to strict `numpy<=1.25` requirement vs Colab's `numpy>=1.26`.
  - Provided manual installation cells (git clone + no-deps install) to bypass these conflicts.

### 4. New Features (Generalizability)
- **Hugging Face Synthesis Module:**
  - Integrated `transformers` to support **Suno Bark** (Codec Artifacts) and **SpeechT5** (Vocoder Artifacts).
  - This diversifies the dataset beyond just GAN artifacts (RVC), crucial for robust detection.
  - Avoids "Dependency Hell" of Coqui TTS by using standard Hugging Face pipelines.

---

## ⚠️ Known Issues & Remaining Work

### 1. RVC / Fairseq Installation is Fragile
**Status:** 🔴 Blocked/Tricky
**Context:** `rvc-python` depends on `fairseq`. standard `fairseq` (0.12.2) crashes on Python 3.12. The GitHub version of `fairseq` works, but conflicts with `hydra-core` during Pip resolution.
**Current Workaround:**
We must use a "Nuclear" installation method in Colab:
```bash
git clone https://github.com/facebookresearch/fairseq.git
cd fairseq
pip install --no-deps .
pip install --no-deps rvc-python
```
**Future Fix:** Create a custom Docker container or pre-built wheel for `fairseq` 3.12 to avoid this manual compilation step.

### 2. Edge-TTS Reliability
**Status:** 🟡 Intermittent Failures
**Context:** ~20-30% of TTS requests fail with "No audio received". This is likely due to API rate limiting or text content.
**Recommendation:** Implement a retry mechanism with exponential backoff in `pipeline/synthesizer/edge_tts_synthesizer.py`.

### 3. Feature Extraction (Next Step)
**Status:** ⚪ Not Run
**Context:** We generated the audio dataset (`Phase 1-3`), but `Phase 4: Extract Features` (CQT/LFCC) has not been verified on the new dataset structure.

---

## 🛠️ Handoff Instructions for Developer

1. **Repo Structure:** Code is in line with `PIPELINE_FLOW.md`.
2. **Configuration:** Check `config.yaml` for defaults. In Colab, we override `max_size_mb` dynamically.
3. **Environment:**
   - Use `requirements_colab.txt` for base deps.
   - **Crucial:** You CANNOT simple `pip install -r requirements.txt` for RVC/Fairseq on Python 3.12. You must follow the manual build steps above or downgrade to Python 3.10.
