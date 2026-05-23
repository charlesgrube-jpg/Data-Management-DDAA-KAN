# HF Integration & Pipeline Upgrade Summary
**Date:** 2025-12-28
**Branch:** `huggingface-stack`

## 1. New Generator Integration (Hugging Face)
Added support for 3 new model types covering different deepfake architectures:
*   **Bark (`suno/bark`):** Codec-based text-to-speech. (Slow on CPU, highly realistic).
*   **MMS (`facebook/mms-tts-eng`):** VITS/Flow-based architecture. (Fast, good quality).
*   **SpeechT5 (`microsoft/speecht5_tts`):** Vocoder/GAN-based architecture.
    *   *Fix:* Replaced deprecated `cmu-arctic-xvectors` dataset usage with random speaker embeddings to ensure functionality.

**Files Created:**
*   `pipeline/synthesizer/huggingface_synthesizer.py`: Logic for loading and running HF models.
*   `utilities/test_hf_synthesis.py`: Verification script for downloading and testing models in isolation.

## 2. Synthesizer Architecture Refactoring
Refactored the synthesizer system to be modular and extensible.
*   **Base Class:** Created `pipeline/synthesizer/base.py` defining `BaseSynthesizer`.
*   **Edge-TTS:** Promoted from a simple fallback to a **First-Class Citizen**.
    *   Created `pipeline/synthesizer/edge_tts_synthesizer.py` with `EdgeTTSSynthesizer` class.
    *   Enabled "edge-tts" in `config.yaml` to participate in random selection.
*   **Manager Update:** Updated `pipeline/synthesizer/synthesizer.py` (`SynthesizerManager`) to:
    *   Dynamically load HF models from config.
    *   Route "edge-tts:..." models to the new wrapper class.
    *   Support `random` picking strategy correctly across 4+ model types.

## 3. Configuration System Upgrades (`config.yaml` & `pipeline/config.py`)
*   **Features Config:** Added `features` section (CQT/LFCC params) to `config.yaml`.
*   **Code Fix:** Updated `pipeline/config.py` to correctly define and load `FeaturesConfig`, `CQTConfig`, and `LFCCConfig` dataclasses (previously missing).
*   **Parameters:**
    *   `silence_threshold_db`: Adjusted from `-40.0` to `60.0` (positive dynamic range) to fix "all audio trimmed" issue.
    *   `max_samples`: Configurable limit for test runs.

## 4. Pipeline Logic (`run_pipeline.py`)
*   **Feature Extraction (Phase 9):** Added automatic execution of feature extraction at the end of the pipeline.
    *   Extracts CQT (`.pt`) and LFCC (`.pt`) for all generated samples.
    *   Saves to `output/features/cqt` and `output/features/lfcc`.
*   **Bug Fixes:**
    *   Fixed `NameError` in final statistics printing (`processed_samples` -> `working_set`).
    *   Fixed `NameError` for missing metadata variables in final print.

## 5. Verification Tools
*   `utilities/test_features.py`: Script to verify CQT/LFCC extractor correctness.
    *   *Fix:* Updated extract calls to match signature `extract(audio)` without `sr`.

## 6. Dependencies
*   Updated `requirements_colab.txt` to include:
    *   `transformers`, `sentencepiece`, `accelerate`, `datasets` (for HF models).
