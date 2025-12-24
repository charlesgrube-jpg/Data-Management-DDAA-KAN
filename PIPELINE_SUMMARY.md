# Deepfake Detection Pipeline - Status & Roadmap

## 1. Project Overview
This project targets the creation of a robust dataset for training **deepfake detection models** capable of defending against **imperceptible adversarial attacks**. The pipeline ingests real speech (Mozilla Common Voice), generates synthetic counterparts (TTS/VC), employs rigorous data splitting, and validates the output for research utility.

## 2. Current Status: Prototype / Proof-of-Concept
| Component | Status | Verification |
|-----------|--------|--------------|
| **Architecture** | ✅ **Robust** | Modular design (Data/Synth/Effects), Config-driven, Reproducible. |
| **Data Ingestion** | ✅ **Functional** | Loads/Filters Mozilla CV MP3s, handles metadata correctly. |
| **Synthesis** | ⚠️ **Partial** | **TTS Only** (Edge-TTS + gTTS). No Voice Conversion (RVC) due to Windows dependency conflicts. |
| **Degradation** | ❌ **Disabled** | Codec/Channel effects implemented but currently disabled (requires FFmpeg binary). |
| **Validation** | ✅ **Strict** | Speaker-disjoint splits work. Leakage checks passing. |
| **Dataset Size** | ⚠️ **Tiny** | ~132 samples (limited by source inputs). Need 10k+ for real research. |

**Verdict:** The pipeline is a **functional skeleton**. It proves the data engineering flow works but produces a dataset **insufficient for training a research-grade detector** due to lack of synthesis variety (only TTS, no VC) and volume.

---

## 3. Changes Implemented (Since Repo Pull)

### A. Architecture Refactor
*   **Modularization:** Split monolithic scripts into `pipeline/` modules: `data_gen`, `synthesizer`, `effects`.
*   **Configuration:** Created `config.yaml` to centralize all parameters (paths, splits, model choices) removing hardcoded values.
*   **Pipeline Manager:** Rewrote `run_pipeline.py` to orchestrate proper setup, execution, and error handling.

### B. Synthesis Engine Overhaul
*   **Edge-TTS Integration:** Added `edge_tts_synthesizer.py` (Microsoft Neural Voices) as a high-quality, pure-Python alternative to conflicting libraries.
*   **Variety Injection:** Configured pipeline to randomly select from 10+ distinct neural voices to prevent model overfitting to a single TTS voice.
*   **Fallback Logic:** Implemented robust fallback: `Coqui TTS` -> `Edge-TTS` -> `gTTS`.

### C. Data Engineering & Validation
*   **Mozilla CV Loader:** Fixed MP3 parsing, TSV path handling (`clips/` prefix issues), and added mixed-type warnings suppression.
*   **Strict Splitting:** Implemented speaker-disjoint splitting (e.g., Speaker A is ONLY in Train, Speaker B ONLY in Test) to prevent "speaker recognition" instead of "deepfake detection".
*   **Validator:** Created `validate_pipeline.py` which automatically audits output for:
    *   Speaker Leakage (Critical)
    *   Class Balance
    *   Metadata integrity

### D. Environment Stabilization
*   **Dependency Hell Fix:** Migrated from a broken Conda environment (conflicting C++ DLLs for RVC/Fairseq) to a clean, minimal environment (`ddaa-clean`) focusing on pure Python libraries (`edge-tts`, `librosa`, `pandas`).

---

## 4. Current Limitations & Risks

### 🚨 Lack of Voice Conversion (VC)
*   **Risk:** The detector will only learn to spot **TTS artifacts** (pronunciation errors, robotic rhythm). It will fail completely against **Voice Conversion** (RVC, So-Vits) which preserves human rhythm and only changes timbre.
*   **Adversarial Context:** An attacker using RVC will bypass a detector trained on this dataset 100% of the time.

### 🚨 Zero-Degradation Training
*   **Risk:** Real social media audio is compressed (AAC/Opus) and noisy. Our current dataset is too "clean". A detector trained on this might fail when presented with a slightly compressed WhatsApp voice note.

### 🚨 Small Data Volume
*   **Risk:** 132 samples is not enough for neural network training. Models will overfit instantly.

---

## 5. Roadmap to Research-Grade

To verify defenses against adversarial attacks, the pipeline must be upgraded.

### Phase 1: Environment & Tools (Immediate)
- [ ] **Dockerization:** Create a Docker image that pre-installs `ffmpeg`, `fairseq`, `RVC`, and `Coqui TTS` on Linux. This solves all Windows DLL conflicts permanently (`conda` on Windows is a dead end for these specific libraries).
- [ ] **FFmpeg Integration:** Ensure `ffmpeg` binaries are available to enable Codec Compression (MP3/AAC artifacts).

### Phase 2: Synthesis Expansion (Critical)
- [ ] **Enable RVC:** reliable Voice Conversion is mandatory.
- [ ] **Add OpenVoice/So-Vits:** More architectures = better generalization.
- [ ] **Cloning:** Implement "Zero-shot" cloning to test against unseen speakers.

### Phase 3: Data Scale
- [ ] **Scale Up:** Run pipeline on full Mozilla Common Voice dataset (or at least 10GB subset).
- [ ] **Target:** 10,000+ samples (5k Real / 5k Fake).

### Phase 4: Adversarial Training
- [ ] **Train Baseline Detector:** ResNet/RawNet2 on the generated dataset.
- [ ] **Attack:** Generate adversarial examples (FGSM, PGD) against the detector.
- [ ] **Defend:** Retrain detector on adversarial examples.
