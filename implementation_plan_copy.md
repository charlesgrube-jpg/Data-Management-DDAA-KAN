# Implementation Plan: Information Disjoint Strategy - External Data 🌐

**Goal:** Expand the dataset with high-quality external deepfakes while ensuring **Strict Split Separation**.
**Constraint:** External datasets must be assigned to `test` or `val` splits to simulate "unseen" attacks.

## External Data Overview

| Dataset | Split Assignment | Rationale |
| :--- | :--- | :--- |
| **ElevenLabs** | `test` (100%) | High-quality "unseen" attack vector. |
| **ASVspoof 2019/21** | `test` (100%) | Academic benchmark standard. |
| **Pre-generated RVC** | `val` / `test` | Validation for voice conversion robustness. |

## Proposed Changes

### 1. New Ingestion Script
#### [NEW] `utilities/download_external_datasets.py`
A comprehensive downloader that:
1.  Downloads datasets from Kaggle/HuggingFace/Url.
2.  Normalizes directory structure.
3.  Generates a `metadata_external.json` mapping each file to:
    *   `split`: 'test' (Hardcoded for ElevenLabs/ASVspoof)
    *   `is_synthetic`: True
    *   `method`: 'elevenlabs' / 'asvspoof' / 'rvc'

### 2. Configuration Update
#### [MODIFY] `config.yaml`
Add `external_datasets` section:
```yaml
external_datasets:
  elevenlabs:
    enabled: false # Disabled: Language bias (skypro1111 is Ukrainian)
    path: "./external_data/elevenlabs"
    split: "test"
  asvspoof:
    enabled: false # Optional
    path: "./external_data/asvspoof"
    split: "test"
```

### 3. Pipeline Integration
#### [MODIFY] `run_pipeline.py`
*   Add **Phase 1.8: Ingest External Data**.
*   Load `metadata_external.json`.
*   Merge external samples into `working_set` **before** Phase 2.
*   **Critical:** Ensure `assign_split` respects the pre-assigned `split` field (Already implemented!).

## Verification Plan
1.  **Download Test:** Run `download_external_datasets.py` (mock small download).
2.  **Ingestion Test:** Run pipeline, verify external samples appear in `output/processed_dataset`.
3.  **Leakage Test:** Run `verify_leakage.py` to confirm they strictly populate the `test` folder.
