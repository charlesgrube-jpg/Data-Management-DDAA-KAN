# Changelog: Hugging Face & RVC Integration (Cumulative)

## 📌 Major Implementations

### 1. Robust HF Integration (`pipeline/utils/huggingface_hub.py`)
- Created centralized `HFClient` for robust model downloading.
- Implemented automatic integrity checks (min size + Zip magic bytes).
- Removed redundant HF logic from `download_models.py` and `download_rvc_models.py`.

### 2. RVC Split-Awareness (Voice Disjointness)
- **Config:** Added `VCPoolsConfig` to enforce split separation (Train/Test/Val pools).
- **Core Logic:** Replaced legacy random selection with `_synthesize_vc_split_aware()` in `synthesizer.py`.
- **Validation:** Updated `verify_leakage.py` to assert RVC model disjointness alongside TTS voices.
- **Paths:** Implemented path normalization to prevent leakage via string mismatch.

### 3. Security Hardening (.pth Scanning)
- Created `pipeline/utils/pth_security.py` using `fickling` (when compatible) or fail-safe checks.
- **Fail-Safe Mode:** If Fickling fails (or on Python 3.14+ where it is incompatible), the system falls back to basic sanity checks (size, extension) and requires explicit `ALLOW_UNSAFE_PTH=1` override to proceed.
- **Quarantine:** Automatic quarantine for failed/suspicious downloads in `models/rvc/quarantine`.

### 4. Registry & Data Management
- Created `utilities/rvc_model_registry.yaml` as the source of truth for RVC models.
- Resolved 5/8 broken "QuickWick" model URLs using `resolve_rvc_urls.py`.
- Refactored `download_rvc_models.py` to use the new Registry + HFClient + Security Scanner.

---

## 🔧 Utilities Created
| Script | Purpose |
| :--- | :--- |
| `utilities/validate_rvc_links.py` | Check liveness of registry URLs |
| `utilities/resolve_rvc_urls.py` | Find correct paths in messy HF repos |
| `utilities/unquarantine_and_scan.py` | Restore quarantined files after verification |
| `tests/test_rvc_logic.py` | Mock integration test for split logic |

## ⚠️ Known Limitations (Handover Notes)
1. **Incomplete Downloads:** The download of new female models (Adele, Rihanna, etc.) was **interrupted**. You MUST run `python utilities/download_rvc_models.py` again to finish fetching them.
2. **Python 3.14+ Compatibility:** `fickling` is incompatible with Python 3.14. Files will be auto-quarantined.
3. **Quarantine & Restoration:**
   - To restore files, run: `$env:ALLOW_UNSAFE_PTH="1"; python utilities/unquarantine_and_scan.py`
   - **Ambiguous Files:** Files like `model.pth` (Taylor Swift) or `SpongebobSquarepants.pth` may remain in `models/rvc/quarantine` because the script can't decide where they belong. **Manual Move Required**:
     - `model.pth` -> `models/rvc/TaylorSwift/model.pth`
     - `SpongebobSquarepants.pth` -> `models/rvc/Spongebob/Spongebob.pth`
     - `D_*.pth` / `G_*.pth` -> Base pre-trained models (can be moved to `models/rvc/base` or ignored).
