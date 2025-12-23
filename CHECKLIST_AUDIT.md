# Checklist Audit - Data Preprocessing Pipeline

**Date:** 2023-12-23  
**Branch:** `feature/tts-pipeline-v1`  
**Status:** TTS-only pipeline complete, VC/codec/logging deferred

---

## Summary

| Phase | Status | Notes |
|-------|--------|-------|
| 1. Input Validation | ✅ Done | Mozilla CV loader working |
| 2. Speaker Splitting | ✅ Done | 70/15/15 with verification |
| 3. Audio Preprocessing | ✅ Done | Resample, normalize, chunk |
| 4. Synthetic Generation | ⚠️ TTS Only | gTTS works, VC placeholder |
| 5. Channel Effects | ⚠️ Partial | Effects work, no real codec |
| 6. Metadata | ✅ Done | All required fields present |
| 7. Balance | ✅ Done | 50/50 verified |
| 8. Leakage Prevention | ✅ Done | Speaker-disjoint verified |
| 9. File System | ✅ Done | 16-bit PCM WAV at 16kHz |
| 10. Config | ✅ Done | Config copied to output |
| 11. Error Handling | ⚠️ Partial | No formal logging |
| 12. Sanity Checks | Manual | User responsibility |

---

## Completed in This Branch ✅

1. **Config copied to output** - For reproducibility
2. **attack_category field** - Tracks attack type (tts/vc)
3. **snr_db field** - Documents noise level when applied
4. **source_real_file field** - Links synthetic → real pairs
5. **Generator versioning** - Now `gTTS-2.5.4` not just `gTTS`
6. **Modular folder structure** - Organized into standardizer/synthesizer/effects/data_gen

---

## What's Implemented ✅

- **Dataset Loading:** Mozilla Common Voice with speaker IDs, transcripts, gender, accent
- **Speaker-Disjoint Splits:** 70/15/15 train/val/test, no speaker overlap
- **Audio Preprocessing:** 16kHz, mono, RMS normalization to -20dB, silence trimming
- **Chunking:** 3-second fixed chunks with padding
- **Synthetic Generation:** gTTS from transcripts
- **Channel Effects:** 3 quality tiers (clean/mobile/noisy), matched pair processing
- **Validation:** Balance checking, duplicate detection, speaker leakage verification
- **Export:** Structured directories, metadata CSV, manifest JSON, config copy

---

## Deferred to Future Branches 🔜

### High Priority (Next Branch)
| Item | Description |
|------|-------------|
| **Voice Conversion** | Implement RVC/FreeVC (requires Python <3.12) |
| **Better TTS** | OpenAI Whisper, Coqui XTTS, Bark |

### Medium Priority
| Item | Description |
|------|-------------|
| **Real codec effects** | Actual MP3/opus compression (not just lowpass) |
| **Per-tier balance check** | Validate 50/50 within each quality tier |
| **Formal logging** | Write to preprocessing.log file |

### Low Priority
| Item | Description |
|------|-------------|
| **Variable SNR** | Dynamic SNR instead of fixed 15dB |
| **More noise types** | Pink noise, babble noise, room noise |
