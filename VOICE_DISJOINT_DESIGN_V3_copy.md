# Voice-Disjoint Pipeline Design V3 (Final)
**Goal:** Prevent "Generator Leakage" and enforce strict Train/Test separation at the **Generator Voice** level, while supporting streaming for large datasets.

This plan synthesizes the architectural shift from V1, the optimization from V2, and the risk mitigations from the quality review.

---

## 1. Objectives & Scope
*   **Primary Objective:** Ensure that synthetic voices used in the Test set are **never** seen during Training.
*   **Secondary Objective:** Maintain O(1) memory usage (Streaming) for dataset splitting.
*   **Scope:**
    *   Refactor `config.yaml` to use strict `voice_pools`.
    *   Refactor `splitter.py` to use stateless hashing.
    *   Refactor `run_pipeline.py` to move splitting to Phase 2 (Pre-Synthesis).
    *   Implement `verify_leakage.py` for post-run validation.

---

## 2. Assumptions & Dependencies
*   **Assumption:** The `speaker_id` field in Common Voice is consistent for a given speaker.
*   **Assumption:** Hash collisions (MD5) are negligible for this purpose.
*   **Dependency:** External datasets (ElevenLabs/RVC) must be loaded *before* the main pipeline loop if we want to run them in one go, OR handled as separate "Eval-Only" runs. (Decision: We will handle them as pre-split samples in the main loop).

---

## 3. Architecture & Logic

### A. Configuration Reform (`config.yaml`)
We **deprecate** `tts_models` list. It will be replaced by `voice_pools`.

```yaml
synthesis:
  # DEPRECATED: tts_models: []  <-- Remove this to prevent confusion
  
  # NEW STRUCTURE
  voice_pools:
    train:
      - "edge-tts:en-US-GuyNeural"
      - "edge-tts:en-US-JennyNeural"
      - "bark:v2/en_speaker_6"
    test:
      - "edge-tts:en-US-AriaNeural"     # Unseen Edge Voice
      - "bark:v2/en_speaker_9"          # Unseen Bark Voice

  # Generators that are allowed but restricted to TRAIN only (Single-voice models)
  train_only_models:
    - "mms:eng"
    - "speecht5_tts" 
```

### B. The Guarded Hash Splitter (`splitter.py`)
Must respect existing splits (External Data) and use deterministic hashing.

```python
def assign_split(sample: dict, train_ratio=0.8) -> dict:
    # 1. GUARD CLAUSE: Respect Pre-assigned Splits (e.g., ElevenLabs)
    if 'split' in sample:
        return sample

    # 2. DETERMINISTIC HASHING
    # Use speaker_id (Real Source) to ensure content disjointness
    speaker_key = sample.get('speaker_id', 'unknown')
    h = int(hashlib.md5(speaker_key.encode()).hexdigest(), 16)
    p = (h % 10000) / 10000.0
    
    # 3. ASSIGN
    if p < train_ratio:
        sample['split'] = 'train'
    elif p < train_ratio + 0.1: # 10% val
        sample['split'] = 'val'
    else:
        sample['split'] = 'test'
        
    return sample
```

### C. The Synthesis Selection Loop (`run_pipeline.py`)

```python
def synthesize_step(sample, config):
    start_time = time.time()
    
    # 1. Determine Target Pool
    split = sample['split']
    
    # 2. Select Voice
    if split == 'test':
        # STRICT: Must come from Test Pool
        valid_voices = config.synthesis.voice_pools['test']
        if not valid_voices:
            raise ValueError("Test set is empty! Check config.")
        selected_model = random.choice(valid_voices)
        
    else: # Train or Val
        # FLEXIBLE: Train Pool OR Train-Only Models
        pool = config.synthesis.voice_pools['train']
        train_only = config.synthesis.train_only_models
        
        # Weighted selection (70% Pool, 30% Train-Only)
        if random.random() < 0.3 and train_only:
            selected_model = random.choice(train_only)
        else:
            selected_model = random.choice(pool)
            
    # 3. FORCE GENERATION
    # Bypass manager strategy, force specific voice
    synth = manager.get_synthesizer(selected_model)
    # The 'synthesize' method needs to accept the specific preset override
    audio = synth.synthesize(sample['text'], voice_preset=selected_model)
    
    # 4. Record Metadata
    sample['generator_voice'] = selected_model
    return audio, sample
```

---

## 4. Prioritized Action Steps

### Phase 1: Infrastructure
1.  **Refactor Config:** Update `config.yaml` and `pipeline/config.py` to support `voice_pools` and `train_only_models`. Add validation to ensure pools are not empty.
2.  **Refactor Splitter:** Rewrite `splitter.py` to implement the Guarded Hash Logic.

### Phase 2: Pipeline Logic
3.  **Update Pipeline Loop:** Modify `run_pipeline.py`.
    *   Insert `assign_split` call immediately after loading.
    *   Update Phase 4 (Synthesis) to use the new "Split-Based Selection" logic.
4.  **Update Synthesizer Manager:** Ensure `synthesize()` accepts and respects the forced `voice_preset`.

### Phase 3: External Data & Verification
5.  **Ingest External Scripts:** Implement `download_benchmarks.py` which pre-tags 11Labs/RVC data with `split='test'`.
6.  **Verification Script:** Create `utilities/verify_leakage.py` to analyze the final `metadata.json` and prove strict disjointness.

---

## 5. Risks & Mitigations

| Risk | Mitigation |
| :--- | :--- |
| **Empty Pools:** Test Pool is empty in config or due to heavy filtering. | **Mitigation:** Add startup validation check in `run_pipeline.py`. Fail fast if pools are empty. |
| **External Data Overwrite:** Hash splitter re-assigns ElevenLabs data to 'train'. | **Mitigation:** Implemented "Guard Clause" in step 3B. |
| **MMS Leakage:** MMS accidentally used in Test because logic defaulted to "random". | **Mitigation:** Explicit `split == 'test'` check enforces STRICT pool selection. MMS is never in the Test pool. |
| **Statistical Imbalance:** Small debug runs (N=10) might get 0 test samples. | **Mitigation:** Acceptable for debug. Warn user if N < 100. |

---

## 6. Deliverables
1.  Updated `config.yaml`.
2.  Updated `pipeline/data_gen/splitter.py`.
3.  Updated `run_pipeline.py`.
4.  New `utilities/verify_leakage.py`.
