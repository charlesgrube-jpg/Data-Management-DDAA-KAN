# Voice-Disjoint Pipeline Design V2 (Optimized)
**Goal:** Prevent "Generator Leakage" while maintaining **Streaming Performance** and supporting **Single-Voice Models**.

This V2 design supersedes V1 by addressing memory bottlenecks and single-speaker constraints.

## 1. The Streaming Splitter (Hash-Based) ⚡
**Problem in V1:** Loading all samples to shuffle/split breaks streaming for large datasets (100GB+).
**Solution:** Deterministic Hash Splitting. We decide the split for a sample *instantly* based on its `speaker_id` string, without needing to see the rest of the dataset.

**Logic:**
```python
def get_split_for_speaker(speaker_id: str, train_ratio=0.8, val_ratio=0.1) -> str:
    # Use MD5 or simple hash to map string -> float [0.0, 1.0]
    # This is DETERMINISTIC. "Speaker_123" will ALWAYS be 'train', today and tomorrow.
    h = int(hashlib.md5(speaker_id.encode()).hexdigest(), 16)
    p = (h % 10000) / 10000.0  # Normalize to 0-1
    
    if p < train_ratio:
        return 'train'
    elif p < train_ratio + val_ratio:
        return 'val'
    else:
        return 'test'
```

**Workflow:**
1.  **Phase 1 (Load):** Pipeline iterates through Common Voice dataset (streaming).
2.  **Phase 2 (Split):** For each sample, calc `split = get_split_for_speaker(sample['client_id'])`.
    *   *No memory overhead.*
    *   *Perfect reproducibility.*

---

## 2. Handling Single-Voice Models (The "Train-Only" Constraint) 🚫
**Problem in V1:** MMS (English) has only 1 voice. It cannot be in Train AND Test (leakage).
**Solution:** `Train-Only` Models.
We explicitly explicitly tag models that lack diversity. These models are **banned** from the Test loop.

**Implication for Testing:**
*   Our Test Set will consist **only** of multi-speaker architectures (Edge-TTS, Bark, RVC).
*   MMS will contribute to training robustness but will not be used for self-evaluation (which is fine, we want to test on *harder* models anyway).

---

## 3. Configuration V2 (`config.yaml`)

```yaml
synthesis:
  # 1. Define Pools for Multi-Voice Generators
  voice_pools:
    train:
      - "edge-tts:en-US-GuyNeural"
      - "edge-tts:en-US-JennyNeural"
      - "bark:v2/en_speaker_6"
    test:
      - "edge-tts:en-US-AriaNeural"     # Unseen Edge Voice
      - "bark:v2/en_speaker_9"          # Unseen Bark Voice

  # 2. Define "Global" Train-Only Models (Single Speaker)
  train_only_models:
    - "mms:eng"
    - "speecht5_tts" 
    # SpeechT5 is here because we are skipping the vector-banking complex logic for now.
    # It will just be a "training augmentor".

  # 3. Strategy
  strategy: "balanced_subset" 
```

---

## 4. Execution Logic (`run_pipeline.py`)

```python
def synthesize_sample(sample, config):
    split = sample['split']  # 'train' or 'test'
    
    # CASE A: Test Set
    if split == 'test':
        # MUST pick from the strict 'test' pool. 
        # NO MMS. NO SpeechT5. NO GuyNeural.
        allowed_voices = config.synthesis.voice_pools['test']
        model = random.choice(allowed_voices)
        
    # CASE B: Train Set
    else:
        # Can pick from 'train' pool OR 'train_only_models'
        pool_voices = config.synthesis.voice_pools['train']
        train_only = config.synthesis.train_only_models
        
        # Weighted choice to ensure MMS gets used
        if random.random() < 0.3:  # 30% chance for "Simple Models"
            model = random.choice(train_only)
        else:
            model = random.choice(pool_voices)
            
    # Run Synthesis...
    manager.synthesize(text, model)
```

## 5. Summary of V2 Improvements
1.  **Infinite Scale:** Hash-splitting supports datasets of any size (TB+).
2.  **Leakage-Proof:** Single-voice models are mathematically prevented from touching the test set.
3.  **Simpler Code:** No complex global shuffling or state management.
