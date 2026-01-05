# Voice-Disjoint Pipeline Design
**Goal:** Prevent "Generator Leakage" by ensuring synthetic voices seen during testing were NEVER seen during training.

## 1. The Architectural Shift
Currently, our pipeline splits data in **Phase 6** (after synthesis). This is too late. If we want to assign specific voices to specific splits, we must determine the split **before** we generate the audio.

**New Flow:**
1.  **Phase 1:** Load Common Voice (Real Audio).
2.  **Phase 2 (NEW):** **Assign Splits Immediately.**
    *   Mark every real sample as `train`, `val`, or `test` based on `speaker_id`.
3.  **Phase 3:** Preprocess & Segment.
4.  **Phase 4:** Synthesize.
    *   **Logic:** Look at `sample['split']`. Pick a voice from the corresponding "Allowed Voice List" (e.g., Train List vs. Test List).
5.  ...Effects, Export...

---

## 2. Configuration (`config.yaml`)
We replace the simple `tts_models` list with strict pools.

```yaml
synthesis:
  # Define pools of voices for each split
  voice_pools:
    train:
      - "edge-tts:en-US-GuyNeural"
      - "edge-tts:en-US-JennyNeural"
      - "bark:v2/en_speaker_6"  # Random pooled speaker
      - "mms:eng"
    val:
      - "edge-tts:en-US-MichelleNeural" # Unseen voice
    test:
      - "edge-tts:en-US-AriaNeural"     # Unseen voice
      - "edge-tts:en-US-ChristopherNeural"
      - "bark:v2/en_speaker_9"
  
  # Generation Strategy
  strategy: "random_from_pool" 
```

---

## 3. Pseudocode: The New Pipeline Loop (`run_pipeline.py`)

```python
def run_pipeline():
    # 1. LOAD DATA
    samples = load_mozilla_cv(...)
    
    # 2. SPLIT IMMEDIATELY (The Key Change)
    # We pass the samples to the splitter NOW, not at the end.
    # This adds a "split": "train"/"test" tag to every metadata dict.
    # Note: 'speaker_id' is from the source (Real), ensuring content disjointness.
    samples = create_speaker_disjoint_splits(samples, config)
    
    # 3. SYNTHESIS LOOP
    for sample in samples:
        target_split = sample['split'] # e.g., 'test'
        
        # A. Select Allowed Voice
        # The manager now needs a valid_voices arg
        valid_voices = config.synthesis.voice_pools[target_split]
        selected_model = random.choice(valid_voices)
        
        # B. Synthesize with SPECIFIC model
        # We override the manager's default random picker
        synth = manager.get_synthesizer(selected_model)
        synthetic_audio = synth.synthesize(text, voice_preset=selected_model)
        
        # C. Save
        sample['generator_voice'] = selected_model
        save(sample)
```

---

## 4. Handling External Datasets (ElevenLabs/RVC)
When we ingest `sleeping-ai/11Labs` or `Audio Rangers RVC`, we treat them as **Pre-Computed Splits**.

```python
def ingest_external_data():
    # ElevenLabs is expensive/commercial -> Hold out for TEST only
    eleven_labs_samples = load_huggingface("sleeping-ai/11Labs")
    for s in eleven_labs_samples:
        s['split'] = 'test' # HARDCODED
        s['generator'] = 'elevenlabs'
    
    # RVC Data -> Split by Model ID
    rvc_samples = load_huggingface("Audio_Rangers")
    # 80% of RVC models to Train, 20% to Test
    train_models, test_models = split_models(rvc_samples.unique_models)
    
    for s in rvc_samples:
        if s.model_id in train_models:
            s['split'] = 'train'
        else:
            s['split'] = 'test'
            
    # Merge into main dataset
    full_dataset = common_voice_samples + eleven_labs_samples + rvc_samples
```

## 5. Summary of Benefits
1.  **Zero Leakage:** The test set contains voices (Aria, Christopher, ElevenLabs) that the model **never heard** during training.
2.  **True Generalization:** Proven ability to detect "Synthetic Artifacts" rather than "GuyNeural's Voice".
3.  **Paper-Ready:** This methodology withstands the strictest scrutiny (ICML/NeurIPS level rigor).
