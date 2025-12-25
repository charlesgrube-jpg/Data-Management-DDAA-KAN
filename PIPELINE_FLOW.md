# Pipeline Flow: The Journey of an Audio File

This document explains exactly how a single audio file travels through the code, from raw download to final dataset.

---

### Step 1: The Master Plan
**File:** `config.yaml`
**Purpose:** The Rulebook.
*   "Download only 50 files."
*   "Make them 16kHz audio."
*   "Use these specific voices."
*   "Add background noise to 40% of them."

### Step 2: Starting the Engine
**File:** `run_pipeline.py`
**Purpose:** The Conductor.
*   It starts the stopwatch.
*   It reads the Rulebook (`config.yaml`).
*   It tells all the other files when to do their job.

### Step 3: Finding the Audio
**File:** `pipeline/data_gen/mozilla_cv_loader.py`
**Purpose:** The Librarian.
*   It looks at the massive folder of Mozilla files.
*   It reads the spreadsheet (`train.tsv`) to find valid English sentences.
*   It picks the files based on your random seed (reproducibility).

### Step 4: Cleaning (Real Audio)
**File:** `pipeline/standardizer/preprocessor.py`
**Purpose:** The Car Wash.
*   **Resample:** Converts everything to 16,000 Hz (standard quality).
*   **Normalize:** Adjusts volume so it’s not too loud or quiet (RMS Normalization).
*   **Trim:** Cuts off the silent parts at the start and end.

### Step 5: Creation (Synthetic Audio)
**File:** `pipeline/synthesizer/synthesizer.py`
**Purpose:** The Actor.
*   It takes the **text** from the Real Audio.
*   It hires a "Voice Actor" (Edge-TTS or Coqui).
*   It speaks the same sentence in a robot voice.
*   **CRITICAL:** It sends this new audio back to the "Car Wash" (Step 4) so it has the same volume/trimming as the real human.

### Step 6: The "Real World" Treatment
**File:** `pipeline/effects/effects.py`
**Purpose:** The Weather Machine.
*   It takes **both** the Real Human and the Robot Actor.
*   It applies the *same* bad weather to both:
    *   **Mobile:** Cuts high frequencies (phone sound) + adds static.
    *   **Noisy:** Adds cafe noise + echo.
    *   **Clean:** Does nothing.

### Step 7: Compression (Optional)
**File:** `pipeline/effects/codec_compression.py`
**Purpose:** The WhatsApp Simulator.
*   It crunches the audio down like a bad MP3.
*   It creates those "swirly" digital artifacts you hear on low-quality calls.
*   (Enabled in Colab, disabled on your PC).

### Step 8: Validation
**File:** `pipeline/data_gen/validator.py`
**Purpose:** The Quality Inspector.
*   "Is this file empty?"
*   "Did we accidentally put the same speaker in Train and Test?"
*   "Is the dataset balanced?"

### Step 9: Final Export
**File:** `pipeline/data_gen/exporter.py`
**Purpose:** The Shipping Department.
*   Saves the audio as `.wav` files in `processed_dataset/`.
*   Writes the `metadata.csv` shipping manifest.

---

### Visual Summary

```
[Raw Mozilla MP3] 
       ⬇
[Librarian] (Loader)
       ⬇
[Car Wash] (Preprocessor - Normalize/Trim) ➡ [Real Audio Ready]
       ⬇                                            ⬇
[Actor] (Synthesizer) reads Text                    ⬇
       ⬇                                            ⬇
[Car Wash] (Preprocessor - Normalize/Trim) ➡ [Fake Audio Ready]
                                                    ⬇
             [Weather Machine] (Effects) ⬅⬅⬅⬅⬅⬅⬅⬅
             (Applies Noise/Echo to BOTH)
                        ⬇
             [WhatsApp Sim] (Codec)
                        ⬇
             [Quality Inspector] (Validator)
                        ⬇
             [Shipping Dept] (Exporter)
```

---

### Step 10: Feature Extraction
**File:** `pipeline/features/extract_features.py`
**Purpose:** The Translator.
*   Takes the final WAV files.
*   Converts them into **CQT Spectrograms** or **LFCC** features.
*   Saves as `.pt` (PyTorch tensors) ready for model training.
*   Command: `python -m pipeline.features.extract_features --type cqt`
