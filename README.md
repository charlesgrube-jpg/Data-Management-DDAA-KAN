# DDAA: Deepfake Audio Detection Under Acoustic Degradation and Adversarial Conditions

DDAA is a large-scale benchmark and dataset pipeline for evaluating deepfake audio detection systems under real-world acoustic conditions. It pairs naturalistic real speech from Mozilla Common Voice with TTS/voice-conversion synthesis (Edge-TTS, RVC), applies codec compression and channel degradation, and enforces speaker-disjoint splits to prevent data leakage.

This repository contains:
- The full dataset generation pipeline
- Four neural architecture baselines (Transformer, KAN, PINN, NeuralODE)
- Training, evaluation, and interpretability scripts
- Adversarial robustness evaluation tools

---

## Repository Structure

```
.
├── run_pipeline.py             # Main entry point: dataset generation
├── run_attack_local.py         # Run adversarial attacks locally
├── requirements.txt            # Core dependencies
│
├── config/                     # Configuration files
│   ├── config.yaml             # Pipeline configuration (sources, synthesis, splits)
│   └── requirements_colab.txt  # Full dependencies (GPU/Linux, includes RVC)
│
├── docs/                       # Documentation and changelogs
│   ├── CHANGELOG.md
│   ├── CHANGELOG_HF.md
│   ├── CONDA_SETUP.md
│   ├── PIPELINE_FLOW.md
│   ├── PIPELINE_SUMMARY.md
│   ├── SUMMARY_HF_INTEGRATION.md
│   └── notebooks/              # Colab/Kaggle notebook versions
│
├── pipeline/                   # Dataset generation pipeline (importable package)
│   ├── config.py               # Config loading (OmegaConf-based)
│   ├── data_gen/               # Data loading, splitting, validation, export
│   │   ├── mozilla_cv_loader.py
│   │   ├── splitter.py         # Speaker-disjoint split assignment
│   │   ├── validator.py
│   │   └── exporter.py
│   ├── synthesizer/            # TTS and voice conversion
│   │   ├── edge_tts_synthesizer.py
│   │   ├── rvc_synthesizer.py
│   │   ├── huggingface_synthesizer.py  # Bark, SpeechT5, MMS-TTS
│   │   └── synthesizer.py      # SynthesizerManager (coordinates all backends)
│   ├── standardizer/           # Audio preprocessing and segmentation
│   │   ├── preprocessor.py
│   │   └── segmenter.py
│   ├── effects/                # Codec compression and channel degradation
│   │   ├── codec_compression.py
│   │   └── effects.py
│   ├── features/               # Feature extraction (CQT, LFCC, SSL)
│   │   ├── cqt_extractor.py
│   │   ├── lfcc_extractor.py
│   │   └── ssl_extractor.py
│   ├── models/                 # Detector architectures (importable)
│   │   ├── base_detector.py
│   │   ├── transformer_detector.py
│   │   ├── kan_detector.py
│   │   ├── pinn_detector.py
│   │   └── neural_ode_detector.py
│   ├── attacks/                # Adversarial attack implementations
│   │   ├── standard_attacks.py     # FGSM, I-FGSM, PGD
│   │   ├── imperceptible_attack.py
│   │   ├── psychoacoustic_masking.py
│   │   └── cleverhans/             # CleverHans-based psychoacoustic attack
│   └── utils/                  # Shared utilities
│
├── scripts/                    # Training, evaluation, and analysis scripts
│   ├── train.py                # Transformer (HighPerformanceDetector)
│   ├── train_kan.py            # KAN detector (B-spline, grid=5 order=5)
│   ├── train_pinn.py           # PINN detector (physics regularization lambda=0.05)
│   ├── train_ode.py            # NeuralODE detector
│   ├── validate_pipeline.py    # Validate pipeline output quality
│   ├── evaluate_robustness.py  # Adversarial robustness evaluation
│   ├── run_interpretability.py # Interpretability analysis (attention, splines, ODE traj)
│   ├── regen_figs_correct.py   # Regenerate paper figures from scores cache
│   ├── regen_bootstrap_ci.py   # Regenerate bootstrap confidence intervals
│   ├── compute_crossarch_auc_cis.py
│   ├── preextract_ssl_features.py  # Pre-extract SSL (wav2vec2) features for fast training
│   ├── bundle_ssl_features.py
│   ├── prepare_training_data.py
│   ├── finetune_ode_v7d_asvspoof.py
│   ├── submit_train.sh         # SLURM: train Transformer
│   ├── submit_kan.sh           # SLURM: train KAN
│   ├── submit_pinn.sh          # SLURM: train PINN
│   ├── submit_ode.sh           # SLURM: train NeuralODE
│   ├── submit_robustness.sh    # SLURM: robustness evaluation
│   └── submit_interpretability.sh
│
├── utilities/                  # Standalone utility scripts (download, setup, testing)
│   ├── download_partial.py     # Download Mozilla Common Voice subset
│   ├── download_rvc_models.py  # Download RVC pretrained models
│   ├── download_models.py
│   ├── test_features.py
│   └── test_hf_synthesis.py
│
├── new_generators/             # Evaluation on unseen TTS systems (Bark, MMS, SpeechT5)
│   └── evaluation/
│
└── papers/                     # LaTeX source and figures for papers
```

---

## Setup

### Requirements

- Python 3.10+
- ffmpeg (for codec compression): `brew install ffmpeg` or `apt install ffmpeg`
- For RVC (voice conversion): Linux or Docker recommended

### Install

```bash
conda create -n ddaa python=3.10 -y
conda activate ddaa
pip install -r requirements.txt
```

For full pipeline including RVC and GPU features (Colab/Linux):

```bash
pip install -r config/requirements_colab.txt
```

### Verify

```bash
python -c "import edge_tts; print('edge-tts OK')"
python -c "import librosa; print('librosa OK')"
python -c "import torchaudio; print('torchaudio OK')"
```

---

## Dataset Generation

### 1. Download Mozilla Common Voice

```bash
python utilities/download_partial.py
```

This downloads a subset of Mozilla Common Voice 24.0 (English) to `mozilla_cv_data/`.

### 2. Configure the pipeline

Edit `config/config.yaml` to set:
- `source.max_samples`: how many audio clips to process
- `synthesis.enable_vc`: enable RVC voice conversion (Linux/Docker only)
- `output.base_dir`: output directory

### 3. Run the pipeline

```bash
python run_pipeline.py
```

This runs all phases in sequence:
1. Load Mozilla Common Voice clips
2. Assign speaker-disjoint splits (train/val/test)
3. Preprocess audio (resample to 16kHz, normalize)
4. Segment into chunks (up to 5 seconds)
5. Synthesize: 50% Edge-TTS, 50% RVC (if enabled)
6. Apply codec compression and channel degradation (both real and synthetic)
7. Validate dataset statistics
8. Export to `processed_dataset_<timestamp>/`
9. Extract CQT and LFCC features

For parallel array job execution (HPC):

```bash
python run_pipeline.py --chunk-id 0 --num-chunks 8
```

### 4. Validate output

```bash
python scripts/validate_pipeline.py
```

---

## Training

All training scripts use a shared wav2vec2-base backbone. Train on the full DDAA dataset (unified_dataset/train.csv) using the scripts in `scripts/`.

### Transformer

```bash
python scripts/train.py \
    --data_dir unified_dataset \
    --epochs 20 \
    --batch_size 24 \
    --output_dir Best\ Models
```

### KAN (Kolmogorov-Arnold Network)

```bash
python scripts/train_kan.py \
    --data_dir unified_dataset \
    --epochs 20 \
    --batch_size 24 \
    --grid_size 5 \
    --spline_order 5 \
    --output_dir Best\ Models
```

### PINN (Physics-Informed Neural Network)

```bash
python scripts/train_pinn.py \
    --data_dir unified_dataset \
    --epochs 20 \
    --batch_size 24 \
    --physics_weight 0.05 \
    --output_dir Best\ Models
```

### NeuralODE

```bash
python scripts/train_ode.py \
    --data_dir unified_dataset \
    --epochs 20 \
    --batch_size 24 \
    --output_dir models_ode_v7
```

### SLURM (Yale HPC)

```bash
sbatch scripts/submit_train.sh       # Transformer
sbatch scripts/submit_kan.sh         # KAN
sbatch scripts/submit_pinn.sh        # PINN
sbatch scripts/submit_ode.sh         # NeuralODE
```

---

## Evaluation

### Pre-extract SSL features (recommended for fast eval)

```bash
python scripts/preextract_ssl_features.py \
    --csv unified_dataset/test.csv \
    --output_dir ssl_features/
```

### Adversarial robustness

```bash
python scripts/evaluate_robustness.py \
    --model pinn \
    --checkpoint "Best Models/pinn_best.pt" \
    --attack pgd \
    --eps 0.01
```

Or on SLURM:

```bash
sbatch scripts/submit_robustness.sh
```

### Interpretability

```bash
python scripts/run_interpretability.py \
    --model pinn \
    --checkpoint "Best Models/pinn_best.pt"
```

---

## Audio Loading (Critical)

All evaluation and training code must use this normalized audio loading pattern:

```python
import torch
import torchaudio
import torchaudio.transforms as T
import torch.nn.functional as F

SR = 16000
MAX_LEN = 5 * SR  # 5 seconds

def load_audio(path):
    wav, sr = torchaudio.load(str(path))
    if wav.shape[0] > 1:
        wav = wav.mean(0, keepdim=True)
    wav = wav.squeeze(0)
    if sr != SR:
        wav = T.Resample(sr, SR)(wav.unsqueeze(0)).squeeze(0)
    if wav.shape[0] > MAX_LEN:
        wav = wav[:MAX_LEN]
    elif wav.shape[0] < MAX_LEN:
        wav = F.pad(wav, (0, MAX_LEN - wav.shape[0]))
    return torch.nan_to_num(wav / (wav.abs().max() + 1e-6))
```

---

## Model Inference Notes

| Model | Output | Notes |
|---|---|---|
| Transformer, KAN, PINN | logits | Pass `attention_mask` to wav2vec2 |
| AASIST | tuple | Use `out[-1]` for logits |
| RawNet2 | log_softmax | Use `torch.exp(out)[:, 1]` for fake probability |
| NeuralODE | `(logits, trajectory)` | Use `out[0]` for logits |

---

## Key Results (DDAA Test Set, 61,803 samples)

| Model | AUC | EER (%) |
|---|---|---|
| PINN | 97.63% | 7.91 |
| KAN | 97.20% | 7.34 |
| AASIST (fine-tuned) | 95.28% | 11.79 |
| NeuralODE v7d | 92.18% | 14.32 |
| Transformer | 88.97% | 22.37 |
| RawNet2 (fine-tuned) | 87.83% | 17.89 |
| AASIST (zero-shot) | 58.44% | 42.51 |
| RawNet2 (zero-shot) | 45.24% | 52.71 |

---

## Citation

If you use DDAA in your research, please cite:

```
[Citation forthcoming — NeurIPS 2026 Datasets & Benchmarks Track submission]
```

---

## License

See LICENSE file. Mozilla Common Voice audio is licensed under CC0.
