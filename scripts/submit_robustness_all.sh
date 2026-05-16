#!/bin/bash
#SBATCH --job-name=robustness_all
#SBATCH --partition=gpu
#SBATCH --output=/gpfs/gibbs/project/lawrence_wilen/ms4726/Data-Management-DDAA-KAN/robustness_all_%j.out
#SBATCH --error=/gpfs/gibbs/project/lawrence_wilen/ms4726/Data-Management-DDAA-KAN/robustness_all_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00   # 48 hours

module load miniconda
source ~/.bashrc
conda activate ddaa

PROJECT_DIR="/gpfs/gibbs/project/lawrence_wilen/ms4726/Data-Management-DDAA-KAN"
cd $PROJECT_DIR
export PYTHONPATH=$PROJECT_DIR:$PYTHONPATH

CKPT_DIR="Best Models"
DATA_DIR="unified_dataset"
OUT_DIR="robustness_results"

# Accept EPSILON as an argument to allow parallel testing of different attack strengths
EPS=${1:-0.005}

echo "========================================================"
echo "ADVERSARIAL ROBUSTNESS EVALUATION — ALL 4 MODELS (EPS=$EPS)"
echo "Date: $(date)"
echo "Host: $(hostname)"
echo "========================================================"

# ─────────────────────────────────────────────────────────────
# Helper: run all 3 attacks for one model checkpoint
# Usage: run_attacks <model_name> <checkpoint_path> <out_subdir>
# ─────────────────────────────────────────────────────────────
run_attacks() {
    local MODEL=$1
    local CKPT=$2
    local OUTDIR=$3

    echo ""
    echo "========================================================"
    echo "MODEL: $MODEL"
    echo "CHECKPOINT: $CKPT"
    echo "========================================================"

    # ── FGSM: full test set ────────────────────────────────
    echo "--- [$(date)] FGSM (full test set) ---"
    python scripts/evaluate_robustness.py \
        --model $MODEL \
        --checkpoint "$CKPT" \
        --attack fgsm \
        --data_dir $DATA_DIR \
        --epsilon $EPS \
        --output_dir "$OUT_DIR/$OUTDIR"
    echo "--- FGSM done ---"

    # ── PGD: 10000 samples ────────────────────────────────
    echo "--- [$(date)] PGD-40 (10000 samples) ---"
    python scripts/evaluate_robustness.py \
        --model $MODEL \
        --checkpoint "$CKPT" \
        --attack pgd \
        --data_dir $DATA_DIR \
        --epsilon $EPS \
        --pgd_steps 40 \
        --pgd_alpha 0.001 \
        --num_samples 10000 \
        --output_dir "$OUT_DIR/$OUTDIR"
    echo "--- PGD done ---"

    # ── CleverHans: 1000 samples, 50+100 iters ───────────
    echo "--- [$(date)] CleverHans (1000 samples, 50+100 iters) ---"
    python scripts/evaluate_robustness.py \
        --model $MODEL \
        --checkpoint "$CKPT" \
        --attack cleverhans \
        --data_dir $DATA_DIR \
        --epsilon $EPS \
        --num_samples 1000 \
        --num_iter_stage1 50 \
        --num_iter_stage2 100 \
        --output_dir "$OUT_DIR/$OUTDIR"
    echo "--- CleverHans done ---"
}

# ─────────────────────────────────────────────────────────────
# Run for each of the 4 best models
# ─────────────────────────────────────────────────────────────

run_attacks "pinn"        "$CKPT_DIR/pinn_best.pt"           "pinn"
run_attacks "kan"         "$CKPT_DIR/kan_best.pt"            "kan"
run_attacks "neural_ode"  "$CKPT_DIR/ode_best.pt"            "ode"
run_attacks "transformer" "$CKPT_DIR/transformer_hp_best.pt" "transformer"

echo ""
echo "========================================================"
echo "ALL EVALUATIONS COMPLETE"
echo "Date: $(date)"
echo "Results in: $OUT_DIR/"
echo "========================================================"
