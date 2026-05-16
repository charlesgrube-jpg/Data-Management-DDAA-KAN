#!/bin/bash
#SBATCH --job-name=cleverhans_only
#SBATCH --partition=gpu
#SBATCH --output=/gpfs/gibbs/project/lawrence_wilen/ms4726/Data-Management-DDAA-KAN/cleverhans_only_%j.out
#SBATCH --error=/gpfs/gibbs/project/lawrence_wilen/ms4726/Data-Management-DDAA-KAN/cleverhans_only_%j.err
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

# Accept EPSILON as an argument (default 0.005)
EPS=${1:-0.005}

echo "========================================================"
echo "CLEVERHANS-ONLY EVALUATION — ALL 4 MODELS (EPS=$EPS)"
echo "Date: $(date)"
echo "Host: $(hostname)"
echo "========================================================"

run_cleverhans() {
    local MODEL=$1
    local CKPT=$2
    local OUTDIR=$3

    echo ""
    echo "========================================================"
    echo "MODEL: $MODEL  |  CHECKPOINT: $CKPT  |  EPS: $EPS"
    echo "========================================================"

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

run_cleverhans "pinn"        "$CKPT_DIR/pinn_best.pt"           "pinn"
run_cleverhans "kan"         "$CKPT_DIR/kan_best.pt"            "kan"
run_cleverhans "neural_ode"  "$CKPT_DIR/ode_best.pt"            "ode"
run_cleverhans "transformer" "$CKPT_DIR/transformer_hp_best.pt" "transformer"

echo ""
echo "========================================================"
echo "ALL CLEVERHANS EVALUATIONS COMPLETE"
echo "Date: $(date)"
echo "Results in: $OUT_DIR/"
echo "========================================================"
