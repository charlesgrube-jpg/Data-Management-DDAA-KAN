#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --job-name=robust_eval
#SBATCH --output=robustness_%j.out
#SBATCH --error=robustness_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=4:00:00

module load miniconda
conda activate ddaa

echo "------------------------------------------------"
echo "Starting Robustness Evaluation (CleverHans Port)"
echo "------------------------------------------------"
echo "Date: $(date)"
echo "Host: $(hostname)"

# Ensure python path includes project root
export PYTHONPATH=$PYTHONPATH:$(pwd)

# Example: Run on PINN (since KAN/ODE are still training/finishing)
# Replace --checkpoint with actual best checkpoint path when ready
# For now, we test the script integrity or run on previous model

# python scripts/evaluate_robustness.py \
#     --model pinn \
#     --checkpoint models_pinn/pinn_best.pt \
#     --epsilon 0.005 \
#     --num_samples 50

# Test run on KAN (if checkpoint exists)
if [ -f "models_kan/kan_best.pt" ]; then
    echo "Evaluating KAN..."
    python scripts/evaluate_robustness.py \
        --model kan \
        --checkpoint models_kan/kan_best.pt \
        --epsilon 0.005 \
        --num_samples 50
else
    echo "KAN checkpoint not found yet. Searching..."
fi

echo "Done"
