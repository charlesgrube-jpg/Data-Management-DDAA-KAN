#!/bin/bash
#SBATCH --job-name=traj_v3
#SBATCH --partition=gpu-shared
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gpus=1
#SBATCH --time=00:30:00
#SBATCH --output=/expanse/lustre/scratch/msahagun/temp_project/logs/traj_v3_%j.out
#SBATCH --error=/expanse/lustre/scratch/msahagun/temp_project/logs/traj_v3_%j.err
#SBATCH --account=csd887

module purge
module load gpu/0.17.3b  cuda/11.8.0

source /home/msahagun/miniconda3/bin/activate
conda activate deepfake_audio

PROJECT=/expanse/lustre/scratch/msahagun/temp_project/Data-Management-DDAA-KAN
cd $PROJECT

# Copy updated ode_interpret.py from worktree
cp .claude/worktrees/bold-shaw/pipeline/interpretability/ode_interpret.py \
   pipeline/interpretability/ode_interpret.py

echo "[SLURM] Starting trajectory_pca v3 regeneration"
echo "[SLURM] Job ID: $SLURM_JOB_ID"
date

python scripts/regen_trajectory_v3.py \
    --ode_ckpt  "Best Models/ode_best.pt" \
    --data_csv  unified_dataset/test.csv \
    --out       papers/figures/trajectory_pca.png \
    --n_samples 40 \
    --n_display 10

echo "[SLURM] Done"
date
