#!/bin/bash
#SBATCH --job-name=ddaa_ode_v8
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH --output=/home/ms4726/project/Data-Management-DDAA-KAN/logs/ode_v8_%j.out

echo "ODE v8 | Job: $SLURM_JOB_ID | Node: $(hostname) | Start: $(date)"
echo "Strategy: load stable v2 Phase-1 checkpoint, run Phase-2 with odeint_adjoint"

module load miniconda
source /vast/palmer/apps/avx2/software/miniconda/24.7.1/etc/profile.d/conda.sh
conda activate ddaa

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH="/home/ms4726/project/Data-Management-DDAA-KAN/scripts:/home/ms4726/project/Data-Management-DDAA-KAN:${PYTHONPATH:-}"

cd /gpfs/gibbs/project/lawrence_wilen/ms4726/Data-Management-DDAA-KAN

# --resume_p2: load Phase-1 checkpoint (v2 stable weights), skip Phase 1,
#              unfreeze encoder and train Phase 2 with odeint_adjoint.
# v2 Phase-1 checkpoint (ode_best.pt) was trained with odeint/RK4 but the
# weights are compatible — gradient method does not affect model state_dict.

python scripts/train_ode.py \
    --epochs 20 \
    --batch_size 24 \
    --freeze_epochs 5 \
    --lr 5e-5 \
    --lr_encoder 1e-7 \
    --physics_weight 0.05 \
    --physics_margin 0.01 \
    --resume_p2 "Best Models/ode_best.pt" \
    --output_dir models_ode_v8

echo "Done: $(date)"
