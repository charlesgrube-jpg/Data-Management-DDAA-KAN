#!/bin/bash
#SBATCH --job-name=rawnet2_native_eval
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --account=pi_ev6
#SBATCH --output=/home/ms4726/project_pi_ev6/ms4726/Data-Management-DDAA-KAN/logs/rawnet2_native_eval_%j.out

echo "RawNet2 Native ASVspoof Eval | Job: $SLURM_JOB_ID | Node: $(hostname) | Start: $(date)"

module load miniconda
source /apps/software/system/software/miniconda/24.11.3/etc/profile.d/conda.sh
conda activate ddaa

export PYTHONPATH="/home/ms4726/project_pi_ev6/ms4726/Data-Management-DDAA-KAN:${PYTHONPATH:-}"

cd /home/ms4726/project_pi_ev6/ms4726/Data-Management-DDAA-KAN

python scripts/eval_rawnet2_native_asvspoof.py

echo "Done: $(date)"
