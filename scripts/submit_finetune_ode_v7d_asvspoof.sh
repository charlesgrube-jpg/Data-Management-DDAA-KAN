#!/bin/bash
#SBATCH --job-name=ode_v7d_asv_ft
#SBATCH --partition=gpu_h200
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --account=pi_ev6
#SBATCH --output=/home/ms4726/project_pi_ev6/ms4726/Data-Management-DDAA-KAN/logs/ode_v7d_asvspoof_ft_%j.out

echo "NeuralODE v7d ASVspoof Fine-tuning | Job: $SLURM_JOB_ID | Node: $(hostname) | Start: $(date)"
echo "2-phase: Phase 1 (epochs 1-5) head only, Phase 2 (epochs 6-10) unfreeze top 4 encoder layers"
echo "CRITICAL: Uses ODEDetector from train_ode_v7d.py (NOT train_ode.py)"

module load miniconda
source /apps/software/system/software/miniconda/24.11.3/etc/profile.d/conda.sh
conda activate ddaa

export PYTHONPATH="/home/ms4726/project_pi_ev6/ms4726/Data-Management-DDAA-KAN:/home/ms4726/project_pi_ev6/ms4726/Data-Management-DDAA-KAN/scripts:${PYTHONPATH:-}"

cd /home/ms4726/project_pi_ev6/ms4726/Data-Management-DDAA-KAN

ASV_ROOT=asvspoof2019/LA/LA
TRAIN_PROTO=${ASV_ROOT}/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.train.trn.txt
EVAL_PROTO=${ASV_ROOT}/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.eval.trl.txt

python scripts/finetune_ode_v7d_asvspoof.py \
    --asvspoof_root  ${ASV_ROOT} \
    --train_protocol ${TRAIN_PROTO} \
    --eval_protocol  ${EVAL_PROTO} \
    --checkpoint     "models_ode_v7/ode_v7_best.pt" \
    --epochs         10 \
    --unfreeze_epoch 5 \
    --unfreeze_layers 4 \
    --lr             1e-4 \
    --lr_backbone    5e-6 \
    --batch_size     16 \
    --output_dir     asvspoof_finetuned_v2

echo "Done: $(date)"
