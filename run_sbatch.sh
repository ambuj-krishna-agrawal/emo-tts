#!/bin/bash
#SBATCH --job-name=tts3
#SBATCH --output=/home/ambuja/logs/output/tts3.out
#SBATCH --error=/home/ambuja/logs/error/tts3.err
#SBATCH --mem=128GB
#SBATCH --time 1-11:55:00
#SBATCH --partition=general
#SBATCH --mail-type=END
#SBATCH --mail-user=ambuja@andrew.cmu.edu

echo $SLURM_JOB_ID

source ~/.bashrc

conda init bash
source ~/anaconda3/etc/profile.d/conda.sh

conda activate emo-tts-new-py39

python -m src_new.multidialogue_emotion_planning_and_reply_3

# DEFAULT_MODEL = "llama_3_70b_q4"
# DEFAULT_MODEL = "llama_3_3b_q4", mistral_7b_q4
