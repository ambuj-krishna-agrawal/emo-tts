#!/bin/bash
#SBATCH --gres=gpu:A6000:1
#SBATCH --job-name=testing1
#SBATCH --output=/home/ambuja/logs/output/testing1.out
#SBATCH --error=/home/ambuja/logs/error/testing1.err
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

python -m src_new.multidialog_emotion_planning_and_reply_2
