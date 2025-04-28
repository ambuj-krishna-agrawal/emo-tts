#!/bin/bash
#SBATCH --gres=gpu:A6000:1
#SBATCH --job-name=tts2
#SBATCH --output=/home/ambuja/logs/output/tts2.out
#SBATCH --error=/home/ambuja/logs/error/tts2.err
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

python -m src_new.multidialogue_emotion_evaluation --no-versa

# DEFAULT_MODEL = "llama_3_70b_q4"
# DEFAULT_MODEL = "llama_3_3b_q4", mistral_7b_q4

# DEFAULT_MODEL = "mistral_7b_q4"

# # Base directories
# DATA_DIR  = Path("/data/group_data/starlight/gpa/tts")
# JSONL_DIR = DATA_DIR / f"multidialog_emotion_planning/{DEFAULT_MODEL}"
# OUT_ROOT  = DATA_DIR / "multidialogue_coqui_out"

# # Coqui TTS detail
# COQUI_MODEL = "tts_models/en/vctk/vits"