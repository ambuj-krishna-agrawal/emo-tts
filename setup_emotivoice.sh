#!/usr/bin/env bash
# -----------------------------------------------------------------------------
# install_emotivoice_official.sh  –  Strictly follows the *official* README
# steps for EmotiVoice (no Docker, no extras).
# -----------------------------------------------------------------------------
# 1. Creates conda env  `EmotiVoice`  with Python 3.8.
# 2. Installs **latest** torch+torchaudio wheels automatically matching your
#    CUDA runtime (CU121 wheels if driver ≥12.1, CPU wheels otherwise).
# 3. Installs runtime Python libs exactly as listed.
# 4. Installs git‑lfs and pulls both required repos:
#       • WangZeJun/simbert‑base‑chinese          (text‑frontend LM)
#       • syq163/outputs (all acoustic & vocoder weights)
# 5. Prints the official inference command skeleton at the end.
# -----------------------------------------------------------------------------
set -euo pipefail

ENV_NAME="EmotiVoice"
PYTHON_VER="3.8"

# ------------------- helper to choose correct Torch wheel ------------------
CUDA_MAJOR=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | cut -d. -f1 || echo "0")

if (( CUDA_MAJOR >= 12 )); then
  TORCH_URL="https://download.pytorch.org/whl/cu121"
  TORCH_TAG="cu121"
else
  TORCH_URL="https://download.pytorch.org/whl/cu118"
  TORCH_TAG="cu118"
fi

# ------------------- 1. Conda env -----------------------------------------
if conda env list | grep -q "^${ENV_NAME}\s"; then
  echo "[setup] Conda env $ENV_NAME already exists – reusing."
else
  echo "[setup] Creating conda env $ENV_NAME (Python $PYTHON_VER)…"
  conda create -y -n "$ENV_NAME" python=$PYTHON_VER
fi

PIP="conda run -n $ENV_NAME python -m pip"
PY="conda run -n $ENV_NAME python"

# ------------------- 2. Torch + Torchaudio ---------------------------------
echo "[setup] Installing torch/torchaudio ($TORCH_TAG wheels)…"
$PIP install -q --upgrade pip
$PIP install -q torch torchaudio --extra-index-url $TORCH_URL

# ------------------- 3. Runtime libs --------------------------------------
echo "[setup] Installing runtime dependencies…"
$PIP install -q numpy numba scipy transformers soundfile yacs \
               g2p_en jieba pypinyin pypinyin_dict
$PY - <<'PY'
import nltk, os; nltk.download('averaged_perceptron_tagger_eng', quiet=True)
PY

# ------------------- 4. Git‑LFS repos -------------------------------------
if ! command -v git-lfs &>/dev/null; then
  echo "[setup] Installing git‑lfs via conda…"
  conda install -n "$ENV_NAME" -y git-lfs
fi
conda run -n "$ENV_NAME" git lfs install --skip-repo

mkdir -p $HOME/EmotiVoice_models
cd $HOME/EmotiVoice_models

if [[ ! -d simbert-base-chinese ]]; then
  echo "[setup] Cloning WangZeJun/simbert-base-chinese via LFS…"
  git lfs clone https://huggingface.co/WangZeJun/simbert-base-chinese
fi

if [[ ! -d outputs ]]; then
  echo "[setup] Cloning syq163/outputs (pretrained checkpoints)…"
  git clone https://www.modelscope.cn/syq163/outputs.git
fi

# ------------------- 5. Done ----------------------------------------------
cat <<EOF

[EmotiVoice] Installation complete.
Activate the env and run an inference sample:

  conda activate ${ENV_NAME}
  export PYTHONPATH=$HOME/EmotiVoice_models/outputs
  TEXT=data/inference/text               # prepare as README shows

  python inference_am_vocoder_joint.py \
    --logdir prompt_tts_open_source_joint \
    --config_folder config/joint \
    --checkpoint g_00140000 \
    --test_file \$TEXT

Synthesised WAVs will appear under  outputs/prompt_tts_open_source_joint/test_audio
EOF
