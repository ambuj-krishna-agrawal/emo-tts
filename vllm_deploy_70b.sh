#!/bin/sh
#SBATCH --gres=gpu:A6000:4
#SBATCH --partition=general
#SBATCH --mem=128GB
#SBATCH --time=2-00:00:00
#SBATCH --job-name=2_meta3_70b_awq
#SBATCH --error=/home/ambuja/logs/error/2_meta3_70b_awq.err
#SBATCH --output=/home/ambuja/logs/output/2_meta3_70b_awq.out
#SBATCH --mail-type=END
#SBATCH --mail-user=ambuja@andrew.cmu.edu

mkdir -p /scratch/ambuja/model

source ~/miniconda3/etc/profile.d/conda.sh
conda activate vllm_host
export HF_HOME=/home/ambuja/hf_cache
huggingface-cli login --token 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


export CUDA_VISIBLE_DEVICES=0,1,2,3 # Explicitly set GPUs

# NCCL configuration - modified for improved local communication
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=1
export NCCL_P2P_LEVEL=NVL
export NCCL_SOCKET_IFNAME=lo
export NCCL_P2P_DISABLE=0
export NCCL_SHM_DISABLE=0
export NCCL_SOCKET_NTHREADS=4
export NCCL_NSOCKS_PERTHREAD=4
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=3600  # Increased timeout

# PyTorch settings
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
export OMP_NUM_THREADS=4  # Control OpenMP threading

# Clear GPU cache before starting (if nvidia-smi is available)
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi -r
    sleep 5
fi

python -m vllm.entrypoints.openai.api_server \
    --model ibnzterrell/Meta-Llama-3.3-70B-Instruct-AWQ-INT4 \
    --port 8081 \
    --download-dir /scratch/ambuja/model \
    --quantization awq \
    --dtype float16 \
    --tensor-parallel-size 4 \
    --max-model-len 2048

echo "✅ Meta‑Llama‑3.3‑70B AWQ running on port 8081 (max_model_len=2048)"
