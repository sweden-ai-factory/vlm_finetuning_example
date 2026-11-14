#!/bin/bash -l
#SBATCH -J qwen2vl_lora_vqarad_arrhenius
#SBATCH -A <insert_allocation>   # the GPU allocation, e.g. naissXXXX-X-XXX-gpu
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH -n 1                     # ONE task, accelerate starts one process per GPU itself
#SBATCH --gpus=4                 # an Arrhenius GPU node has 4 x GH200 (~96 GB GPU memory each)
#SBATCH --cpus-per-task=32       # 8 cores per GPU
#SBATCH --mem=96G
#SBATCH -t 00:30:00
#SBATCH -o logs/%x_%j.out

# Submit from the repository folder: sbatch run_arrhenius.slurm

# The model to fine-tune. Any image-text-to-text model on the HF hub should work,
# e.g. Qwen/Qwen2.5-VL-3B-Instruct or HuggingFaceTB/SmolVLM2-2.2B-Instruct (see README).
MODEL=Qwen/Qwen2-VL-2B-Instruct

# The GPU nodes are ARM (aarch64) while the login node is x86_64: a venv made
# on the login node does NOT work here. Create it on a GPU node (see README).
module load GPU/Python/3.13.5-bare-gcc-2025b-eb
source .venv-arrhenius/bin/activate

# Clone the repo in your project storage so the cache ends up there.
export HF_HOME=${SCRATCH:-$PWD}/.cache/huggingface
export HF_DATASETS_CACHE=$HF_HOME
export HF_HUB_CACHE=$HF_HOME

export TOKENIZERS_PARALLELISM=false
export NCCL_DEBUG=WARN

python tools/check_env.py

# Arrhenius compute nodes have internet, so we can fetch model and data here.
# It is a no-op if they are already there.
python tools/prefetch.py --model_name ${MODEL}

# One process per GPU. Each process has its own DataLoader workers, so share the
# cores between them and leave one per process for the main thread.
NUM_GPUS=${SLURM_GPUS}
WORKERS=$(( SLURM_CPUS_PER_TASK / NUM_GPUS - 1 ))

accelerate launch --multi_gpu --num_processes ${NUM_GPUS} --num_machines 1 \
    src/train.py \
    --model_name ${MODEL} \
    --bf16 \
    --output_dir outputs/$(basename ${MODEL})_lora_vqarad_arrhenius \
    --per_device_batch_size 4 \
    --epochs 6 \
    --num_workers ${WORKERS}
