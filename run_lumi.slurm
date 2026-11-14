#!/bin/bash
#SBATCH -J qwen2vl_lora_vqarad_lumi
#SBATCH -A <insert_allocation>
#SBATCH -p dev-g                 # development partition: short queue time. Use small-g / standard-g for long jobs
#SBATCH -N 1
#SBATCH -n 1                     # ONE task, accelerate starts one process per GPU itself
#SBATCH --gpus=8                 # a LUMI-G node has 4 x MI250X = 8 GPUs as seen by slurm (64 GB each)
#SBATCH --cpus-per-task=56       # ... and 56 usable cores, so 7 per GPU
#SBATCH --mem=128G
#SBATCH -t 00:30:00
#SBATCH -o logs/%x_%j.out

# Submit from the repository folder: sbatch run_lumi.slurm

# The model to fine-tune. Any image-text-to-text model on the HF hub should work,
# e.g. Qwen/Qwen2.5-VL-3B-Instruct or HuggingFaceTB/SmolVLM2-2.2B-Instruct (see README).
MODEL=Qwen/Qwen2-VL-2B-Instruct

# LUMI has AMD GPUs. We use the PyTorch container from the LUMI AI Factory,
# everything python related runs through "singularity run $SIF".
module purge
module use /appl/local/laifs/modules
module load lumi-aif-singularity-bindings
export SIF=/appl/local/laifs/containers/lumi-multitorch-latest.sif

# LUMI has no $SCRATCH variable: clone the repo under /scratch/<project> or
# /project/<project> (NOT in $HOME, 20 GB) so that the cache ends up there.
export HF_HOME=${SCRATCH:-$PWD}/.cache/huggingface
export HF_DATASETS_CACHE=$HF_HOME
export HF_HUB_CACHE=$HF_HOME

export TOKENIZERS_PARALLELISM=false
export NCCL_DEBUG=WARN

# AMD specific: MIOpen (used for the convolutions in the vision encoder) wants
# a writable cache, by default it tries in $HOME which is slow / may fail.
export MIOPEN_CUSTOM_CACHE_DIR="/tmp/miopen-cache-${USER}"
export MIOPEN_USER_DB_PATH="/tmp/miopen-config-${USER}"
srun mkdir -p "$MIOPEN_CUSTOM_CACHE_DIR" "$MIOPEN_USER_DB_PATH"

singularity run $SIF python tools/check_env.py

# LUMI compute nodes have internet, so this could be done here. We still do the
# prefetch as a separate step so the workflow is the same as on Leonardo.
# It is a no-op if model and data are already there.
singularity run $SIF python tools/prefetch.py --model_name ${MODEL}

# One process per GPU. Each process has its own DataLoader workers, so share the
# cores between them and leave one per process for the main thread.
NUM_GPUS=${SLURM_GPUS}
WORKERS=$(( SLURM_CPUS_PER_TASK / NUM_GPUS - 1 ))

# 8 GPUs here instead of 4, so batch size 2 per GPU to keep the same effective
# batch size (2 x 8 = 16) as on the other clusters.
singularity run $SIF accelerate launch --multi_gpu --num_processes ${NUM_GPUS} --num_machines 1 \
    src/train.py \
    --model_name ${MODEL} \
    --bf16 \
    --output_dir outputs/$(basename ${MODEL})_lora_vqarad_lumi \
    --per_device_batch_size 2 \
    --epochs 6 \
    --num_workers ${WORKERS}
