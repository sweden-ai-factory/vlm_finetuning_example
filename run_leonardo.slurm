#!/bin/bash
#SBATCH -J qwen2vl_lora_vqarad_leonardo
#SBATCH -A <insert_allocation>
#SBATCH -p boost_usr_prod
#SBATCH --qos boost_qos_dbg      # debug QOS: short queue time, max 30 min. Remove for long jobs
#SBATCH -N 1
#SBATCH -n 1                     # ONE task, accelerate starts one process per GPU itself
#SBATCH --gpus=4                 # a Leonardo booster node has 4 x A100 (64 GB)
#SBATCH --cpus-per-task=32       # ... and 32 cores, so 8 per GPU
#SBATCH --mem=120G
#SBATCH -t 00:30:00
#SBATCH -o logs/%x_%j.out

# Submit from the repository folder: sbatch run_leonardo.slurm

# The model to fine-tune. Any image-text-to-text model on the HF hub should work,
# e.g. Qwen/Qwen2.5-VL-3B-Instruct or HuggingFaceTB/SmolVLM2-2.2B-Instruct (see README).
MODEL=Qwen/Qwen2-VL-2B-Instruct

module purge
ml profile/deeplrn cineca-ai/4.3.0
#source .venv/bin/activate       # if you need packages that are not in the module

# Keep the HF cache out of $HOME (small quota). Must be the same path you used
# when you ran tools/prefetch.py on the login node.
export HF_HOME=${SCRATCH:-$PWD}/.cache/huggingface
export HF_DATASETS_CACHE=$HF_HOME
export HF_HUB_CACHE=$HF_HOME

# Leonardo compute nodes have NO internet. Everything must be prefetched from
# the login node (see README). Offline mode makes HF fail fast instead of
# hanging on a connection that never comes.
export HF_HUB_OFFLINE=1

export TOKENIZERS_PARALLELISM=false
export NCCL_DEBUG=WARN

python tools/check_env.py

# One process per GPU. Each process has its own DataLoader workers, so share the
# cores between them and leave one per process for the main thread.
NUM_GPUS=${SLURM_GPUS}
WORKERS=$(( SLURM_CPUS_PER_TASK / NUM_GPUS - 1 ))

accelerate launch --multi_gpu --num_processes ${NUM_GPUS} --num_machines 1 \
    src/train.py \
    --model_name ${MODEL} \
    --bf16 \
    --output_dir outputs/$(basename ${MODEL})_lora_vqarad_leonardo \
    --per_device_batch_size 4 \
    --epochs 6 \
    --num_workers ${WORKERS}
