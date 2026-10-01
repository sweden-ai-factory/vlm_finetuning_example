#!/bin/bash
#SBATCH -J finetune_test
#SBATCH -A <insert_allocation>
#SBATCH -p boost_usr_prod
#SBATCH --qos boost_qos_dbg
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --gpus=4
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH -t 00:30:00
#SBATCH -o logs/%x_%j.out

#source .venv/bin/activate
#You might need to change this
module purge
ml profile/deeplrn cineca-ai/4.3.0

export HF_HOME=${SCRATCH:-$PWD}/.cache/huggingface
export TRANSFORMERS_CACHE=$HF_HOME
export HF_DATASETS_CACHE=$HF_HOME
export HF_HUB_CACHE=$HF_HOME
export HF_HUB_OFFLINE=1

# Rely on OMP parallelism
export TOKENIZERS_PARALLELISM=false

export NCCL_DEBUG=WARN

#Prefetch dataset on Leonardo
#python tools/prep_dolly.py --max_train_samples 10000 --max_val_samples 1000

# Example run you can use with HF accelerate
accelerate launch   --multi_gpu   --num_processes ${SLURM_GPUS}   --num_machines 1   src/train.py   --bf16   --output_dir outputs/qwen15b_lora_dolly_leonardo   --per_device_batch_size 2   --epochs 1   --num_workers 8 --max_seq_len 256
