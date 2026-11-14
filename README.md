# VLM Finetune (VQA-RAD)

Fine-tune a vision-language model with **LoRA** on **VQA-RAD** (questions about
radiology images) on one GPU node. The default model is
**Qwen/Qwen2-VL-2B-Instruct**, and you can replace it with another VLM. There are SLURM scripts for
Leonardo, LUMI, MeluXina and Arrhenius.

The task is only an example. The point is to show how to run a VLM fine-tuning
job on EuroHPC clusters. It is small on purpose and runs in about 10 minutes.

> The dataset contains medical images. Do not use the model for anything clinical.

## Files

```
tools/prefetch.py      download model and dataset
tools/check_env.py     print versions and GPUs (runs at the start of every job)
src/train.py           LoRA fine-tuning on all GPUs of the node
src/predict.py         test the base or fine-tuned model (1 GPU)
run_<cluster>.slurm    job script per cluster
```

## Quick start

**1.** Clone the repo on project or scratch storage, **not in `$HOME`** (small quota).

**2.** Set up the software for your cluster:

| Cluster | Setup |
|---|---|
| Leonardo | `ml profile/deeplrn cineca-ai/4.3.0`, nothing to install |
| LUMI | container, already set in the script |
| MeluXina | venv, created **on a compute node** (see below) |
| Arrhenius | venv, created **on a GPU node** (see below) |

```bash
# MeluXina
salloc -A <account> -p gpu --qos test -N 1 --gpus=1 -t 00:30:00
module load PyTorch
python -m venv .venv && source .venv/bin/activate
pip install "transformers==4.52.4" accelerate peft datasets pillow

# Arrhenius (GPU nodes are ARM, the login node is x86)
srun -A <allocation> -p gpu --gpus=1 -c 8 -t 00:20:00 --pty bash -l
module load GPU/Python/3.13.5-bare-gcc-2025b-eb
python3 -m venv .venv-arrhenius && source .venv-arrhenius/bin/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install "transformers==4.52.4" accelerate peft datasets pillow
```

**3. Leonardo only:** compute nodes have no internet, so download first on the login node.
The other clusters do this inside the job.

```bash
ml profile/deeplrn cineca-ai/4.3.0
export HF_HOME=${SCRATCH:-$PWD}/.cache/huggingface HF_HUB_CACHE=$HF_HOME HF_DATASETS_CACHE=$HF_HOME
python tools/prefetch.py --model_name Qwen/Qwen2-VL-2B-Instruct
```

**4.** Put your project account in `#SBATCH -A` and submit **from the repo folder**:

```bash
sbatch run_leonardo.slurm      # or run_lumi / run_meluxina / run_arrhenius
squeue --me                    # is it running?
tail -f logs/*_<jobid>.out     # follow the output
```

**5.** Test the result:

```bash
python src/predict.py                                          # base model
python src/predict.py --adapter_dir outputs/<model>_lora_vqarad_<cluster>
```

## Use another model

Change one line at the top of the slurm script:

```bash
MODEL=Qwen/Qwen2.5-VL-3B-Instruct
```

Pass the same `--model_name` to `predict.py` (and to `prefetch.py` on Leonardo).
Tested with `Qwen/Qwen2-VL-2B-Instruct`, `Qwen/Qwen2.5-VL-3B-Instruct` and
`HuggingFaceTB/SmolVLM2-2.2B-Instruct`. Other chat VLMs that load with
`AutoModelForImageTextToText` should also work. If they do not:

- **Out of GPU memory** with a bigger model: lower `--per_device_batch_size`.
- **LoRA finds no layers**: the layer names differ. Run `print(model)` and set
  `--lora_targets`.
- **Missing package** error: some models need an extra one (SmolVLM: `pip install num2words`).
- `--max_pixels` (image size) only applies to Qwen models. SmolVLM cuts every
  image into tiles and needs much more memory: 42 GB per GPU with
  `--per_device_batch_size 2`, compared with 12 GB for Qwen2-VL-2B at batch size 4.

Use your own data: make `tools/prefetch.py` save a dataset with columns
`image`, `question`, `answer` and splits `train`, `validation`, `test`.

## Expected result

Arrhenius, 4 x GH200, about 9 minutes, peak GPU memory 12 GB.
Exact match on the test split (451 questions):

| | all | yes/no | open |
|---|---|---|---|
| base model | 0.452 | 0.657 | 0.195 |
| fine-tuned | 0.508 | 0.729 | 0.230 |

The Leonardo, LUMI and MeluXina scripts have not been run with this example yet.

## Common pitfalls

| Problem | Fix |
|---|---|
| Job hangs, then connection errors (Leonardo) | Prefetch on the login node with the same `HF_HOME` as in the script |
| `Disk quota exceeded` | Repo and cache must be on project/scratch, not `$HOME` |
| No log file appears | Submit from the repo folder, `logs/` must exist |
| `CUDA out of memory` | Lower `--per_device_batch_size` or `--max_pixels` |
| Only one GPU is used | Start with `accelerate launch`, not `python` |
| `waiting for data` is high in the log | Images are decoded on the CPU: more `--cpus-per-task` / `--num_workers` |
| `Exec format error` / import errors | The venv was created on the login node, create it on a compute node |
| Job stays `PD` | Read the reason in `squeue --me` (account, time limit, QOS) |

## Licensing

- Code: MIT, see `LICENSE`.
- Model: Qwen2-VL-2B-Instruct, Apache 2.0. Check the licence if you use another model.
- Dataset: VQA-RAD, CC0 1.0. Lau et al., Scientific Data 5, 180251 (2018),
  via `flaviagiammarino/vqa-rad` on Hugging Face.
