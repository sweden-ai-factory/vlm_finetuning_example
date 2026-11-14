# tools/prefetch.py
"""
Download everything the training job needs, so the job itself can run offline.
Run this ONCE on a node with internet access (the login node on Leonardo).

  - model   -> HF cache ($HF_HOME, must be the same path as in the slurm script!)
  - dataset -> data/vqa_rad (train / validation / test), saved as Arrow files

The dataset is saved with save_to_disk: a handful of Arrow files with the
images inside, instead of thousands of small image files on the parallel
filesystem.
"""

import argparse
import os

from datasets import DatasetDict, load_dataset
from huggingface_hub import snapshot_download


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_name", type=str, default="Qwen/Qwen2-VL-2B-Instruct")
    ap.add_argument("--dataset", type=str, default="flaviagiammarino/vqa-rad")
    ap.add_argument("--out_dir", type=str, default="data/vqa_rad")
    ap.add_argument("--val_frac", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--skip_model", action="store_true")
    args = ap.parse_args()

    print("HF_HOME =", os.environ.get("HF_HOME", "(not set, using ~/.cache/huggingface)"))

    if not args.skip_model:
        path = snapshot_download(args.model_name)
        print("Model downloaded to", path)

    if os.path.isdir(args.out_dir):
        print(args.out_dir, "already exists, not downloading the dataset again")
        return

    # VQA-RAD comes with train (1793) and test (451). We cut a validation set
    # out of train. Note: the split is over questions, the same image can show
    # up in several splits (that is also the case for the original train/test).
    ds = load_dataset(args.dataset)
    split = ds["train"].train_test_split(test_size=args.val_frac, seed=args.seed)
    out = DatasetDict(
        {"train": split["train"], "validation": split["test"], "test": ds["test"]}
    )
    out.save_to_disk(args.out_dir)
    print("Wrote", args.out_dir, {k: len(v) for k, v in out.items()})


if __name__ == "__main__":
    main()
