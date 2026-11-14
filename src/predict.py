# src/predict.py
"""
Run the model on a split of VQA-RAD (default: test) on a single GPU and report
exact-match accuracy. Without --adapter_dir you get the base model, with it the
fine-tuned one.

  python src/predict.py                                  # base model
  python src/predict.py --adapter_dir outputs/qwen2vl2b_lora_vqarad
"""
import argparse
import json
import os

import torch
from datasets import load_from_disk
from peft import PeftModel
from torch.utils.data import DataLoader
from transformers import AutoModelForImageTextToText, AutoProcessor

from train import Collator, normalize


@torch.no_grad()
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model_name", type=str, default="Qwen/Qwen2-VL-2B-Instruct")
    p.add_argument("--adapter_dir", type=str, default=None)
    p.add_argument("--data_dir", type=str, default="data/vqa_rad")
    p.add_argument("--split", type=str, default="test")
    p.add_argument("--max_pixels", type=int, default=448 * 448)
    p.add_argument("--min_pixels", type=int, default=224 * 224)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--max_new_tokens", type=int, default=16)
    p.add_argument("--max_samples", type=int, default=None)
    p.add_argument("--out_file", type=str, default=None)
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    # min/max_pixels only exist for Qwen-VL models, other processors have their own settings
    proc_kwargs = {}
    if "qwen" in args.model_name.lower():
        proc_kwargs = dict(min_pixels=args.min_pixels, max_pixels=args.max_pixels)
    processor = AutoProcessor.from_pretrained(args.model_name, **proc_kwargs)
    model = AutoModelForImageTextToText.from_pretrained(args.model_name, torch_dtype=torch.bfloat16)
    if args.adapter_dir:
        model = PeftModel.from_pretrained(model, args.adapter_dir)
        model = model.merge_and_unload()  # fold LoRA into the weights, faster inference
    model.to(device).eval()

    ds = load_from_disk(args.data_dir)[args.split]
    if args.max_samples:
        ds = ds.select(range(min(args.max_samples, len(ds))))
    ds = ds.add_column("idx", list(range(len(ds))))
    answers, questions = ds["answer"], ds["question"]
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        collate_fn=Collator(processor, train=False),
        num_workers=args.num_workers,
    )

    rows = []
    for batch in loader:
        idx = batch.pop("idx")
        # images to bf16 like the model (not every model casts them itself)
        batch = {k: v.to(device, torch.bfloat16) if v.is_floating_point() else v.to(device) for k, v in batch.items()}
        gen = model.generate(
            **batch, max_new_tokens=args.max_new_tokens, do_sample=False,
            temperature=None, top_p=None, top_k=None,
        )
        preds = processor.batch_decode(gen[:, batch["input_ids"].shape[1] :], skip_special_tokens=True)
        for i, pred in zip(idx.tolist(), preds):
            rows.append(
                {
                    "question": questions[i],
                    "answer": answers[i],
                    "prediction": pred.strip(),
                    "correct": normalize(pred) == normalize(answers[i]),
                }
            )

    yes_no = [r for r in rows if normalize(r["answer"]) in ("yes", "no")]
    open_q = [r for r in rows if normalize(r["answer"]) not in ("yes", "no")]

    def acc(rs):
        return sum(r["correct"] for r in rs) / max(1, len(rs))

    name = args.adapter_dir or "base model"
    print(
        f"[{name}] {args.split}: exact match all={acc(rows):.3f} (n={len(rows)}) "
        f"yes/no={acc(yes_no):.3f} (n={len(yes_no)}) open={acc(open_q):.3f} (n={len(open_q)})"
    )
    for r in rows[:5]:
        print(f"    Q: {r['question']} | truth: {r['answer']} | model: {r['prediction']}")

    if args.out_file:
        os.makedirs(os.path.dirname(args.out_file) or ".", exist_ok=True)
        with open(args.out_file, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print("Wrote", args.out_file)


if __name__ == "__main__":
    main()
