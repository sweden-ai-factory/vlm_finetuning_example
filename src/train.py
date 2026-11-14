# src/train.py
"""
LoRA fine-tuning of a small vision-language model (Qwen2-VL-2B) on VQA-RAD
(visual question answering on radiology images), multi-GPU via HF accelerate.

Expects the model in the HF cache and the dataset in data/vqa_rad
(run tools/prefetch.py on a node with internet first).
"""
import argparse
import json
import math
import os
import random
import string
import time

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from accelerate import Accelerator
from datasets import load_from_disk
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoModelForImageTextToText,
    AutoProcessor,
    get_linear_schedule_with_warmup,
)

# VQA-RAD answers are short ("yes", "no", "right lung", ...), so we ask for that.
PROMPT_SUFFIX = " Answer with a single word or a short phrase."


def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def normalize(s: str) -> str:
    # lower-case, drop punctuation and extra spaces, for exact-match scoring
    s = s.lower().strip()
    s = s.translate(str.maketrans("", "", string.punctuation))
    return " ".join(s.split())


class Collator:
    """
    Turns a list of {image, question, answer} into model inputs.
    The image resizing/patching happens here, i.e. in the DataLoader workers
    on the CPU. This is why --num_workers and --cpus-per-task matter for VLMs.

    train=True : prompt + answer, right padded, labels only on the answer tokens
    train=False: prompt only, left padded (needed for batched generation)
    """

    def __init__(self, processor, train: bool):
        self.processor = processor
        self.train = train

    def __call__(self, batch):
        tok = self.processor.tokenizer
        # one list of images per sample, the format all VLM processors accept
        images = [[ex["image"].convert("RGB")] for ex in batch]
        prompts = []
        for ex in batch:
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": ex["question"] + PROMPT_SUFFIX},
                    ],
                }
            ]
            prompts.append(
                self.processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
            )

        if not self.train:
            tok.padding_side = "left"
            enc = self.processor(
                text=prompts, images=images, padding=True, return_tensors="pt"
            )
            out = dict(enc)
            out["idx"] = torch.tensor([ex["idx"] for ex in batch])
            return out

        tok.padding_side = "right"
        answers = [ex["answer"].strip() + tok.eos_token for ex in batch]
        enc = self.processor(
            text=[p + a for p, a in zip(prompts, answers)],
            images=images,
            padding=True,
            return_tensors="pt",
        )
        # Loss only on the answer: the answer is the last n tokens before padding
        labels = torch.full_like(enc["input_ids"], -100)
        for i, a in enumerate(answers):
            n_ans = len(tok(a, add_special_tokens=False).input_ids)
            end = int(enc["attention_mask"][i].sum())
            labels[i, end - n_ans : end] = enc["input_ids"][i, end - n_ans : end]
        out = dict(enc)
        out["labels"] = labels
        return out


@torch.no_grad()
def evaluate(model, loss_loader, gen_loader, answers, questions, processor, accelerator, max_new_tokens):
    """Validation loss + exact-match accuracy of generated answers, over all GPUs."""
    model.eval()

    # gather_for_metrics collects from all processes and drops the samples that
    # accelerate duplicated to fill the last batch.
    losses = []
    for batch in loss_loader:
        loss = model(**batch).loss
        losses.append(accelerator.gather_for_metrics(loss.repeat(batch["input_ids"].shape[0])))
    val_loss = torch.cat(losses).float().mean().item()

    gen_model = accelerator.unwrap_model(model)
    all_idx, all_correct, examples = [], [], []
    for batch in gen_loader:
        idx = batch.pop("idx")
        with accelerator.autocast():
            gen = gen_model.generate(
                **batch,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=None,
                top_p=None,
                top_k=None,
            )
        new_tokens = gen[:, batch["input_ids"].shape[1] :]
        preds = processor.batch_decode(new_tokens, skip_special_tokens=True)
        correct = torch.tensor(
            [normalize(p) == normalize(answers[int(i)]) for p, i in zip(preds, idx)],
            device=idx.device,
        )
        if len(examples) < 4:
            i = int(idx[0])
            examples.append((questions[i], answers[i], preds[0].strip()))
        idx, correct = accelerator.gather_for_metrics((idx, correct))
        all_idx.append(idx.cpu())
        all_correct.append(correct.cpu())

    all_idx = torch.cat(all_idx)
    all_correct = torch.cat(all_correct).float()
    yes_no = torch.tensor([normalize(answers[int(i)]) in ("yes", "no") for i in all_idx])
    res = {
        "val_loss": val_loss,
        "acc": all_correct.mean().item(),
        "acc_yes_no": all_correct[yes_no].mean().item() if yes_no.any() else float("nan"),
        "acc_open": all_correct[~yes_no].mean().item() if (~yes_no).any() else float("nan"),
        "n": len(all_idx),
    }
    model.train()
    return res, examples


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model_name", type=str, default="Qwen/Qwen2-VL-2B-Instruct")
    p.add_argument("--data_dir", type=str, default="data/vqa_rad")
    p.add_argument("--output_dir", type=str, default="outputs/qwen2vl2b_lora_vqarad")
    # Image resolution cap (Qwen-VL models only). Qwen2-VL turns every 28x28 pixels
    # into one token, so 448*448 pixels -> 256 image tokens. Lower it if you run out of GPU memory.
    p.add_argument("--max_pixels", type=int, default=448 * 448)
    p.add_argument("--min_pixels", type=int, default=224 * 224)
    p.add_argument("--per_device_batch_size", type=int, default=4)
    p.add_argument("--eval_batch_size", type=int, default=16)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--epochs", type=int, default=6)
    p.add_argument("--learning_rate", type=float, default=2e-4)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--warmup_ratio", type=float, default=0.05)
    p.add_argument("--grad_accum_steps", type=int, default=1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--lora_r", type=int, default=16)
    p.add_argument("--lora_alpha", type=int, default=32)
    p.add_argument("--lora_dropout", type=float, default=0.05)
    # Names of the layers that get LoRA adapters: attention + MLP of the language model.
    # In Qwen2-VL these names do not exist in the vision encoder, so that one stays frozen.
    # Other models may use other names: print(model) shows them.
    p.add_argument(
        "--lora_targets",
        type=str,
        default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
    )
    p.add_argument("--bf16", action="store_true")
    p.add_argument("--max_new_tokens", type=int, default=16)
    p.add_argument("--log_every", type=int, default=10)
    p.add_argument("--max_train_samples", type=int, default=None)
    p.add_argument("--max_eval_samples", type=int, default=None)
    args = p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    set_seed(args.seed)
    accelerator = Accelerator(
        mixed_precision="bf16" if args.bf16 else "no",
        gradient_accumulation_steps=args.grad_accum_steps,
    )
    is_main = accelerator.is_main_process
    if not is_main:
        # only one process should talk
        import transformers

        transformers.utils.logging.set_verbosity_error()

    torch.backends.cuda.matmul.allow_tf32 = True

    if is_main:
        print(f"Processes (GPUs): {accelerator.num_processes} | device: {accelerator.device}")
        print(f"Args: {vars(args)}")

    # ---------------- model ----------------
    # min/max_pixels only exist for Qwen-VL models, other processors have their own settings
    proc_kwargs = {}
    if "qwen" in args.model_name.lower():
        proc_kwargs = dict(min_pixels=args.min_pixels, max_pixels=args.max_pixels)
    processor = AutoProcessor.from_pretrained(args.model_name, **proc_kwargs)
    dtype = torch.bfloat16 if args.bf16 else torch.float32
    model = AutoModelForImageTextToText.from_pretrained(args.model_name, torch_dtype=dtype)

    lora = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        task_type="CAUSAL_LM",
        target_modules=args.lora_targets.split(","),
    )
    model = get_peft_model(model, lora)
    if is_main:
        model.print_trainable_parameters()

    # ---------------- data ----------------
    dataset = load_from_disk(args.data_dir)
    train_ds, val_ds = dataset["train"], dataset["validation"]
    if args.max_train_samples:
        train_ds = train_ds.select(range(min(args.max_train_samples, len(train_ds))))
    if args.max_eval_samples:
        val_ds = val_ds.select(range(min(args.max_eval_samples, len(val_ds))))
    val_ds = val_ds.add_column("idx", list(range(len(val_ds))))
    val_answers, val_questions = val_ds["answer"], val_ds["question"]
    if is_main:
        print(f"Train samples: {len(train_ds)} | val samples: {len(val_ds)}")

    train_loader = DataLoader(
        train_ds,
        batch_size=args.per_device_batch_size,
        shuffle=True,
        collate_fn=Collator(processor, train=True),
        num_workers=args.num_workers,
        pin_memory=True,
    )
    val_loss_loader = DataLoader(
        val_ds,
        batch_size=args.eval_batch_size,
        shuffle=False,
        collate_fn=Collator(processor, train=True),
        num_workers=args.num_workers,
    )
    val_gen_loader = DataLoader(
        val_ds,
        batch_size=args.eval_batch_size,
        shuffle=False,
        collate_fn=Collator(processor, train=False),
        num_workers=args.num_workers,
    )

    # ---------------- optimizer ----------------
    optimizer = AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    # len(train_loader) is still the single-process length here. After prepare()
    # each process sees 1/num_processes of the batches, and the prepared scheduler
    # steps num_processes times per optimizer step, so this stays consistent.
    total_steps = (len(train_loader) * args.epochs) // args.grad_accum_steps
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * args.warmup_ratio),
        num_training_steps=total_steps,
    )

    model, optimizer, train_loader, val_loss_loader, val_gen_loader, scheduler = accelerator.prepare(
        model, optimizer, train_loader, val_loss_loader, val_gen_loader, scheduler
    )
    eff_bs = args.per_device_batch_size * accelerator.num_processes * args.grad_accum_steps
    if is_main:
        print(f"Steps per epoch: {len(train_loader)} | effective batch size: {eff_bs}")

    history = []

    def run_eval(tag):
        t0 = time.time()
        res, examples = evaluate(
            model, val_loss_loader, val_gen_loader, val_answers, val_questions,
            processor, accelerator, args.max_new_tokens,
        )
        if is_main:
            ppl = math.exp(res["val_loss"]) if res["val_loss"] < 30 else float("inf")
            print(
                f"[{tag}] val_loss={res['val_loss']:.4f} ppl={ppl:.2f} | exact match: "
                f"all={res['acc']:.3f} yes/no={res['acc_yes_no']:.3f} open={res['acc_open']:.3f} "
                f"(n={res['n']}, {time.time() - t0:.0f}s)"
            )
            for q, a, pred in examples:
                print(f"    Q: {q} | truth: {a} | model: {pred}")
            history.append({"stage": tag, **res})

    # ---------------- train ----------------
    run_eval("before training")
    t_start = time.time()
    for epoch in range(args.epochs):
        model.train()
        t_epoch = time.time()
        t_data, t_last = 0.0, time.time()
        for step, batch in enumerate(train_loader):
            t_data += time.time() - t_last  # time spent waiting for the DataLoader
            with accelerator.accumulate(model):
                loss = model(**batch).loss
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
            if is_main and step % args.log_every == 0:
                seen = (step + 1) * args.per_device_batch_size * accelerator.num_processes
                print(
                    f"Epoch {epoch} Step {step}/{len(train_loader)} | loss {loss.item():.4f} "
                    f"| lr {scheduler.get_last_lr()[0]:.2e} "
                    f"| {seen / (time.time() - t_epoch):.1f} samples/s "
                    f"| waiting for data {100 * t_data / (time.time() - t_epoch):.0f}%"
                )
            t_last = time.time()
        if is_main:
            print(f"Epoch {epoch} training took {time.time() - t_epoch:.0f}s")
        run_eval(f"epoch {epoch}")

        # Save after every epoch: if the job hits the time limit you keep the last one.
        accelerator.wait_for_everyone()
        if is_main:
            accelerator.unwrap_model(model).save_pretrained(args.output_dir, save_embedding_layers=False)
            processor.save_pretrained(args.output_dir)
            with open(os.path.join(args.output_dir, "history.json"), "w") as f:
                json.dump(history, f, indent=2)

    if is_main:
        print(f"Total training time: {(time.time() - t_start) / 60:.1f} min")
        if torch.cuda.is_available():
            print(f"Peak GPU memory (GPU 0): {torch.cuda.max_memory_allocated() / 1e9:.1f} GB")
        print("Training finished. LoRA adapter saved to:", args.output_dir)


if __name__ == "__main__":
    main()
