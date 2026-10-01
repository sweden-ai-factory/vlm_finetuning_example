# src/train.py
import argparse


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--model_name', type=str, default='Qwen/Qwen2.5-1.5B-Instruct')
    p.add_argument('--train_file', type=str, default='data/train.jsonl')
    p.add_argument('--val_file', type=str, default='data/val.jsonl')
    p.add_argument('--output_dir', type=str, default='outputs/qwen15b_lora_dolly')
    p.add_argument('--max_seq_len', type=int, default=512)
    p.add_argument('--per_device_batch_size', type=int, default=6)
    p.add_argument('--num_workers', type=int, default=8)
    p.add_argument('--epochs', type=int, default=1)
    p.add_argument('--learning_rate', type=float, default=2e-4)
    p.add_argument('--weight_decay', type=float, default=0.01)
    p.add_argument('--warmup_ratio', type=float, default=0.05)
    p.add_argument('--grad_accum_steps', type=int, default=1)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--lora_r', type=int, default=16)
    p.add_argument('--lora_alpha', type=int, default=32)
    p.add_argument('--lora_dropout', type=float, default=0.05)
    p.add_argument('--bf16', action='store_true')
    p.add_argument('--max_train_samples', type=int, default=None)
    p.add_argument('--max_eval_samples', type=int, default=None)
    args=p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    set_seed(args.seed)

if __name__=='__main__':
    main()
