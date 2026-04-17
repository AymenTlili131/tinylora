from __future__ import annotations
import os
import json
import argparse
import pandas as pd
import torch

from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig,
)

import wandb

from peft import prepare_model_for_kbit_training
from adapters import AdapterSpec, attach_adapter, count_trainable_params
from data import load_task_dataset
from metrics import reset_cuda_peak, get_cuda_peak_gb, Timer
from evals import generate_batch, eval_math, eval_cls, eval_instruct


def build_lm_dataset(tokenizer, ds: Dataset, max_length: int = 1024):
    def tok(ex):
        text = ex["prompt"] + ex["response"]
        out = tokenizer(
            text,
            truncation=True,
            max_length=max_length,
        )
        out["labels"] = out["input_ids"].copy()
        return out
    return ds.map(tok, remove_columns=ds.column_names)


def quick_eval(task: str, model, tokenizer, eval_ds: Dataset, n: int = 200):
    if n and len(eval_ds) > n:
        eval_ds = eval_ds.select(range(n))

    prompts = [str(x) for x in eval_ds["prompt"]]
    targets = [str(x) for x in eval_ds["response"]]
    generations = generate_batch(model, tokenizer, prompts, max_new_tokens=64, temperature=0.0, batch_size=4)

    task = task.lower()
    if task == "math":
        return eval_math(generations, targets)
    if task == "cls":
        return eval_cls(generations, targets)
    if task == "instruct":
        return eval_instruct(generations, targets)
    raise ValueError(task)


def load_model_and_tokenizer(model_name: str, load_in_4bit: bool, load_in_8bit: bool):
    tok = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    quant_config = None
    if load_in_4bit:
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
    elif load_in_8bit:
        quant_config = BitsAndBytesConfig(load_in_8bit=True)

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto" if torch.cuda.is_available() else None,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        quantization_config=quant_config,
    )
    return model, tok


def run_one(task: str, args, results_rows: list):
    # Transformers 5.x blocks training on *purely quantized* models unless it detects PEFT adapters.
    # Our TinyLoRA demo is custom (not a PeftModel), so we disallow 4bit/8bit for that method.
    if args.method == "tinylora" and (args.load_in_4bit or args.load_in_8bit):
        raise ValueError("TinyLoRA demo method is custom (non-PEFT). Run without --load_in_4bit/--load_in_8bit.")

    model, tok = load_model_and_tokenizer(args.model, args.load_in_4bit, args.load_in_8bit)

    # ----- CRITICAL STEP FOR 4bit/8bit TRAINING (PEFT methods) -----
    if args.load_in_4bit or args.load_in_8bit:
        model = prepare_model_for_kbit_training(model)

    spec = AdapterSpec(
        method=args.method,
        r=args.r,
        alpha=args.alpha,
        dropout=args.dropout,
        target_modules=args.target_modules.split(",") if args.target_modules else None,
        total_step=args.max_steps if args.max_steps and args.max_steps > 0 else None,
    )

    model, adapter_info = attach_adapter(model, spec)

    # Fix mixed precision generation crashes on some models
    if hasattr(model, "lm_head"):
        model.lm_head = model.lm_head.to(torch.float16)
    if hasattr(model, "base_model") and hasattr(model.base_model, "lm_head"):
        model.base_model.lm_head = model.base_model.lm_head.to(torch.float16)

    trainable = count_trainable_params(model)

    # Data
    train_raw = load_task_dataset(task, split="train", limit=args.train_limit)
    eval_raw = load_task_dataset(task, split="test" if task in ["math", "cls"] else "train", limit=args.eval_limit)

    train_tok = build_lm_dataset(tok, train_raw, max_length=args.max_length)
    eval_tok = build_lm_dataset(tok, eval_raw, max_length=args.max_length)

    collator = DataCollatorForLanguageModeling(tok, mlm=False)

    # Train
    reset_cuda_peak()
    # TinyLoRA: disable AMP GradScaler to avoid "Attempting to unscale FP16 gradients"
    use_fp16 = torch.cuda.is_available() and (not args.bf16) and (args.method != "tinylora")
    use_bf16 = bool(args.bf16) and (args.method != "tinylora")

    with Timer() as t:
        training_args = TrainingArguments(
            output_dir=os.path.join(args.outdir, f"{task}_{args.method}"),
            per_device_train_batch_size=args.bs,
            per_device_eval_batch_size=args.bs,
            gradient_accumulation_steps=args.grad_accum,
            learning_rate=args.lr,
            num_train_epochs=args.epochs,
            max_steps=args.max_steps if args.max_steps > 0 else None,
            warmup_steps=20,
            logging_steps=10,
            save_steps=0,
            eval_strategy="no",
            report_to=["wandb"] if args.wandb else [],
            fp16=use_fp16,
            bf16=use_bf16,
            optim="adamw_torch",
            lr_scheduler_type="cosine",
            gradient_checkpointing=args.grad_ckpt,
        )
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_tok,
            eval_dataset=eval_tok,
            data_collator=collator,
        )
        trainer.train()

    peak_gb = get_cuda_peak_gb()

    # Tokens/sec approximate
    avg_len = sum(len(x) for x in train_tok["input_ids"]) / max(1, len(train_tok))
    steps = trainer.state.global_step
    toks_processed = steps * args.bs * args.grad_accum * avg_len
    toks_per_sec = toks_processed / max(1e-9, t.seconds)

    # Eval
    model.eval()
    # Force fp16 inference to avoid dtype mismatches during generation on some stacks
    if torch.cuda.is_available():
        model = model.to(torch.float16)

    scores = quick_eval(task, model, tok, eval_raw, n=args.eval_n)

    row = {
        "task": task,
        "method": args.method,
        "model": args.model,
        "trainable_params": int(trainable),
        "peak_vram_gb": float(peak_gb),
        "train_seconds": float(t.seconds),
        "tokens_per_sec_est": float(toks_per_sec),
        "steps": int(steps),
        **{f"adapter_{k}": v for k, v in adapter_info.items()},
        **scores,
    }
    results_rows.append(row)
    print(json.dumps(row, indent=2))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=str, default="Qwen/Qwen2.5-0.5B-Instruct")
    p.add_argument("--method", type=str, default="lora",
                   choices=["lora", "adalora", "vera", "vblora", "tinylora"])
    p.add_argument("--tasks", type=str, default="math,cls,instruct")
    p.add_argument("--outdir", type=str, default="results")

    # Adapter params
    p.add_argument("--r", type=int, default=8)
    p.add_argument("--alpha", type=int, default=16)
    p.add_argument("--dropout", type=float, default=0.05)
    p.add_argument("--target_modules", type=str, default="")  # comma separated override

    # Training params
    p.add_argument("--bs", type=int, default=1)
    p.add_argument("--grad_accum", type=int, default=8)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--epochs", type=float, default=1.0)
    p.add_argument("--max_steps", type=int, default=200)  # set -1 to use epochs
    p.add_argument("--max_length", type=int, default=768)
    p.add_argument("--grad_ckpt", action="store_true")

    # Data sizes
    p.add_argument("--train_limit", type=int, default=2000)
    p.add_argument("--eval_limit", type=int, default=1000)
    p.add_argument("--eval_n", type=int, default=200)

    # Quantization
    p.add_argument("--load_in_4bit", action="store_true")
    p.add_argument("--load_in_8bit", action="store_true")
    p.add_argument("--bf16", action="store_true")

    # Logging
    p.add_argument("--wandb", action="store_true")
    p.add_argument("--wandb_project", type=str, default="tinylora-variants-demo")
    p.add_argument("--run_name", type=str, default="")

    args = p.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    if args.wandb:
        wandb.init(project=args.wandb_project, name=args.run_name or f"{args.method}_{args.model}")

    results_rows = []
    for task in [t.strip() for t in args.tasks.split(",") if t.strip()]:
        run_one(task, args, results_rows)

    df = pd.DataFrame(results_rows)
    csv_path = os.path.join(args.outdir, f"summary_{args.method}.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nSaved: {csv_path}")

    if args.wandb:
        wandb.log({"summary_table": wandb.Table(dataframe=df)})
        wandb.finish()


if __name__ == "__main__":
    main()
