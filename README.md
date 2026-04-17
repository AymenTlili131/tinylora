# TinyLoRA — LoRA Variant Benchmarking Framework

A lightweight benchmarking framework for comparing **LoRA-family adapters** on causal language models. Fine-tune and evaluate multiple parameter-efficient methods side-by-side on math reasoning, text classification, and instruction-following tasks — all from a single CLI.

## Supported Adapter Methods

| Method | Backend | Description |
|---|---|---|
| `lora` | PEFT | Standard Low-Rank Adaptation |
| `adalora` | PEFT | Adaptive rank allocation with SVD-based importance scoring |
| `vera` | PEFT | Vector-based Random Matrix Adaptation (very few trainable params) |
| `vblora` | PEFT | Vector-Bank LoRA with shared codebook vectors |
| `tinylora` | Custom | Demo adapter — frozen random low-rank factors with a single trainable scalar per layer |

## Evaluation Tasks

| Task key | Dataset | Metric |
|---|---|---|
| `math` | GSM8K | Exact-match accuracy on final numeric answer |
| `cls` | AG News | Classification accuracy (World / Sports / Business / Sci/Tech) |
| `instruct` | Databricks Dolly 15k | ROUGE-L |

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

> **Requirements:** Python 3.9+, a CUDA GPU is strongly recommended. `bitsandbytes` is needed only for 4-bit / 8-bit quantisation.

### 2. Run a single experiment

```bash
python run.py --method lora --tasks math --max_steps 200
```

This will fine-tune `Qwen/Qwen2.5-0.5B-Instruct` (default) with LoRA on GSM8K for 200 steps, evaluate, and save a CSV summary to `results/`.

### 3. Run the full benchmark suite

The included `TXT.txt` is a Bash helper script that sweeps all methods across quantised and fp16 settings:

```bash
bash TXT.txt
```

Results are saved as timestamped CSVs under `results/`.

## CLI Reference

```
python run.py [OPTIONS]
```

### Model & method

| Flag | Default | Description |
|---|---|---|
| `--model` | `Qwen/Qwen2.5-0.5B-Instruct` | HuggingFace model ID or local path |
| `--method` | `lora` | One of `lora`, `adalora`, `vera`, `vblora`, `tinylora` |
| `--tasks` | `math,cls,instruct` | Comma-separated task keys |
| `--outdir` | `results` | Output directory for CSVs and checkpoints |

### Adapter hyperparameters

| Flag | Default | Description |
|---|---|---|
| `--r` | `8` | Rank (used by LoRA / AdaLoRA / TinyLoRA) |
| `--alpha` | `16` | LoRA alpha scaling factor |
| `--dropout` | `0.05` | Adapter dropout |
| `--target_modules` | auto-detected | Comma-separated module suffixes (e.g. `q_proj,v_proj`) |

### Training

| Flag | Default | Description |
|---|---|---|
| `--bs` | `1` | Per-device batch size |
| `--grad_accum` | `8` | Gradient accumulation steps |
| `--lr` | `2e-4` | Learning rate |
| `--epochs` | `1.0` | Number of epochs (overridden when `--max_steps > 0`) |
| `--max_steps` | `200` | Max training steps (`-1` to use epochs instead) |
| `--max_length` | `768` | Max token sequence length |
| `--grad_ckpt` | off | Enable gradient checkpointing (saves VRAM) |

### Quantisation

| Flag | Description |
|---|---|
| `--load_in_4bit` | NF4 quantisation via bitsandbytes (QLoRA-style) |
| `--load_in_8bit` | 8-bit quantisation via bitsandbytes |
| `--bf16` | Use bfloat16 mixed precision instead of fp16 |

> **Note:** `tinylora` is a custom (non-PEFT) method and does not support 4-bit / 8-bit quantisation.

### Data limits

| Flag | Default | Description |
|---|---|---|
| `--train_limit` | `2000` | Max training examples per task |
| `--eval_limit` | `1000` | Max evaluation examples loaded |
| `--eval_n` | `200` | Samples used during generation-based eval |

### Logging

| Flag | Description |
|---|---|
| `--wandb` | Enable Weights & Biases logging |
| `--wandb_project` | W&B project name (default: `tinylora-variants-demo`) |
| `--run_name` | Custom W&B run name |

## Project Structure

```
├── run.py           # Main entry point — training loop, evaluation, CLI
├── adapters.py      # Adapter definitions (PEFT configs + custom TinyLoRA)
├── data.py          # Dataset loading & prompt formatting (GSM8K, AG News, Dolly)
├── evals.py         # Generation utility and task-specific evaluation metrics
├── metrics.py       # VRAM tracking and wall-clock timer
├── requirements.txt # Python dependencies
├── TXT.txt          # Bash script for running the full benchmark sweep
└── viz.ipynb        # Notebook for visualising results
```

## Example Recipes

**LoRA with 4-bit quantisation (QLoRA) on all tasks:**
```bash
python run.py --method lora --load_in_4bit --tasks math,cls,instruct --max_steps 400
```

**AdaLoRA, rank 16, longer training:**
```bash
python run.py --method adalora --r 16 --alpha 32 --max_steps 600 --train_limit 5000
```

**TinyLoRA (custom demo) on classification only:**
```bash
python run.py --method tinylora --tasks cls --max_steps 200
```

**Compare all methods in fp16:**
```bash
for m in lora adalora vera vblora tinylora; do
  python run.py --method $m --tasks math,cls,instruct --max_steps 300
done
```

## Output

Each run appends a JSON summary to stdout and writes `results/summary_<method>.csv` with columns:

- `task`, `method`, `model`
- `trainable_params` — number of trainable parameters
- `peak_vram_gb` — peak GPU memory during training
- `train_seconds`, `tokens_per_sec_est`
- Task-specific scores (`math_exact_match`, `cls_accuracy`, `rougeL`)

## License

This project is provided for research and educational purposes.
