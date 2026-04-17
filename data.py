# data.py
from __future__ import annotations
from typing import Dict, List
import re

from datasets import load_dataset

def format_gsm8k(example: Dict) -> Dict:
    q = example["question"].strip()
    a = example["answer"].strip()
    # GSM8K answers often have "#### 42" as the final.
    prompt = (
        "You are a helpful math tutor.\n"
        "Solve the problem step by step, then give the final answer on the last line as:\n"
        "Final: <number>\n\n"
        f"Problem: {q}\n\nSolution:\n"
    )
    # Normalize to our "Final:" format
    final = extract_final_number(a)
    if final is None:
        # fallback: keep original
        target = a
    else:
        target = f"{strip_reasoning(a)}\nFinal: {final}"
    return {"prompt": prompt, "response": target}

def strip_reasoning(answer: str) -> str:
    # Keep everything before "####" if present
    if "####" in answer:
        return answer.split("####")[0].strip()
    return answer.strip()

def extract_final_number(text: str):
    # Prefer GSM8K marker
    m = re.search(r"####\s*([-+]?\d+(?:\.\d+)?)", text)
    if m:
        return m.group(1)
    # Otherwise try "Final: x"
    m = re.search(r"Final:\s*([-+]?\d+(?:\.\d+)?)", text, flags=re.IGNORECASE)
    if m:
        return m.group(1)
    # Last resort: last number in string
    nums = re.findall(r"[-+]?\d+(?:\.\d+)?", text)
    return nums[-1] if nums else None

AG_LABELS = ["World", "Sports", "Business", "Sci/Tech"]

def format_agnews(example: Dict) -> Dict:
    text = example["text"].strip()
    label = AG_LABELS[int(example["label"])]
    prompt = (
        "Classify the news article into one of these labels:\n"
        f"{', '.join(AG_LABELS)}\n"
        "Return only the label.\n\n"
        f"Article: {text}\n\nLabel:"
    )
    return {"prompt": prompt, "response": f" {label}"}

def format_dolly(example: Dict) -> Dict:
    instr = (example.get("instruction") or "").strip()
    ctx = (example.get("context") or "").strip()
    resp = (example.get("response") or "").strip()

    if ctx:
        prompt = (
            "You are a helpful assistant.\n\n"
            f"Instruction: {instr}\n"
            f"Context: {ctx}\n\n"
            "Response:"
        )
    else:
        prompt = (
            "You are a helpful assistant.\n\n"
            f"Instruction: {instr}\n\n"
            "Response:"
        )
    return {"prompt": prompt, "response": f" {resp}"}

def load_task_dataset(task: str, split: str = "train", limit: int = 5000):
    task = task.lower()
    if task == "math":
        ds = load_dataset("gsm8k", "main", split=split)
        ds = ds.map(format_gsm8k, remove_columns=ds.column_names)
    elif task == "cls":
        ds = load_dataset("ag_news", split=split)
        ds = ds.map(format_agnews, remove_columns=ds.column_names)
    elif task == "instruct":
        # Dolly is lightweight for demos; Alpaca-cleaned is also possible.
        ds = load_dataset("databricks/databricks-dolly-15k", split="train")
        ds = ds.map(format_dolly, remove_columns=ds.column_names)
    else:
        raise ValueError("task must be one of: math, cls, instruct")

    if limit and len(ds) > limit:
        ds = ds.select(range(limit))
    return ds

