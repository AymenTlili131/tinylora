from __future__ import annotations
import re
from typing import List, Dict

import torch
from evaluate import load as load_eval

rouge = load_eval("rouge")

AG_LABELS = ["World", "Sports", "Business", "Sci/Tech"]

def extract_final_number_from_generation(text: str):
    m = re.search(r"Final:\s*([-+]?\d+(?:\.\d+)?)", text, flags=re.IGNORECASE)
    if m:
        return m.group(1)
    nums = re.findall(r"[-+]?\d+(?:\.\d+)?", text)
    return nums[-1] if nums else None

def normalize_label(text: str):
    t = text.strip().lower()
    for lab in AG_LABELS:
        if lab.lower() in t:
            return lab
    return text.strip().split()[0] if text.strip() else ""

@torch.no_grad()
def generate_batch(
    model,
    tokenizer,
    prompts: List[str],
    max_new_tokens: int = 64,
    temperature: float = 0.0,
    batch_size: int = 4,
    max_length: int = 512,
):
    """
    Safer generation for demos:
    - chunked micro-batches to reduce peak VRAM
    - shorter context cap to avoid OOM on instruction prompts
    """
    outs: List[str] = []
    prompts = ["" if p is None else str(p) for p in prompts]

    for i in range(0, len(prompts), batch_size):
        chunk = prompts[i:i + batch_size]
        inputs = tokenizer(
            chunk,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        )
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        gen = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=(temperature > 0),
            temperature=temperature if temperature > 0 else None,
            pad_token_id=tokenizer.eos_token_id,
        )
        outs.extend(tokenizer.batch_decode(gen, skip_special_tokens=True))
    return outs

def eval_math(generations: List[str], targets: List[str]) -> Dict[str, float]:
    correct = 0
    total = len(generations)
    for g, t in zip(generations, targets):
        gn = extract_final_number_from_generation(g)
        tn = extract_final_number_from_generation(t)
        if gn is not None and tn is not None and str(gn).strip() == str(tn).strip():
            correct += 1
    return {"math_exact_match": correct / max(1, total)}

def eval_cls(generations: List[str], targets: List[str]) -> Dict[str, float]:
    correct = 0
    total = len(generations)
    for g, t in zip(generations, targets):
        gl = normalize_label(g)
        tl = normalize_label(t)
        if gl == tl:
            correct += 1
    return {"cls_accuracy": correct / max(1, total)}

def eval_instruct(generations: List[str], targets: List[str]) -> Dict[str, float]:
    scores = rouge.compute(predictions=generations, references=targets, use_aggregator=True)
    return {"rougeL": float(scores["rougeL"])}
