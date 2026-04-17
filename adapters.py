from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, List, Dict

import torch
import torch.nn as nn

from peft import (
    LoraConfig,
    AdaLoraConfig,
    VeraConfig,
    VBLoRAConfig,
    get_peft_model,
)

# ----------- Helpers -----------

def infer_target_modules(model) -> List[str]:
    """
    Reasonable defaults for decoder-only transformer attention projections.
    Works with LLaMA/Mistral/Qwen-like naming.
    If your model uses different names, pass --target_modules explicitly.
    """
    names = [n for n, _ in model.named_modules()]
    if any(n.endswith("q_proj") for n in names):
        return ["q_proj", "k_proj", "v_proj", "o_proj"]
    if any(n.endswith("query_key_value") for n in names):
        return ["query_key_value"]
    if any(n.endswith("c_attn") for n in names):
        return ["c_attn", "c_proj"]
    return ["q_proj", "v_proj"]

def count_trainable_params(model) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# ----------- TinyLoRA-style demo (tiny trainable params) -----------

class TinySharedLoRALinear(nn.Module):
    """
    A *very* tiny adapter wrapper for demos.
    We keep low-rank factors as BUFFERS (frozen) and train only a scalar 'scale' per layer.

        y = Wx + scale * (B @ (A @ x))

    This gives vivid "Tiny adapter" behavior without exploding trainable params.
    """
    def __init__(self, base: nn.Linear, A: torch.Tensor, B: torch.Tensor, init_scale: float = 0.0):
        super().__init__()
        self.base = base
        self.register_buffer("A", A)  # [r, in] frozen
        self.register_buffer("B", B)  # [out, r] frozen
        self.scale = nn.Parameter(torch.tensor(init_scale, dtype=torch.float32))  # trainable

    def forward(self, x):
        y = self.base(x)
        adapter = torch.matmul(torch.matmul(x, self.A.t()), self.B.t())
        return y + self.scale * adapter


def apply_tinylora_demo(model, target_modules: Optional[List[str]] = None, r: int = 2):
    """
    Replaces selected nn.Linear layers (by name ending) with TinySharedLoRALinear.
    For robustness (different in/out dims), we create frozen A,B PER LAYER and train only scale.
    Trainable params ~= (#replaced layers).

    NOTE: This is a demo variant inspired by TinyLoRA-style parameter efficiency. If you need the
    *exact* paper method, you'd implement their specific factor-sharing rule.
    """
    if target_modules is None:
        target_modules = infer_target_modules(model)

    replaced = 0
    for name, module in list(model.named_modules()):
        if not (isinstance(module, nn.Linear) and any(name.endswith(tm) for tm in target_modules)):
            continue

        # Get parent module
        parent = model
        parts = name.split(".")
        for p in parts[:-1]:
            parent = getattr(parent, p)
        leaf = parts[-1]
        base_linear: nn.Linear = getattr(parent, leaf)

        in_dim = base_linear.in_features
        out_dim = base_linear.out_features
        dtype = base_linear.weight.dtype
        device = base_linear.weight.device

        # frozen low-rank factors
        A = (torch.randn(r, in_dim, device=device, dtype=dtype) * 0.01)
        B = (torch.randn(out_dim, r, device=device, dtype=dtype) * 0.01)

        wrapped = TinySharedLoRALinear(base_linear, A, B, init_scale=0.0)

        # freeze base weights
        for p in wrapped.base.parameters():
            p.requires_grad = False

        setattr(parent, leaf, wrapped)
        replaced += 1

    # Freeze everything, then re-enable only scales
    for p in model.parameters():
        p.requires_grad = False
    for m in model.modules():
        if isinstance(m, TinySharedLoRALinear):
            m.scale.requires_grad = True

    return model, {"tinylora_r": r, "replaced_layers": replaced, "trainable_per_layer": "scale"}

# ----------- PEFT variants -----------

@dataclass
class AdapterSpec:
    method: str
    r: int = 8
    alpha: int = 16
    dropout: float = 0.05
    target_modules: Optional[List[str]] = None
    total_step: Optional[int] = None  # for AdaLoRA

def attach_adapter(model, spec: AdapterSpec):
    method = spec.method.lower()
    target_modules = spec.target_modules or infer_target_modules(model)

    if method == "lora":
        cfg = LoraConfig(
            r=spec.r,
            lora_alpha=spec.alpha,
            lora_dropout=spec.dropout,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=target_modules,
        )
        model = get_peft_model(model, cfg)
        return model, {"target_modules": target_modules}

    if method == "adalora":
        total_step = spec.total_step
        if total_step is None or total_step <= 0:
            raise ValueError(
                "AdaLoRA requires total_step > 0. "
                "Run with --max_steps > 0 (we pass that into AdapterSpec.total_step)."
            )

        # small-step friendly schedule
        tinit = max(10, int(0.05 * total_step))
        tfinal = max(tinit + 20, int(0.5 * total_step))

        cfg = AdaLoraConfig(
            r=spec.r,
            lora_alpha=spec.alpha,
            lora_dropout=spec.dropout,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=target_modules,
            total_step=int(total_step),
            tinit=int(tinit),
            tfinal=int(tfinal),
            deltaT=5,
            beta1=0.85,
            beta2=0.85,
        )
        model = get_peft_model(model, cfg)
        return model, {
            "target_modules": target_modules,
            "total_step": int(total_step),
            "tinit": int(tinit),
            "tfinal": int(tfinal),
        }

    if method == "vera":
        cfg = VeraConfig(
            task_type="CAUSAL_LM",
            target_modules=target_modules,
        )
        model = get_peft_model(model, cfg)
        return model, {"target_modules": target_modules}

    if method == "vblora":
        # Qwen2.5-0.5B hidden size is 896, so vector_length must divide it.
        cfg = VBLoRAConfig(
            task_type="CAUSAL_LM",
            target_modules=target_modules,
            vector_length=128,
        )
        model = get_peft_model(model, cfg)
        return model, {"target_modules": target_modules, "vector_length": 128}

    if method == "tinylora":
        model, info = apply_tinylora_demo(
            model,
            target_modules=target_modules,
            r=max(1, min(8, spec.r)),
        )
        info["target_modules"] = target_modules
        return model, info

    raise ValueError(f"Unknown adapter method: {spec.method}")
