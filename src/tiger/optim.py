"""Optimizer and LR schedule.

BlinkAdamW reproduces BLINK's optimizer
(pytorch_transformers.optimization.AdamW with correct_bias=False):
no bias correction and decoupled weight decay applied after the update.
torch.optim.AdamW always applies bias correction, so it is not a
drop-in replacement.
"""

import math

import torch
from torch.optim.lr_scheduler import LambdaLR

# Parameter-name patterns selecting which encoder layers are optimized,
# following BLINK's common/optimizer.py.
PATTERNS_OPTIMIZER = {
    "additional_layers": ["additional"],
    "top_layer": ["additional", "bert_model.encoder.layer.11."],
    "top4_layers": [
        "additional",
        "bert_model.encoder.layer.11.",
        "encoder.layer.10.",
        "encoder.layer.9.",
        "encoder.layer.8",
    ],
    "all_encoder_layers": ["additional", "bert_model.encoder.layer"],
    "all": ["additional", "bert_model.encoder.layer", "bert_model.embeddings"],
}
NO_DECAY = ("bias", "gamma", "beta")


class BlinkAdamW(torch.optim.Optimizer):
    def __init__(self, params, lr, betas=(0.9, 0.999), eps=1e-6,
                 weight_decay=0.0, correct_bias=False):
        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay, correct_bias=correct_bias)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                state = self.state[p]
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p)
                    state["exp_avg_sq"] = torch.zeros_like(p)
                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                state["step"] += 1

                exp_avg.mul_(beta1).add_(grad, alpha=1.0 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)
                denom = exp_avg_sq.sqrt().add_(group["eps"])

                step_size = group["lr"]
                if group["correct_bias"]:
                    bias_c1 = 1.0 - beta1 ** state["step"]
                    bias_c2 = 1.0 - beta2 ** state["step"]
                    step_size = step_size * math.sqrt(bias_c2) / bias_c1

                p.addcdiv_(exp_avg, denom, value=-step_size)
                if group["weight_decay"] > 0.0:
                    p.add_(p, alpha=-group["lr"] * group["weight_decay"])
        return loss


def build_optimizer(model, train_cfg: dict, graph_cfg: dict | None = None):
    """Group encoder params exactly like BLINK's get_bert_optimizer;
    GCN/fusion params (if present) get their own Adam-style group."""
    patterns = PATTERNS_OPTIMIZER[train_cfg.get("type_optimization", "all_encoder_layers")]

    with_decay, without_decay, graph_params = [], [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if name.startswith(("gcn.", "fusion_proj.")):
            graph_params.append(param)
        elif any(t in name for t in patterns):
            if any(t in name for t in NO_DECAY):
                without_decay.append(param)
            else:
                with_decay.append(param)

    groups = [
        {"params": with_decay, "weight_decay": 0.01},
        {"params": without_decay, "weight_decay": 0.0},
    ]
    if graph_params:
        gopt = (graph_cfg or {}).get("optimizer", {})
        groups.append({
            "params": graph_params,
            "lr": gopt.get("lr", 5e-4),
            "weight_decay": gopt.get("weight_decay", 5e-4),
            "correct_bias": True,  # plain Adam-style group for the GCN
        })
    return BlinkAdamW(groups, lr=train_cfg["learning_rate"], correct_bias=False)


def build_scheduler(optimizer, num_train_steps: int, warmup_proportion: float):
    num_warmup = int(num_train_steps * warmup_proportion)

    def lr_lambda(step):
        if step < num_warmup:
            return step / max(1, num_warmup)
        return max(0.0, (num_train_steps - step) / max(1, num_train_steps - num_warmup))

    return LambdaLR(optimizer, lr_lambda)
