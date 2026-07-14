#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "torch==2.9.1",
# ]
# ///
"""Export deterministic PyTorch parity fixtures for nanochat modules, models, and optimizers."""

import argparse
import importlib.metadata
import json
import os
import sys
from pathlib import Path

os.environ["NANOCHAT_DTYPE"] = "float32"
os.environ["TORCHDYNAMO_DISABLE"] = "1"
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch
from nanochat.gpt import (CausalSelfAttention, GPT, GPTConfig, MLP,
    apply_rotary_emb, norm)
from nanochat.optim import MuonAdamW


def fixed(shape: tuple[int, ...], low: float, high: float) -> torch.Tensor:
    return torch.linspace(low, high, torch.tensor(shape).prod().item(),
        dtype=torch.float32).reshape(shape)


def tensor(value: torch.Tensor) -> dict:
    value = value.detach().cpu().float().contiguous()
    return {"shape": list(value.shape), "values": value.flatten().tolist()}


def write_fixture(output: Path, fixture: dict, label: str) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(fixture, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Wrote {label} parity fixture to {output}")


def rotary(seq_len: int, head_dim: int) -> tuple[torch.Tensor, torch.Tensor]:
    channels = torch.arange(0, head_dim, 2, dtype=torch.float32)
    inv_freq = 1.0 / (100000 ** (channels / head_dim))
    freqs = torch.outer(torch.arange(seq_len, dtype=torch.float32), inv_freq)
    return freqs.cos()[None, :, None, :], freqs.sin()[None, :, None, :]


def export_modules(output: Path) -> None:
    config = GPTConfig(sequence_len=4, vocab_size=32, n_layer=1, n_head=4,
        n_kv_head=2, n_embd=16, window_pattern="L")
    x = fixed((1, 4, 16), -1.0, 1.0)
    cos, sin = rotary(config.sequence_len, config.n_embd // config.n_head)
    rope_input = fixed((1, 4, config.n_head, 4), -0.8, 0.9)

    mlp = MLP(config)
    mlp.c_fc.weight.data.copy_(fixed(tuple(mlp.c_fc.weight.shape), -0.12, 0.11))
    mlp.c_proj.weight.data.copy_(fixed(tuple(mlp.c_proj.weight.shape), -0.09, 0.08))

    attention = CausalSelfAttention(config, layer_idx=0)
    attention.c_q.weight.data.copy_(fixed(tuple(attention.c_q.weight.shape), -0.15, 0.14))
    attention.c_k.weight.data.copy_(fixed(tuple(attention.c_k.weight.shape), -0.13, 0.12))
    attention.c_v.weight.data.copy_(fixed(tuple(attention.c_v.weight.shape), -0.11, 0.10))
    attention.c_proj.weight.data.copy_(
        fixed(tuple(attention.c_proj.weight.shape), -0.08, 0.07))
    attention.ve_gate.weight.data.copy_(
        fixed(tuple(attention.ve_gate.weight.shape), -0.04, 0.05))
    ve = fixed((1, 4, config.n_kv_head * 4), -0.3, 0.3)

    fixture = {
        "schema_version": 1,
        "source": {"implementation": "nanochat.gpt",
            "torch": importlib.metadata.version("torch"), "dtype": "float32",
            "linear_weight_layout": "out_in"},
        "config": {"sequence_len": config.sequence_len, "n_head": config.n_head,
            "n_kv_head": config.n_kv_head, "n_embd": config.n_embd},
        "rms_norm": {"input": tensor(x), "output": tensor(norm(x))},
        "rope": {"input": tensor(rope_input), "cos": tensor(cos), "sin": tensor(sin),
            "output": tensor(apply_rotary_emb(rope_input, cos, sin))},
        "mlp": {"input": tensor(x), "c_fc_weight": tensor(mlp.c_fc.weight),
            "c_proj_weight": tensor(mlp.c_proj.weight), "output": tensor(mlp(x))},
        "attention": {"input": tensor(x), "value_embedding": tensor(ve),
            "cos": tensor(cos), "sin": tensor(sin),
            "c_q_weight": tensor(attention.c_q.weight),
            "c_k_weight": tensor(attention.c_k.weight),
            "c_v_weight": tensor(attention.c_v.weight),
            "c_proj_weight": tensor(attention.c_proj.weight),
            "ve_gate_weight": tensor(attention.ve_gate.weight),
            "output": tensor(attention(x, ve, (cos, sin),
                (config.sequence_len, 0), None))},
    }
    write_fixture(output, fixture, "module")


def initialize_model_parameter(name: str, parameter: torch.Tensor) -> None:
    explicit = {"resid_lambdas": [1.11, 1.04], "x0_lambdas": [0.14, 0.06],
        "smear_lambda": [0.23], "backout_lambda": [0.17]}
    if name in explicit:
        parameter.copy_(torch.tensor(explicit[name], dtype=torch.float32))
        return

    if name == "transformer.wte.weight":
        bounds = (-0.45, 0.42)
    elif name.startswith("value_embeds."):
        bounds = (-0.21, 0.19)
    elif name == "lm_head.weight":
        bounds = (-0.08, 0.07)
    elif name == "smear_gate.weight":
        bounds = (-0.035, 0.045)
    elif name.endswith("ve_gate.weight"):
        bounds = (-0.04, 0.05)
    else:
        scale = 0.04 + (sum(name.encode()) % 7) * 0.01
        bounds = (-scale, scale * 0.9)
    parameter.copy_(fixed(tuple(parameter.shape), *bounds))


def export_model(output: Path) -> None:
    config = GPTConfig(sequence_len=4, vocab_size=16, n_layer=2, n_head=4,
        n_kv_head=2, n_embd=24, window_pattern="L")
    model = GPT(config)
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            initialize_model_parameter(name, parameter)

    idx = torch.tensor([[1, 2, 3, 4], [4, 3, 2, 1]], dtype=torch.long)
    targets = torch.tensor([[2, 3, 4, -1], [3, 2, 1, 0]], dtype=torch.long)
    logits = model(idx)
    loss = model(idx, targets)
    loss.backward()
    gradient_names = ["transformer.wte.weight", "transformer.h.0.attn.c_q.weight",
        "transformer.h.0.attn.c_proj.weight", "transformer.h.1.attn.ve_gate.weight",
        "transformer.h.1.mlp.c_fc.weight", "transformer.h.1.mlp.c_proj.weight",
        "lm_head.weight", "resid_lambdas", "x0_lambdas", "smear_gate.weight",
        "smear_lambda", "backout_lambda", "value_embeds.1.weight"]
    named_parameters = dict(model.named_parameters())
    fixture = {
        "schema_version": 1,
        "source": {"implementation": "nanochat.gpt.GPT",
            "torch": importlib.metadata.version("torch"), "dtype": "float32",
            "linear_weight_layout": "out_in"},
        "config": {"sequence_len": config.sequence_len, "vocab_size": config.vocab_size,
            "n_layer": config.n_layer, "n_head": config.n_head,
            "n_kv_head": config.n_kv_head, "n_embd": config.n_embd,
            "window_pattern": config.window_pattern},
        "input_ids": idx.tolist(), "targets": targets.tolist(),
        "parameters": {name: tensor(parameter) for name, parameter in named_parameters.items()},
        "logits": tensor(logits), "loss": loss.detach().item(),
        "gradients": {name: tensor(named_parameters[name].grad) for name in gradient_names},
    }
    write_fixture(output, fixture, "full-model")


def adamw_case() -> dict:
    parameter = torch.nn.Parameter(fixed((2, 3), -0.4, 0.5))
    gradient = fixed((2, 3), 0.13, -0.17)
    initial = parameter.detach().clone()
    parameter.grad = gradient.clone()
    hyper = {"lr": 0.003, "betas": [0.8, 0.95], "eps": 1e-10,
        "weight_decay": 0.01}
    optimizer = MuonAdamW([{"kind": "adamw", "params": [parameter], **hyper}])
    optimizer.step()
    state = optimizer.state[parameter]
    return {"parameter": tensor(initial), "gradient": tensor(gradient), "hyper": hyper,
        "output": tensor(parameter), "exp_avg": tensor(state["exp_avg"]),
        "exp_avg_sq": tensor(state["exp_avg_sq"])}


def muon_case(shape: tuple[int, int]) -> dict:
    parameter = torch.nn.Parameter(fixed(shape, -0.35, 0.45))
    gradient = fixed(shape, -0.19, 0.16)
    initial = parameter.detach().clone()
    parameter.grad = gradient.clone()
    hyper = {"lr": 0.02, "momentum": 0.95, "ns_steps": 5, "beta2": 0.9,
        "weight_decay": 0.1}
    optimizer = MuonAdamW([{"kind": "muon", "params": [parameter], **hyper}])
    optimizer.step()
    state = optimizer.state[parameter]
    return {"parameter": tensor(initial), "gradient": tensor(gradient), "hyper": hyper,
        "output": tensor(parameter),
        "momentum_buffer": tensor(state["momentum_buffer"].squeeze(0)),
        "second_momentum_buffer": tensor(state["second_momentum_buffer"].squeeze(0))}


def export_optimizer(output: Path) -> None:
    fixture = {"schema_version": 1,
        "source": {"implementation": "nanochat.optim.MuonAdamW",
            "torch": importlib.metadata.version("torch"), "dtype": "float32"},
        "adamw": adamw_case(), "muon_tall": muon_case((5, 3)),
        "muon_wide": muon_case((3, 5))}
    write_fixture(output, fixture, "optimizer")


def main() -> None:
    burn_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("target", nargs="?", default="all",
        choices=("all", "modules", "model", "optimizer"))
    parser.add_argument("--output-dir", type=Path,
        default=burn_root / "data/fixtures/parity")
    args = parser.parse_args()
    exporters = {"modules": export_modules, "model": export_model,
        "optimizer": export_optimizer}
    targets = exporters if args.target == "all" else (args.target,)
    filenames = {"modules": "modules.json", "model": "model.json",
        "optimizer": "optimizer.json"}
    for target in targets:
        exporters[target](args.output_dir / filenames[target])


if __name__ == "__main__":
    main()
