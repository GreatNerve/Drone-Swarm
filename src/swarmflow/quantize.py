"""Post-training quantization: small actor-only models for deployment, compared with the original.

Every variant keeps only what flies the drones (the encoder and the action head); the critic, the
exploration noise and the optimizer state are dropped. Weights are stored as plain tensors, so every
file loads with torch.load(weights_only=True) on any device and works with `eval` and `view`:

- fp32: the actor as trained;
- fp16: weights in half precision, computed in float32 (runs on any CPU);
- int8: symmetric 8-bit weights with float32 scales (per output channel or per weight matrix), dequantized
  at load and computed in float32: a quarter of the fp32 size and the same arithmetic on every device;
- int8-dynamic: the same 8-bit weights run by PyTorch's dynamic INT8 kernels, which also quantize each
  layer's input at runtime, when this machine has a quantized engine (x86, fbgemm or onednn on Intel and
  AMD, qnnpack on ARM). Where there is none, or on PyTorch releases without quantized tensors (they are
  deprecated), it falls back to the int8 behaviour, so the file still works. Attention projections are
  always computed in float32.
"""
import copy
import time
import warnings
from pathlib import Path
import numpy as np
import torch
from torch import nn

from .files import load_torch, portable_path, save_torch_atomic, write_json_atomic, write_text_atomic

VARIANTS = {
    "fp32": {"precision": "fp32", "description": "actor only, float32 weights (critic and optimizer dropped)"},
    "fp16": {"precision": "fp16", "description": "actor only, float16 weights computed in float32"},
    "int8": {"precision": "int8", "per_channel": True, "keep_head": False, "kernels": False,
             "description": "8-bit weights, one scale per output channel, computed in float32"},
    "int8-dynamic": {"precision": "int8", "per_channel": True, "keep_head": False, "kernels": True,
                     "description": "8-bit weights run by PyTorch dynamic INT8 kernels (inputs quantized too)"},
    "int8-per-tensor": {"precision": "int8", "per_channel": False, "keep_head": False, "kernels": False,
                        "description": "8-bit weights, one scale per weight matrix (coarsest)"},
    "int8-keep-head": {"precision": "int8", "per_channel": True, "keep_head": True, "kernels": False,
                       "description": "8-bit weights per channel; the final action layer stays float32"},
}
DEFAULT_VARIANTS = ("fp32", "fp16", "int8", "int8-dynamic")
ACTOR_PREFIXES = ("own.", "neighbors.", "obstacles.", "trunk.", "actor.")
SAMPLE_SEED_START = 7000  # observations for the action-error check; not a training, selection or test seed


def print_variants():
    print("Quantization presets (python -m swarmflow quantize --variants ...):")
    for name, settings in VARIANTS.items():
        default = " (default)" if name in DEFAULT_VARIANTS else ""
        print(f"  {name:16s} {settings['description']}{default}")


def _quantize(weight, per_channel):
    if per_channel:
        scale = weight.abs().amax(dim=1).clamp(min=1e-12) / 127.0
        quantized = torch.round(weight / scale[:, None])
    else:
        scale = (weight.abs().max().clamp(min=1e-12) / 127.0).reshape(1)
        quantized = torch.round(weight / scale)
    return quantized.clamp(-127, 127).to(torch.int8), scale.float()


def build_variant(data, name, source=None):
    """Actor-only checkpoint dict for preset `name` from a full training checkpoint dict."""
    from .policy import ACTOR_FORMAT, CHECKPOINT_FORMAT
    if name not in VARIANTS:
        raise ValueError(f"Unknown quantization preset {name!r}; choose from {', '.join(VARIANTS)}")
    if data.get("format") != CHECKPOINT_FORMAT:
        raise ValueError(f"Quantization needs a full {CHECKPOINT_FORMAT} training checkpoint (a stage's best.pt or "
                         f"models/showcase/best.pt), not {data.get('format')}")
    settings = VARIANTS[name]
    weights, scales = {}, {}
    for key, value in data["model"].items():
        if not key.startswith(ACTOR_PREFIXES):
            continue
        value = value.detach().float().cpu()
        if settings["precision"] == "fp16":
            weights[key] = value.half()
        elif (settings["precision"] == "int8" and value.dim() == 2
              and not (settings["keep_head"] and key.startswith("actor."))):
            weights[key], scales[key] = _quantize(value, settings["per_channel"])
        else:
            weights[key] = value
    network = data.get("network") or {"width": 64}
    return {"format": ACTOR_FORMAT, "variant": name, "precision": settings["precision"],
            "settings": {k: v for k, v in settings.items() if k != "description"},
            "network": {"width": int(network.get("width", 64)), "gap_features": bool(network.get("gap_features"))},
            "config": data["config"], "weights": weights, "scales": scales,
            "source_checkpoint": portable_path(source) if source else None,
            "last_trained_at": data.get("last_trained_at"),
            "environment_steps": data.get("environment_steps"), "training": data.get("training"),
            "selection": data.get("selection"),
            "status": f"{settings['precision']} actor-only model; {data.get('status', '')}".rstrip("; ")}


def _int8_engine():
    engines = list(getattr(torch.backends.quantized, "supported_engines", []))
    return next((name for name in ("x86", "fbgemm", "onednn", "qnnpack") if name in engines), None)


def _example_input(model):
    rows = torch.zeros(2, 3, 8)
    rows[:, 0, :7] = torch.tensor([1.0, 0.5, 0.0, 0.1, 0.0, 0.0, 0.18])
    rows[:, :, 7] = torch.tensor([1.0, 0.0, 0.0])
    return {"self": torch.linspace(-1, 1, 34).reshape(2, 17), "neighbors": rows, "obstacles": rows.clone()}


def _with_int8_kernels(model, weights, scales):
    """A copy of `model` whose linear layers run PyTorch dynamic INT8 kernels, or None if unavailable here."""
    engine = _int8_engine()
    if engine is None:
        return None
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            from torch.ao.nn.quantized.dynamic import Linear as DynamicLinear
            torch.backends.quantized.engine = engine
            fast = copy.deepcopy(model)
            for name, module in list(fast.named_modules()):
                if type(module) is not nn.Linear or f"{name}.weight" not in scales:
                    continue
                q, scale = weights[f"{name}.weight"], scales[f"{name}.weight"]
                if scale.numel() > 1:
                    qweight = torch._make_per_channel_quantized_tensor(
                        q, scale.double(), torch.zeros(scale.numel(), dtype=torch.long), 0)
                else:
                    qweight = torch._make_per_tensor_quantized_tensor(q, float(scale), 0)
                layer = DynamicLinear(module.in_features, module.out_features, dtype=torch.qint8)
                layer.set_weight_bias(qweight, None if module.bias is None else module.bias.detach().clone())
                parent_name, _, child = name.rpartition(".")
                setattr(fast.get_submodule(parent_name) if parent_name else fast, child, layer)
            with torch.no_grad():
                example = _example_input(model)
                if not torch.allclose(fast.act(example), model.act(example), atol=0.1):
                    return None
        fast.compute = f"int8 kernels ({engine})"
        return fast
    except Exception:
        return None


def load_actor(data, kernels=True):
    """ActorOnly model from an actor-only checkpoint dict (any precision), ready for ModelController."""
    from .policy import ActorOnly
    model = ActorOnly(**data["network"])
    scales = data.get("scales") or {}
    state = {}
    for key, value in data["weights"].items():
        state[key] = value.float() * scales[key].float().reshape(-1, 1) if key in scales else value.float()
    model.load_state_dict(state)
    model.eval()
    model.compute = "float32"
    if kernels and (data.get("settings") or {}).get("kernels") and scales:
        fast = _with_int8_kernels(model, data["weights"], scales)
        if fast is not None:
            fast.eval()
            return fast
    return model


def _sample_observations(config, scenarios, controller, seeds=2, every=4):
    from .env import SwarmEnv
    samples = []
    for scenario in scenarios:
        for seed in range(SAMPLE_SEED_START, SAMPLE_SEED_START + seeds):
            env = SwarmEnv(config, scenario)
            observations, _ = env.reset(seed=seed)
            while env.agents:
                if env.steps % every == 0:
                    samples.append(observations)
                observations, *_ = env.step(controller(observations))
    return samples


def _actions(controller, samples):
    return np.stack([np.stack(list(controller(observations).values())) for observations in samples])


def _latency_ms(controller, samples, count=300):
    chosen = samples[:count]
    for observations in chosen[:10]:
        controller(observations)
    began = time.perf_counter()
    for observations in chosen:
        controller(observations)
    return 1000 * (time.perf_counter() - began) / max(1, len(chosen))


def _metrics(report):
    from .evaluate import RECOVERY_LIMIT_S, filter_share, learning_checks, min_obstacle_clearance
    episodes = report["episodes"]
    failures = [e for e in episodes if e["failure"]]
    events = [event for e in episodes for event in e["events"]]
    quick = [event for e in episodes if not e["failure"] for event in e["events"]
             if event["recovery_s"] is not None and event["recovery_s"] <= RECOVERY_LIMIT_S]
    checks = learning_checks(report)
    rows = list(report["summary"].values())
    result = {"episodes": len(episodes), "failures": len(failures),
              "failed": [f"{e['scenario']}:{e['seed']}" for e in failures],
              "events": len(events), "recovered_within_limit": len(quick),
              "mean_slot_rmse_m": float(np.mean([r["mean_slot_rmse_m"] for r in rows])),
              "min_obstacle_clearance_m": min_obstacle_clearance(report["summary"]),
              "checks_passed": sum(c["passed"] for c in checks), "checks": len(checks),
              "passes_verdict": all(c["passed"] for c in checks)}
    share = filter_share(report)
    if share is not None:
        result["filter_share"] = share
    return result


def quantize_checkpoint(checkpoint, variants=DEFAULT_VARIANTS, output=None, seeds=20, seed_start=1000,
                        scenarios=None, safety_filter=False):
    """Build each preset from `checkpoint`, check it against the original and write report.json + QUANTIZATION.md."""
    from .curriculum import SHOWCASE_SCENARIOS
    from .evaluate import evaluate
    from .policy import CHECKPOINT_FORMAT, LearnedController
    checkpoint = Path(checkpoint)
    if not checkpoint.exists():
        raise ValueError(f"{checkpoint} does not exist; train first (make training) or pass --checkpoint")
    variants = list(dict.fromkeys(variants))
    unknown = [name for name in variants if name not in VARIANTS]
    if unknown:
        raise ValueError(f"Unknown quantization preset(s) {', '.join(unknown)}; choose from {', '.join(VARIANTS)}")
    data = load_torch(checkpoint)
    if data.get("format") != CHECKPOINT_FORMAT:
        raise ValueError(f"{checkpoint} is not a full {CHECKPOINT_FORMAT} training checkpoint")
    output = Path(output) if output else checkpoint.parent / "quantized"
    scenarios = list(scenarios or SHOWCASE_SCENARIOS)
    reference = LearnedController(checkpoint)
    config = reference.config
    print(f"\nQuantizing {checkpoint} into {output}/ ({', '.join(variants)})", flush=True)
    samples = _sample_observations(config, scenarios, reference)
    reference_actions = _actions(reference, samples)
    full_size = checkpoint.stat().st_size
    seed_range = range(seed_start, seed_start + seeds)
    rows = []

    def measure(name, path, controller, description, precision, folder):
        error = np.abs(_actions(controller, samples) - reference_actions)
        size = Path(path).stat().st_size
        row = {"variant": name, "description": description, "precision": precision,
               "file": portable_path(path), "size_bytes": size, "size_ratio": size / full_size,
               "mean_action_error": float(error.mean()), "max_action_error": float(error.max()),
               "ms_per_team_step": _latency_ms(controller, samples),
               "compute": getattr(controller.model, "compute", "float32")}
        if seeds:
            report = evaluate(config, scenarios, seed_range, "learned", str(path), str(folder / "evaluation.json"),
                              quiet=True)
            row.update(_metrics(report))
            if safety_filter:
                filtered = evaluate(config, scenarios, seed_range, "learned", str(path),
                                    str(folder / "evaluation_filter.json"), safety_filter=True, quiet=True)
                row["with_filter"] = _metrics(filtered)
        rows.append(row)
        result = (f"  failures {row['failures']}/{row['episodes']}, recovered {row['recovered_within_limit']}/"
                  f"{row['events']}, slot RMSE {row['mean_slot_rmse_m']:.3f} m" if seeds else "")
        print(f"{name:16s} {size / 1024:8.1f} KB ({row['size_ratio']:.1%} of the checkpoint)  action error mean "
              f"{row['mean_action_error']:.5f} max {row['max_action_error']:.4f}  {row['ms_per_team_step']:.2f} ms/step "
              f"[{row['compute']}]{result}", flush=True)

    measure("original", checkpoint, reference, "full training checkpoint (actor, critic, optimizer)", "fp32",
            output / "original")
    for name in variants:
        path = output / name / "model.pt"
        save_torch_atomic(build_variant(data, name, checkpoint), path)
        measure(name, path, LearnedController(path), VARIANTS[name]["description"], VARIANTS[name]["precision"],
                output / name)
    report = {"checkpoint": portable_path(checkpoint), "scenarios": scenarios,
              "seeds": [seed_range.start, seed_range.stop - 1] if seeds else None,
              "action_samples": int(reference_actions.shape[0] * reference_actions.shape[1]),
              "torch_threads": torch.get_num_threads(), "variants": rows}
    write_json_atomic(output / "report.json", report)
    write_text_atomic(output / "QUANTIZATION.md", _markdown(report))
    print(f"Quantization report: {output / 'QUANTIZATION.md'}", flush=True)
    return report


def _markdown(report):
    seeds = report["seeds"]
    lines = ["# Quantized models", "",
             f"- Source: `{report['checkpoint']}`",
             f"- Action error: largest absolute difference from the original's command (each axis in [-1, 1]) over "
             f"{report['action_samples']:,} drone states",
             f"- Latency: one 8-drone team step on this CPU with {report['torch_threads']} PyTorch thread(s)"]
    if seeds:
        lines.append(f"- Flight check: {len(report['scenarios'])} scenarios x {seeds[1] - seeds[0] + 1} seeds "
                     f"({seeds[0]}..{seeds[1]}); verdict = the three acceptance checks")
    lines += ["", "| Variant | Size | vs checkpoint | Mean / max action error | ms per step | Compute"
              + (" | Failures | Recovered within 5 s | Slot RMSE m | Min obstacle clearance m | Verdict" if seeds else "")
              + " |", "| --- " * (11 if seeds else 6) + "|"]
    for row in report["variants"]:
        cells = [row["variant"], f"{row['size_bytes'] / 1024:.0f} KB", f"{row['size_ratio']:.1%}",
                 f"{row['mean_action_error']:.5f} / {row['max_action_error']:.4f}", f"{row['ms_per_team_step']:.2f}",
                 row["compute"]]
        if seeds:
            clearance = row["min_obstacle_clearance_m"]
            cells += [f"{row['failures']}/{row['episodes']}", f"{row['recovered_within_limit']}/{row['events']}",
                      f"{row['mean_slot_rmse_m']:.3f}", "-" if clearance is None else f"{clearance:.3f}",
                      "PASS" if row["passes_verdict"] else f"{row['checks_passed']}/{row['checks']} checks"]
        lines.append("| " + " | ".join(cells) + " |")
    filtered = [row for row in report["variants"] if "with_filter" in row]
    if filtered:
        lines += ["", "With the safety filter:", "", "| Variant | Failures | Filter share | Verdict |",
                  "| --- | --- | --- | --- |"]
        for row in filtered:
            f = row["with_filter"]
            lines.append(f"| {row['variant']} | {f['failures']}/{f['episodes']} | {f.get('filter_share', 0):.3%} | "
                         f"{'PASS' if f['passes_verdict'] else str(f['checks_passed']) + '/' + str(f['checks']) + ' checks'} |")
    lines += ["", "Use a variant like any checkpoint: `python -m swarmflow view --checkpoint <variant>/model.pt` or "
              "`python -m swarmflow eval --policy learned --checkpoint <variant>/model.pt`.", ""]
    return "\n".join(lines)
