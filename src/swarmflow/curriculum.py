"""Training curriculum, the resumable full pipeline, and the showcase (best model) folder."""
import re
import shutil
import time
from dataclasses import dataclass
from pathlib import Path

from .config import Config
from .files import (local_path, portable_path, read_json, read_jsonl, rename_with_retry, retry, write_json_atomic,
                    write_text_atomic)

MODELS_DIR = Path("models")
SHOWCASE_DIR = MODELS_DIR / "showcase"
TEST_SEED_START = 100000
# Slalom (static spheres sitting on the formation slots) is left out of the learned-policy benchmark;
# the baseline tests still cover it, and `swarmflow eval --scenarios slalom` tests a model on it.
SHOWCASE_SCENARIOS = ("hold", "follow", "crossing", "head_on", "double_crossing", "turn", "gust")


@dataclass(frozen=True)
class Stage:
    number: int
    title: str
    scenarios: tuple
    steps: int
    resume_from: int | None = None
    clone_steps: int = 0
    dagger_rounds: int = 0
    lr: float = 3e-4
    failure_reset: bool = False
    hard_weight: float = 1.0


# Version 2: every stage clones the teacher on its own scenarios, keeps that data as an imitation
# loss, attributes crashes to the drones involved and warns 1 m before obstacles.
# Version 3: the teacher keeps 0.6 m / 0.25 m clearance while demonstrating (the baseline's 0.08 m
# left no room for imitation error), the imitation loss stays at 0.5 or more, and slalom is dropped.
# Version 4: teacher clearance 0.7 m / 0.3 m, and 1M PPO steps in stages 2-3 (was 2M): run 3's PPO never
# beat the cloned start of those stages, and run 2's best came after 600k; the best checkpoint is kept anyway.
# Version 5: best.pt is chosen by a clearance-aware score on 20 selection seeds per scenario (run 3's
# chosen model kept 5 cm in head-on and grazed on the test), and every run can be stopped and continued
# exactly (training.md Part F). Cloning stays at v4's 3 DAgger rounds until more rounds with near-object
# emphasis (--dagger-rounds 5 --hard-weight 3) are shown to help.
PIPELINE_VERSION = 5

STAGES = {
    1: Stage(1, "Formation keeping: hold, follow and turn with the target",
             ("hold", "follow", "turn"), 1_000_000, clone_steps=10_000, dagger_rounds=3),
    2: Stage(2, "Moving obstacles: crossing and head-on objects, keeping the stage 1 tasks",
             ("hold", "follow", "turn", "crossing", "head_on"), 1_000_000, resume_from=1,
             clone_steps=20_000, dagger_rounds=3, lr=1e-4, failure_reset=True),
    3: Stage(3, "Full benchmark: repeated crossings and wind gusts, keeping the earlier tasks",
             SHOWCASE_SCENARIOS, 1_000_000, resume_from=2,
             clone_steps=20_000, dagger_rounds=3, lr=1e-4, failure_reset=True),
}

# Shared PPO settings for every stage; 8 environments x 256 steps x 8 drones = 16,384 samples per update.
COMMON = {"envs": 8, "horizon": 256, "epochs": 5, "minibatch": 2048, "eval_every": 100_000,
          "eval_seeds": 20, "snapshot_every": 500_000, "width": 128, "min_log_std": -3.0,
          "entropy": 0.0, "bc_coef": 1.0, "bc_coef_end": 0.5, "teacher_margins": (0.7, 0.3),
          "pipeline_version": PIPELINE_VERSION}


def stage_dir(number, seed, models_dir=MODELS_DIR):
    return Path(models_dir) / f"stage{number}-seed{seed}"


def stage_arguments(number, seed=0, models_dir=MODELS_DIR):
    """Keyword arguments for train() that reproduce one curriculum stage."""
    stage = STAGES[number]
    args = {**COMMON, "stage": number, "seed": seed, "scenarios": list(stage.scenarios),
            "eval_scenarios": list(stage.scenarios), "total_steps": stage.steps,
            "clone_steps": stage.clone_steps, "dagger_rounds": stage.dagger_rounds,
            "lr": stage.lr, "failure_reset": stage.failure_reset, "hard_weight": stage.hard_weight,
            "output": str(stage_dir(number, seed, models_dir))}
    if stage.resume_from:
        args["resume"] = str(stage_dir(stage.resume_from, seed, models_dir) / "best.pt")
    return args


def train_stage(number, seed=0, models_dir=MODELS_DIR, config=None, retrain=False, **overrides):
    """Train one stage, or finish it if it was interrupted; skip it if it already finished."""
    from .policy import LearnedController
    from .train import train
    folder = stage_dir(number, seed, models_dir)
    started_without_update = folder.exists() and not (folder / "latest.pt").exists() and any(folder.iterdir())
    outdated = _pipeline_version(folder) not in (None, PIPELINE_VERSION)
    if (retrain or started_without_update or outdated) and folder.exists():
        _archive(folder, "trained by an older pipeline version" if outdated else "restarting this stage")
    if (folder / "finished.json").exists():
        print(f"Stage {number} (seed {seed}) already finished in {folder}; skipping.", flush=True)
        return folder / "best.pt"
    args = {**stage_arguments(number, seed, models_dir), **overrides}
    print(f"\n=== Stage {number}, seed {seed}: {STAGES[number].title} ===", flush=True)
    if (folder / "latest.pt").exists():
        print(f"Continuing the interrupted run in {folder}", flush=True)
        args["continue_run"] = True
        config = LearnedController(folder / "latest.pt").config
    elif args.get("resume"):
        if not Path(args["resume"]).exists():
            raise ValueError(f"Stage {number} starts from {args['resume']}; train stage {number - 1} first")
        config = LearnedController(args["resume"]).config
    train(config or Config(), **args)
    return folder / "best.pt"


def _read(path):
    try:
        return read_json(path)
    except (OSError, ValueError):
        return None


def _pipeline_version(folder):
    """Pipeline version that trained `folder` (1 for runs from before versioning), or None if untrained."""
    for name in ("config.json", "summary.json"):
        settings = _read(Path(folder) / name)
        if settings is not None:
            return settings.get("pipeline_version") or 1
    return None


def _archive(folder, reason):
    from datetime import datetime
    archived = folder.with_name(f"{folder.name}-old-{datetime.now():%Y%m%d-%H%M%S}")
    rename_with_retry(folder, archived)
    print(f"Moved {folder} to {archived} ({reason})", flush=True)


def _archive_outdated(models_dir):
    """Set aside stage runs and a showcase from older pipeline versions, so nothing mixes with them."""
    folders = [f for f in Path(models_dir).glob("stage*-seed*") if re.fullmatch(r"stage\d+-seed\d+", f.name)]
    for folder in sorted(folders) + [Path(models_dir) / "showcase"]:
        if _pipeline_version(folder) not in (None, PIPELINE_VERSION):
            _archive(folder, "trained by an older pipeline version")


def run_pipeline(seeds=(0,), stages=(1, 2, 3), models_dir=MODELS_DIR, config=None, showcase=True,
                 test_seeds=20, quantize=True, **overrides):
    total = sum(STAGES[n].steps for n in stages) * len(seeds)
    print("SwarmFlow training pipeline", flush=True)
    for number in stages:
        stage = STAGES[number]
        print(f"  stage {number}: {stage.steps:,} PPO steps on {', '.join(stage.scenarios)}"
              + (f" (after {stage.dagger_rounds + 1} behavior-cloning rounds)" if stage.clone_steps else ""),
              flush=True)
    print(f"  seeds {list(seeds)}: {total:,} PPO steps in total, results in {Path(models_dir)}/stageN-seedS/; "
          f"the best model goes to {Path(models_dir) / 'showcase'}/", flush=True)
    print("  Ctrl+C stops after the current update and saves; run the same command again to continue exactly "
          "where it stopped.", flush=True)
    _archive_outdated(models_dir)
    for seed in seeds:
        for number in stages:
            train_stage(number, seed, models_dir, config, **overrides)
    if showcase:
        output = Path(models_dir) / "showcase"
        build_showcase(trained_runs(models_dir, max(stages)), output, test_seeds)
        if quantize:
            from .quantize import DEFAULT_VARIANTS, quantize_checkpoint
            quantize_checkpoint(output / "best.pt", DEFAULT_VARIANTS)


def trained_runs(models_dir=MODELS_DIR, stage=None):
    """Run folders (one per seed) that already have a best.pt, for `stage` or else the highest stage."""
    found = {}
    for folder in Path(models_dir).glob("stage*-seed*"):
        match = re.fullmatch(r"stage(\d+)-seed(\d+)", folder.name)
        if match and (folder / "best.json").exists():
            found.setdefault(int(match.group(1)), []).append(folder)
    if not found:
        return []
    return sorted(found.get(stage, []) if stage is not None else found[max(found)])


def training_chain(run):
    """Run folders from the first stage to `run`, following each stage's initial checkpoint."""
    chain, folder = [], Path(run)
    while folder is not None and (folder / "config.json").exists() and folder not in chain:
        chain.append(folder)
        settings = _read(folder / "config.json") or {}
        source = local_path(settings.get("initialized_from") or settings.get("resume"))
        folder = source.parent if source else None
    return chain[::-1]


def _stage_label(folder):
    settings = _read(Path(folder) / "config.json")
    if settings is None:
        return Path(folder).name
    return f"Stage {settings['stage']}" if settings.get("stage") else Path(folder).name


def _scenario_metrics(row):
    keys = ("success_rate", "mean_slot_rmse_m", "events", "recovered_events", "min_surface_clearance_m")
    metrics = {k: row[k] for k in keys}
    if "filter_share" in row:
        metrics["filter_share"] = row["filter_share"]
    return metrics


def build_showcase(runs=None, output=SHOWCASE_DIR, test_seeds=20, seed_start=TEST_SEED_START,
                   scenarios=SHOWCASE_SCENARIOS, safety_filter=True):
    """Select the best trained run, test it on held-out seeds against the baseline and store it.

    The candidate is chosen by its validation (selection-seed) score only; the held-out test seeds
    are used once, after the choice, so the reported test result is not tuned to them. The policy is
    tested on its own (the verdict) and, separately, behind the opt-in safety filter.
    """
    from .evaluate import compare_reports, evaluate, filter_share, learning_checks, selection_score
    from .policy import LearnedController
    runs = [Path(r) for r in runs] if runs else trained_runs(Path(output).parent)
    if not runs:
        raise ValueError("No trained runs found; train first (make training)")
    candidates = []
    for run in runs:
        best = run / "best.json"
        if not best.exists():
            raise ValueError(f"{run} has no best.json; train or finish it first")
        row = read_json(best)
        candidates.append({"run": portable_path(run), "score": row["score"],
                           "environment_steps": row["environment_steps"], "scenarios": row["scenarios"]})
    if len({tuple(c["scenarios"]) for c in candidates}) > 1:
        raise ValueError("Candidate runs were validated on different scenarios; compare runs of the same stage")
    chosen = max(candidates, key=lambda c: c["score"])
    source = local_path(chosen["run"])
    output = Path(output)
    if (output / "best.pt").exists():
        from datetime import datetime
        archive = output.parent / f"{output.name}-archive" / f"{datetime.now():%Y%m%d-%H%M%S}"
        archive.parent.mkdir(parents=True, exist_ok=True)
        rename_with_retry(output, archive)
        print(f"Previous showcase kept in {archive}", flush=True)
    output.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source / "best.pt", output / "best.pt")
    shutil.copy2(source / "best.json", output / "validation.json")
    chain = training_chain(source)
    history = output / "history"
    if history.exists():
        retry(lambda: shutil.rmtree(history))
    for folder in chain:
        for name in ("config.json", "training.jsonl", "evaluations.jsonl", "best.json", "finished.json"):
            if (folder / name).exists():
                (history / folder.name).mkdir(parents=True, exist_ok=True)
                shutil.copy2(folder / name, history / folder.name / name)
    print(f"Selected {source} (validation score {chosen['score']:.3f} at "
          f"{chosen['environment_steps']:,} steps) from {len(candidates)} run(s)", flush=True)
    checkpoint = str(output / "best.pt")
    config = LearnedController(checkpoint).config
    seeds = range(seed_start, seed_start + test_seeds)
    print(f"Held-out test: {len(scenarios)} scenarios x {test_seeds} seeds ({seeds.start}..{seeds.stop - 1})",
          flush=True)
    learned = evaluate(config, list(scenarios), seeds, "learned", checkpoint, str(output / "test_learned.json"))
    baseline = evaluate(config, list(scenarios), seeds, "baseline", None, str(output / "test_baseline.json"))
    filtered = None
    if safety_filter:
        print("Held-out test of the same policy behind the safety filter", flush=True)
        filtered = evaluate(config, list(scenarios), seeds, "learned", checkpoint,
                            str(output / "test_learned_filter.json"), safety_filter=True)
    comparison = compare_reports((output / "test_baseline.json", output / "test_learned.json"),
                                 str(output / "test_compare.json"))
    checks = learning_checks(learned)
    table = []
    for name in scenarios:
        entry = {"scenario": name, "baseline": _scenario_metrics(baseline["summary"][name]),
                 "learned": _scenario_metrics(learned["summary"][name])}
        if filtered:
            entry["learned_with_filter"] = _scenario_metrics(filtered["summary"][name])
        table.append(entry)
    summary = {"pipeline_version": _pipeline_version(source), "selected_run": portable_path(source),
               "selected_environment_steps": chosen["environment_steps"],
               "validation_score": chosen["score"], "candidates": candidates,
               "chain": [folder.name for folder in chain], "stages": [_stage_label(f) for f in chain],
               "training_stages": [_stage_record(folder) for folder in chain],
               "test_seeds": [seeds.start, seeds.stop - 1], "test_scenarios": list(scenarios),
               "test_score_learned": selection_score(learned["summary"]),
               "test_score_baseline": selection_score(baseline["summary"]),
               "learned_correctly": all(check["passed"] for check in checks),
               "checks": checks, "scenarios": table, "comparison": comparison["summary"]}
    if filtered:
        filter_checks = learning_checks(filtered)
        summary.update({"learned_correctly_with_filter": all(c["passed"] for c in filter_checks),
                        "checks_with_filter": filter_checks,
                        "test_score_learned_with_filter": selection_score(filtered["summary"]),
                        "safety_filter": {**filtered["safety_filter"], "intervention_share": filter_share(filtered)}})
    write_json_atomic(output / "summary.json", summary)
    write_text_atomic(output / "SUMMARY.md", _summary_markdown(summary))
    print(f"\nShowcase saved in {output}. Learned correctly (policy alone): "
          f"{'YES' if summary['learned_correctly'] else 'NOT YET'}", flush=True)
    for check in checks:
        print(f"  [{'PASS' if check['passed'] else 'FAIL'}] {check['name']}: {check['detail']}", flush=True)
    if filtered:
        share = summary["safety_filter"]["intervention_share"]
        print(f"With the safety filter (teacher on {share:.3%} of drone decisions): "
              f"{'YES' if summary['learned_correctly_with_filter'] else 'NOT YET'}", flush=True)
        for check in summary["checks_with_filter"]:
            print(f"  [{'PASS' if check['passed'] else 'FAIL'}] {check['name']}: {check['detail']}", flush=True)
    return summary


def _stage_record(folder):
    settings = _read(Path(folder) / "config.json") or {}
    best = _read(Path(folder) / "best.json") or {}
    finished = _read(Path(folder) / "finished.json") or {}
    return {"run": Path(folder).name, "label": _stage_label(folder), "scenarios": settings.get("scenarios"),
            "clone_steps": settings.get("clone_steps"), "dagger_rounds": settings.get("dagger_rounds"),
            "environment_steps": finished.get("environment_steps"), "finished": bool(finished),
            "best_score": best.get("score"), "best_environment_steps": best.get("environment_steps"),
            "training_seconds": finished.get("training_seconds_total", finished.get("training_seconds_this_run"))}


def _summary_markdown(summary):
    def cell(row, key):
        value = row.get(key)
        return "-" if value is None else f"{value:.3f}" if isinstance(value, float) else str(value)

    filtered = "learned_correctly_with_filter" in summary
    lines = ["# Showcase model", "",
             f"- Model: `best.pt`, copied from `{summary['selected_run']}/best.pt`",
             f"- Trained through: {' -> '.join(summary['chain'])}",
             f"- Chosen at {summary['selected_environment_steps']:,} environment steps with validation score "
             f"{summary['validation_score']:.3f} (selection seeds 5000+)",
             f"- Held-out test seeds: {summary['test_seeds'][0]}..{summary['test_seeds'][1]}; "
             f"test score learned {summary['test_score_learned']:.3f} vs baseline {summary['test_score_baseline']:.3f}",
             f"- **Learned correctly (policy alone): {'YES' if summary['learned_correctly'] else 'NOT YET'}**"]
    if filtered:
        share = summary["safety_filter"]["intervention_share"]
        lines.append(f"- With the opt-in safety filter (the teacher takes over a drone predicted within "
                     f"{summary['safety_filter']['margin_m']:g} m of contact in the next "
                     f"{summary['safety_filter']['horizon_s']:g} s; active on {share:.3%} of drone decisions): "
                     f"**{'YES' if summary['learned_correctly_with_filter'] else 'NOT YET'}**. "
                     "This is the policy plus a safety layer, not the policy alone.")
    lines += ["", "| Stage | Scenarios | Steps at end | Best validation score | Best at step |",
              "| --- | --- | --- | --- | --- |"]
    for stage in summary.get("training_stages", []):
        steps = "-" if stage["environment_steps"] is None else f"{stage['environment_steps']:,}"
        best = "-" if stage["best_score"] is None else f"{stage['best_score']:.3f}"
        at = "-" if stage["best_environment_steps"] is None else f"{stage['best_environment_steps']:,}"
        lines.append(f"| {stage['label']} | {', '.join(stage['scenarios'] or [])} | {steps} | {best} | {at} |")
    lines += ["", "| Check (policy alone) | Result | Detail |", "| --- | --- | --- |"]
    lines += [f"| {c['name']} | {'PASS' if c['passed'] else 'FAIL'} | {c['detail']} |" for c in summary["checks"]]
    if filtered:
        lines += ["", "| Check (with safety filter) | Result | Detail |", "| --- | --- | --- |"]
        lines += [f"| {c['name']} | {'PASS' if c['passed'] else 'FAIL'} | {c['detail']} |"
                  for c in summary["checks_with_filter"]]
    header = ("| Scenario | Success (baseline / learned" + (" / filtered" if filtered else "") + ") | "
              "Slot RMSE m (baseline / learned) | Recovered events (baseline / learned) | Min clearance m (learned"
              + (" / filtered" if filtered else "") + ") |")
    lines += ["", header, "| --- | --- | --- | --- | --- |"]
    for row in summary["scenarios"]:
        b, l, f = row["baseline"], row["learned"], row.get("learned_with_filter")
        success = f"{cell(b, 'success_rate')} / {cell(l, 'success_rate')}" + (
            f" / {cell(f, 'success_rate')}" if f else "")
        clearance = cell(l, "min_surface_clearance_m") + (f" / {cell(f, 'min_surface_clearance_m')}" if f else "")
        lines.append(f"| {row['scenario']} | {success} | "
                     f"{cell(b, 'mean_slot_rmse_m')} / {cell(l, 'mean_slot_rmse_m')} | "
                     f"{b['recovered_events']}/{b['events']} / {l['recovered_events']}/{l['events']} | {clearance} |")
    lines += ["", "View it: `python -m swarmflow view` (loads this model automatically).", ""]
    return "\n".join(lines)


def default_model(models_dir=MODELS_DIR):
    """The checkpoint the viewer should show: the showcase model, else the best stage checkpoint."""
    showcase = Path(models_dir) / "showcase" / "best.pt"
    if showcase.exists():
        return showcase
    scored = []
    for run in trained_runs(models_dir):
        row = _read(run / "best.json")
        if row is not None and (run / "best.pt").exists():
            scored.append((row["score"], run))
    if not scored:
        return None
    return max(scored, key=lambda item: item[0])[1] / "best.pt"


def _thin(rows, limit=300):
    if len(rows) <= limit:
        return rows
    step = len(rows) / limit
    return [rows[int(i * step)] for i in range(limit)] + [rows[-1]]


def _history_folders(checkpoint):
    folder = Path(checkpoint).parent
    summary = _read(folder / "summary.json")
    if summary is not None:
        names = summary.get("chain", [])
        return [folder / "history" / name for name in names if (folder / "history" / name).exists()]
    return training_chain(folder)


_METADATA_CACHE = {}


def checkpoint_metadata(checkpoint):
    """Checkpoint contents without the weights, read once per file version (the viewer polls every 10 s)."""
    from .files import load_torch
    path = Path(checkpoint)
    stat = path.stat()
    key = (str(path.resolve()), stat.st_mtime_ns, stat.st_size)
    cached = _METADATA_CACHE.get(key)
    if cached is None:
        data = load_torch(path)
        cached = {k: v for k, v in data.items() if k not in ("model", "optimizer", "weights", "scales", "progress")}
        if len(_METADATA_CACHE) > 16:
            _METADATA_CACHE.clear()
        _METADATA_CACHE[key] = cached
    return cached


def model_report(checkpoint):
    """What the viewer shows about a trained model: training history, selection and held-out test."""
    checkpoint = Path(checkpoint)
    metadata = checkpoint_metadata(checkpoint)
    source = checkpoint
    if metadata.get("source_checkpoint") and local_path(metadata["source_checkpoint"]).exists():
        source = local_path(metadata["source_checkpoint"])
    training = metadata.get("training") or {}
    stages = []
    # Each stage restarts from the previous stage's best checkpoint, whose step count can be lower than
    # where that stage stopped; `x` lays the stages end to end so the curves never overlap.
    pipeline_steps = 0
    for folder in _history_folders(source):
        settings = _read(folder / "config.json") or {}
        evaluations = [{k: row.get(k) for k in ("environment_steps", "score", "best", "success_rate",
                                                 "mean_slot_rmse_m", "events", "recovered_events", "reason")}
                       for row in read_jsonl(folder / "evaluations.jsonl")]
        rows = read_jsonl(folder / "training.jsonl")
        updates = [{k: row.get(k) for k in ("environment_steps", "phase", "mean_step_reward",
                                             "recent_failure_rate", "recent_slot_rmse_m", "action_std",
                                             "environment_steps_per_second")}
                   for row in rows if row.get("type") == "ppo"]
        cloning = [row for row in rows if row.get("type") == "behavior_cloning"]
        start = settings.get("stage_start_environment_steps") or 0
        for row in evaluations + updates:
            row["x"] = pipeline_steps + row["environment_steps"] - start
        end = max((row["environment_steps"] for row in evaluations + updates), default=start)
        stages.append({"name": folder.name, "label": _stage_label(folder), "stage": settings.get("stage"),
                       "scenarios": settings.get("scenarios"), "algorithm": settings.get("algorithm"),
                       "target_environment_steps": settings.get("target_environment_steps"),
                       "ppo_steps": end - start, "finished": (folder / "finished.json").exists(),
                       "cloning": cloning, "evaluations": evaluations, "updates": _thin(updates)})
        pipeline_steps += end - start
    report = {"checkpoint": str(checkpoint), "format": metadata.get("format"), "status": metadata.get("status"),
              "precision": metadata.get("precision"),
              "environment_steps": metadata.get("environment_steps"), "pipeline_steps": pipeline_steps,
              "drones": metadata["config"]["drones"],
              "algorithm": training.get("algorithm"),
              "scenarios": training.get("scenarios"), "stage": training.get("stage"),
              "selection": metadata.get("selection"), "stages": stages, "test": None}
    summary = _read(source.parent / "summary.json")
    if summary is not None:
        report["test"] = summary
    return report


def training_status(models_dir=MODELS_DIR, active_within_s=900):
    """Progress of the most recently active stage run, shown live in the viewer while training runs."""
    runs = [folder for folder in Path(models_dir).glob("stage*-seed*")
            if re.fullmatch(r"stage\d+-seed\d+", folder.name) and (folder / "config.json").exists()]
    if not runs:
        return None
    run = max(runs, key=_touched)
    return _run_status(run, active_within_s)


def _touched(folder):
    try:
        return max((p.stat().st_mtime for p in Path(folder).iterdir() if p.is_file()), default=0.0)
    except OSError:
        return 0.0


def _run_status(run, active_within_s=900):
    settings = _read(run / "config.json")
    if settings is None:
        return None
    rows = read_jsonl(run / "training.jsonl")
    updates = [row for row in rows if row.get("type") == "ppo"]
    evaluations = read_jsonl(run / "evaluations.jsonl")
    start = settings.get("stage_start_environment_steps") or 0
    target = settings.get("target_environment_steps")
    last = updates[-1] if updates else {}
    steps = last.get("environment_steps", start)
    rate = last.get("environment_steps_per_second")
    finished = (run / "finished.json").exists()
    age = time.time() - _touched(run)
    cloning_rounds = sum(1 for row in rows if row.get("type") == "behavior_cloning")
    if finished:
        phase = "finished"
    elif last:
        phase = last.get("phase", "ppo")
    else:
        phase = "starting PPO" if evaluations else "behavior cloning"
    return {"run": str(run), "stage": settings.get("stage"), "stages": len(STAGES),
            "scenarios": settings.get("scenarios"), "phase": phase, "finished": finished,
            "active": not finished and age < active_within_s, "seconds_since_update": round(age),
            "environment_steps": steps, "start_environment_steps": start, "target_environment_steps": target,
            "progress": (steps - start) / (target - start) if target and target > start else None,
            "steps_per_second": rate,
            "eta_seconds": (target - steps) / rate if rate and target and not finished else None,
            "cloning_rounds_done": cloning_rounds,
            "cloning_rounds_total": (settings.get("dagger_rounds") or 0) + 1 if settings.get("clone_steps") else 0,
            "resumable": (run / "latest.pt").exists() and not finished,
            "recent_failure_rate": last.get("recent_failure_rate"),
            "recent_slot_rmse_m": last.get("recent_slot_rmse_m"),
            "last_score": evaluations[-1].get("score") if evaluations else None,
            "best_score": max((row["score"] for row in evaluations if row.get("score") is not None), default=None)}


def print_status(models_dir=MODELS_DIR):
    """Print each stage run's state and the commands to continue or test it (`swarmflow status`)."""
    runs = sorted(folder for folder in Path(models_dir).glob("stage*-seed*")
                  if re.fullmatch(r"stage\d+-seed\d+", folder.name))
    if not runs:
        print(f"No training runs under {models_dir}/ yet. Start one with: python -m swarmflow pipeline")
        return
    python = "python -m swarmflow"
    unfinished = False
    for run in runs:
        status = _run_status(run)
        if status is None:
            print(f"{run.name}: started but has no settings yet (it restarts when the pipeline runs)")
            continue
        progress = status["progress"]
        if status["finished"]:
            state = "finished"
        elif status["phase"] == "behavior cloning":
            state = f"behavior cloning, {status['cloning_rounds_done']}/{status['cloning_rounds_total']} rounds done"
        else:
            state = f"{status['phase']} {progress:.1%}" if progress is not None else status["phase"]
        if not status["finished"]:
            unfinished = True
            state += " (running now)" if status["active"] and status["seconds_since_update"] < 120 else \
                " (stopped; resumable)" if status["resumable"] else " (stopped before its first save; restarts)"
        best = "-" if status["best_score"] is None else f"{status['best_score']:.3f}"
        steps = status["environment_steps"] - status["start_environment_steps"]
        total = (status["target_environment_steps"] or 0) - status["start_environment_steps"]
        print(f"{run.name}: {state}; PPO steps {steps:,}/{total:,}; best validation score {best}")
        if (run / "best.pt").exists():
            scenarios = " ".join(status["scenarios"] or [])
            print(f"    test it:  {python} eval --policy learned --checkpoint {(run / 'best.pt').as_posix()} "
                  f"--scenarios {scenarios} --seeds 20 --output {(run / 'dev-evaluation.json').as_posix()}")
            print(f"    watch it: {python} view --checkpoint {(run / 'best.pt').as_posix()} --port 8766")
    showcase = Path(models_dir) / "showcase" / "summary.json"
    if showcase.exists():
        summary = _read(showcase) or {}
        print(f"showcase: learned correctly (policy alone) "
              f"{'YES' if summary.get('learned_correctly') else 'NOT YET'}"
              + (f"; with safety filter {'YES' if summary.get('learned_correctly_with_filter') else 'NOT YET'}"
                 if "learned_correctly_with_filter" in summary else ""))
    if unfinished:
        print(f"\nContinue training with the same command you started it with, e.g. {python} pipeline "
              "(or make training).")


def start_viewer(port=8765, browser=True, log=Path("runs") / "viewer-training.log"):
    """Start the viewer in its own process so it can show training live without slowing it down."""
    import socket
    import subprocess
    import sys
    import webbrowser
    url = f"http://127.0.0.1:{port}/"

    def listening():
        try:
            socket.create_connection(("127.0.0.1", port), timeout=0.3).close()
            return True
        except OSError:
            return False

    if listening():
        print(f"Port {port} is already in use; if a SwarmFlow viewer runs there it follows this training too: "
              f"{url}", flush=True)
        return None
    log.parent.mkdir(parents=True, exist_ok=True)
    with open(log, "a", encoding="utf-8") as handle:
        process = subprocess.Popen([sys.executable, "-m", "swarmflow", "view", "--port", str(port)],
                                   stdout=handle, stderr=subprocess.STDOUT)
    for _ in range(150):
        if process.poll() is not None:
            print(f"The viewer could not start (see {log}); training continues without it.", flush=True)
            return None
        if listening():
            break
        time.sleep(0.1)
    print(f"Viewer: {url}  (live training progress; it switches to each new best model automatically)",
          flush=True)
    if browser:
        webbrowser.open(url)
    return process
