import argparse
import sys
from dataclasses import replace
from pathlib import Path

from .config import Config
from .scenarios import SCENARIOS

TRAIN_OPTIONS = ("total_steps", "horizon", "epochs", "seed", "scenarios", "clone_steps", "clone_epochs",
                 "dagger_rounds", "clone_noise", "resume", "device", "output", "critic", "separate_critic",
                 "failure_reset", "envs", "lr", "minibatch", "entropy", "target_kl", "init_log_std",
                 "critic_warmup", "eval_every", "eval_seeds", "eval_scenarios", "snapshot_every", "threads",
                 "continue_run", "width", "min_log_std", "bc_coef", "bc_coef_end", "teacher_margins", "hard_weight",
                 "gap_features")


def _safe_output():
    """Never crash on a character the console or a redirected log cannot encode (Windows cp1252)."""
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(errors="replace")
        except (AttributeError, ValueError):
            pass


def _filter_options(args):
    return {key: value for key, value in (("margin", args.filter_margin), ("horizon", args.filter_horizon))
            if value is not None}


def main():
    _safe_output()
    parser = argparse.ArgumentParser(description="SwarmFlow: target following and formation recovery")
    commands = parser.add_subparsers(dest="command", required=True)
    view = commands.add_parser("view", help="Open an interactive local flight laboratory")
    view.add_argument("--port", type=int, default=8765)
    view.add_argument("--checkpoint", help="Trained policy to load (default: models/showcase/best.pt, "
                                           "else the best stage model under models/); quantized files work too")
    view.add_argument("--baseline-only", action="store_true", help="Do not load any trained policy")
    view.add_argument("--safety-filter", action="store_true",
                      help="Fly the learned policy behind the safety filter (reported separately from the policy)")
    evaluation = commands.add_parser("eval", help="Run seeded benchmark scenarios")
    evaluation.add_argument("--policy", choices=("baseline", "slot", "learned"), default="baseline")
    evaluation.add_argument("--checkpoint")
    evaluation.add_argument("--seeds", type=int, default=3, help="Number of episodes per scenario")
    evaluation.add_argument("--seed-start", type=int, default=1000)
    evaluation.add_argument("--scenarios", nargs="+", choices=SCENARIOS, default=list(SCENARIOS))
    evaluation.add_argument("--output", default="runs/evaluation.json")
    evaluation.add_argument("--trace", action="store_true")
    evaluation.add_argument("--obstacles", type=int)
    evaluation.add_argument("--rays", type=int)
    evaluation.add_argument("--ray-range", type=float, dest="ray_range")
    for command in (view, evaluation):
        command.add_argument("--filter-margin", type=float,
                             help="Safety filter: take over below this predicted surface gap in m (default 0.1)")
        command.add_argument("--filter-horizon", type=float,
                             help="Safety filter: prediction horizon in s (default 1.0)")
    evaluation.add_argument("--safety-filter", action="store_true",
                            help="Run the controller behind the safety filter and report how often it intervened")
    compare = commands.add_parser("compare", help="Diff success rate and slot RMSE from two reports")
    compare.add_argument("report1")
    compare.add_argument("report2")
    compare.add_argument("--output", default="runs/compare.json")
    suite = commands.add_parser("test", help="Pick and run numbered validation tests (interactive if none given)")
    suite.add_argument("tests", nargs="*", help="Test numbers, ranges or names, e.g. `2 3`, `2-4`, `baseline`, `all`")
    suite.add_argument("--list", action="store_true", help="Show the test catalog and exit")
    suite.add_argument("--episodes", type=int, help="Override the number of seeds per scenario")
    suite.add_argument("--seed", type=int, default=0, help="Training seed for tests 8-11")
    suite.add_argument("--retrain", action="store_true", help="Train again even if a checkpoint exists")
    suite.add_argument("-k", dest="keyword", help="Only run pytest checks matching this keyword (test 1)")

    training = commands.add_parser("train", help="Train the shared drone policy (behavior cloning + PPO)")
    training.add_argument("--stage", type=int, choices=(1, 2, 3),
                          help="Use the curriculum preset for this stage (see training.md); other flags override it. "
                               "Re-running continues an unfinished stage")
    training.add_argument("--continue", dest="continue_run", action="store_true", default=None,
                          help="Finish the interrupted run in --output from its latest.pt, with its own settings")
    training.add_argument("--steps", dest="total_steps", type=int,
                          help="Environment steps in this run, summed over the parallel environments")
    training.add_argument("--horizon", type=int, help="Rollout steps per environment before each PPO update")
    training.add_argument("--epochs", type=int, help="PPO passes over each rollout (stopped early by --target-kl)")
    training.add_argument("--seed", type=int)
    training.add_argument("--scenarios", nargs="+", choices=SCENARIOS)
    training.add_argument("--clone-steps", type=int,
                          help="Environment steps per behavior-cloning round; 0 means PPO from random weights")
    training.add_argument("--clone-epochs", type=int)
    training.add_argument("--dagger-rounds", type=int, help="Extra cloning rounds flown by the student itself")
    training.add_argument("--clone-noise", type=float, help="Action noise injected while the teacher flies")
    training.add_argument("--resume", help="Start from this checkpoint (weights, optimizer, reward scale)")
    training.add_argument("--device", choices=("cpu", "mps", "cuda"))
    training.add_argument("--output")
    training.add_argument("--critic", choices=("local", "central"))
    training.add_argument("--shared-critic", dest="separate_critic", action="store_false", default=None,
                          help="Let the critic share the actor encoder (the old, unstable setup)")
    training.add_argument("--failure-reset", action="store_true", default=None)
    training.add_argument("--envs", type=int, help="Parallel environments per rollout")
    training.add_argument("--lr", type=float, help="Adam learning rate, annealed linearly to 10%%")
    training.add_argument("--minibatch", type=int)
    training.add_argument("--entropy", type=float)
    training.add_argument("--target-kl", type=float)
    training.add_argument("--init-log-std", type=float)
    training.add_argument("--min-log-std", type=float, help="Floor on the exploration noise (log standard deviation)")
    training.add_argument("--width", type=int, help="Network width (64 = the original network; stages use 128)")
    training.add_argument("--bc-coef", type=float, help="Imitation-loss weight at the start of the run")
    training.add_argument("--bc-coef-end", type=float, help="Imitation-loss weight at the end of the run")
    training.add_argument("--teacher-margins", type=float, nargs=2, metavar=("SOFT", "HARD"),
                          help="Teacher clearance margins in m while cloning (stages use 0.7 0.3; baseline uses 0.35 0.08)")
    training.add_argument("--hard-weight", type=float,
                          help="How much more often near-object states are sampled for imitation (default 1: off)")
    training.add_argument("--gap-features", dest="gap_features", action="store_true", default=None,
                          help="Give the network each neighbour's and object's surface gap and closing speed")
    training.add_argument("--critic-warmup", type=int, help="First PPO updates that train only the critic")
    training.add_argument("--eval-every", type=int, help="Validate (and maybe save best.pt) every N environment steps")
    training.add_argument("--eval-seeds", type=int, help="Selection episodes per scenario at each validation")
    training.add_argument("--eval-scenarios", nargs="+", choices=SCENARIOS)
    training.add_argument("--snapshot-every", type=int, help="Keep checkpoints/step_N.pt every N environment steps")
    training.add_argument("--threads", type=int, help="CPU threads for PyTorch (default: up to 4)")

    pipeline = commands.add_parser("pipeline", help="Run the full curriculum (stages 1-3), build the showcase and "
                                                    "quantize it; re-run the same command to continue after a stop")
    pipeline.add_argument("--seeds", type=int, nargs="+", default=[0], help="Independent training seeds")
    pipeline.add_argument("--stages", type=int, nargs="+", choices=(1, 2, 3), default=[1, 2, 3])
    pipeline.add_argument("--models", default="models", help="Folder that receives stageN-seedS and showcase")
    pipeline.add_argument("--test-seeds", type=int, default=20, help="Held-out test episodes per scenario")
    pipeline.add_argument("--no-showcase", action="store_true")
    pipeline.add_argument("--no-quantize", action="store_true", help="Do not build the quantized models at the end")
    pipeline.add_argument("--device", default="cpu", choices=("cpu", "mps", "cuda"))
    pipeline.add_argument("--threads", type=int)
    pipeline.add_argument("--view", action="store_true",
                          help="Also open the viewer: live training progress, then the best model")
    pipeline.add_argument("--port", type=int, default=8765, help="Viewer port for --view")
    pipeline.add_argument("--no-browser", action="store_true", help="With --view, do not open a browser window")
    for command in (training, pipeline):
        command.add_argument("--allow-sleep", action="store_true",
                             help="Let the computer sleep during training (by default sleep is blocked)")

    showcase = commands.add_parser("showcase", help="Pick the best trained run, test it on held-out seeds against "
                                                    "the baseline and store it in models/showcase")
    showcase.add_argument("runs", nargs="*", help="Run folders to choose from (default: every seed of the "
                                                  "highest trained stage)")
    showcase.add_argument("--output", default="models/showcase")
    showcase.add_argument("--seeds", type=int, default=20, help="Held-out test episodes per scenario")
    showcase.add_argument("--seed-start", type=int, default=100000)
    showcase.add_argument("--no-filter", action="store_true", help="Skip the test behind the safety filter")

    status = commands.add_parser("status", help="Show every training run's progress and how to continue or test it")
    status.add_argument("--models", default="models")

    quantize = commands.add_parser("quantize", help="Build small actor-only models (fp32, fp16, int8) from a "
                                                    "checkpoint and compare them with the original")
    quantize.add_argument("--checkpoint", default="models/showcase/best.pt")
    quantize.add_argument("--variants", nargs="+", help="Presets to build (default: fp32 fp16 int8 int8-dynamic); "
                                                         "see --list")
    quantize.add_argument("--list", action="store_true", help="Show the available presets and exit")
    quantize.add_argument("--output", help="Folder for the variants (default: <checkpoint folder>/quantized)")
    quantize.add_argument("--seeds", type=int, default=20, help="Evaluation episodes per scenario (0: skip)")
    quantize.add_argument("--seed-start", type=int, default=1000,
                          help="First evaluation seed (development seeds 1000+; use 100000 for final reporting)")
    quantize.add_argument("--scenarios", nargs="+", choices=SCENARIOS, help="Default: the 7 showcase scenarios")
    quantize.add_argument("--safety-filter", action="store_true", help="Also evaluate each variant behind the filter")

    for command in (view, evaluation, training, pipeline):
        command.add_argument("--drones", type=int)
        command.add_argument("--duration", type=float)
        command.add_argument("--noise", type=float)
        command.add_argument("--dropout", type=float)
    args = parser.parse_args()
    try:
        if args.command == "compare":
            from .evaluate import compare_reports
            compare_reports((args.report1, args.report2), args.output)
            print(f"Saved comparison to {args.output}")
            return
        if args.command == "test":
            from .suite import Options, print_catalog, run_tests
            if args.list:
                print_catalog()
                return
            if args.episodes is not None and args.episodes < 1:
                parser.error("--episodes must be positive")
            raise SystemExit(run_tests(args.tests, Options(args.episodes, args.seed, args.retrain, args.keyword)))
        if args.command == "showcase":
            from .curriculum import build_showcase
            if args.seeds < 1:
                parser.error("--seeds must be positive")
            build_showcase(args.runs or None, args.output, args.seeds, args.seed_start,
                           safety_filter=not args.no_filter)
            return
        if args.command == "status":
            from .curriculum import print_status
            print_status(args.models)
            return
        if args.command == "quantize":
            from .quantize import DEFAULT_VARIANTS, print_variants, quantize_checkpoint
            if args.list:
                print_variants()
                return
            if args.seeds < 0:
                parser.error("--seeds must be zero or positive")
            quantize_checkpoint(args.checkpoint, args.variants or DEFAULT_VARIANTS, args.output, args.seeds,
                                args.seed_start, args.scenarios, args.safety_filter)
            return
        overrides = {key: getattr(args, key) for key in
                     ("drones", "duration", "noise", "dropout", "obstacles", "rays", "ray_range")
                     if getattr(args, key, None) is not None}
        if args.command == "pipeline":
            from .awake import keep_awake
            from .curriculum import run_pipeline, start_viewer
            extra = {"device": args.device} | ({"threads": args.threads} if args.threads else {})
            viewer = start_viewer(args.port, not args.no_browser) if args.view else None
            try:
                with keep_awake(not args.allow_sleep):
                    run_pipeline(args.seeds, sorted(set(args.stages)), args.models,
                                 replace(Config(), **overrides) if overrides else None,
                                 not args.no_showcase, args.test_seeds, not args.no_quantize, **extra)
                if viewer is not None and viewer.poll() is None:
                    print(f"\nTraining finished. The viewer at http://127.0.0.1:{args.port} now flies the best model "
                          "(models/showcase/best.pt). Press Ctrl+C to stop it.", flush=True)
                    viewer.wait()
            finally:
                if viewer is not None and viewer.poll() is None:
                    viewer.terminate()
            return
        if args.command == "train":
            from .awake import keep_awake
            with keep_awake(not args.allow_sleep):
                train_command(args, overrides)
            return
        from .policy import LearnedController
        path = args.checkpoint
        if args.command == "view" and not path and not args.baseline_only:
            from .curriculum import default_model
            found = default_model()
            path = str(found) if found else None
            print(f"Trained policy: {path}" if path else
                  "No trained policy found under models/; showing the predictive baseline only", flush=True)
        config = Config()
        if path and not (args.command == "view" and overrides and not args.checkpoint):
            config = LearnedController(path).config
        config = replace(config, **overrides)
        if args.command == "view":
            if args.checkpoint and overrides:
                parser.error("For learned viewing, use checkpoint configuration; change swarm size in the viewer")
            from .server import serve
            serve(config, args.port, None if args.baseline_only else path,
                  follow=not args.checkpoint and not args.baseline_only,
                  safety_filter=args.safety_filter and bool(path) and not args.baseline_only)
        else:
            if args.seeds < 1 or args.seed_start < 0:
                parser.error("Seed count must be positive and seed start nonnegative")
            if args.policy != "learned" and args.checkpoint:
                parser.error("--checkpoint requires --policy learned")
            from .evaluate import evaluate
            report = evaluate(config, args.scenarios, range(args.seed_start, args.seed_start + args.seeds),
                              args.policy, args.checkpoint, args.output, args.trace,
                              args.safety_filter, _filter_options(args))
            print(f"Saved {len(report['episodes'])} episodes to {args.output}")
    except (ValueError, FileNotFoundError) as error:
        parser.error(str(error))
    except KeyboardInterrupt:
        raise SystemExit(130)


def train_command(args, overrides):
    from .policy import LearnedController
    from .train import train
    options = {}
    if args.stage:
        from .curriculum import stage_arguments
        options = stage_arguments(args.stage, args.seed if args.seed is not None else 0)
    options.update({key: getattr(args, key) for key in TRAIN_OPTIONS if getattr(args, key, None) is not None})
    options.setdefault("scenarios", ["hold", "follow"])
    options.setdefault("output", "runs/train")
    folder = Path(options["output"])
    if args.stage and not options.get("continue_run"):
        from .curriculum import PIPELINE_VERSION, _archive, _pipeline_version
        if _pipeline_version(folder) not in (None, PIPELINE_VERSION):
            _archive(folder, "trained by an older pipeline version")
        if (folder / "finished.json").exists():
            print(f"Stage {args.stage} already finished in {folder}; nothing to do. Move that folder away to "
                  "train it again.")
            return
        if (folder / "latest.pt").exists():
            print(f"Continuing the interrupted run in {folder}", flush=True)
            options["continue_run"] = True
    if options.get("continue_run"):
        source = folder / "latest.pt"
        options.pop("resume", None)
    else:
        source = options.get("resume")
    config = LearnedController(source).config if source else Config()
    config = replace(config, **overrides)
    scenarios = options.pop("scenarios")
    latest = train(config, scenarios, **options)
    best = latest.with_name("best.pt")
    print(f"Latest checkpoint: {latest}\nBest validated checkpoint: {best}\n"
          f"Test it on held-out seeds with `python -m swarmflow showcase {latest.parent.as_posix()}`.")
