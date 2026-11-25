# Training guide: how to train, store and show the model

This guide lists the exact commands to train the drone policy yourself, store the results, and show the best model in the viewer. It also lists what **not** to do. [training.md](training.md) explains how the training works and why.

Run every command from the project folder (`BTP/Drone-Swarm`). The training runs the same way on macOS, Windows and Linux.

**Windows without `make`:** every `make` command below has a Python equivalent; run it with `.venv\Scripts\python` instead of `.venv/bin/python` (PowerShell or Command Prompt), for example:

```bat
.venv\Scripts\python -m swarmflow pipeline        :: make training
.venv\Scripts\python -m swarmflow status          :: make status
.venv\Scripts\python -m swarmflow quantize        :: make quantize
.venv\Scripts\python -m pytest -q                 :: make test
```

`make` itself also works from Git Bash once it is installed (for example `choco install make` or `winget install GnuWin32.Make`).

---

## 1. One-time setup

```sh
make install          # uv sync --python 3.12 --extra dev --cache-dir .uv-cache
make test             # under a minute; every test must pass before you train
```

If Python 3.12 is not installed, use the version you have (3.11 to 3.13 work): `make install PY_VERSION=3.11`, or `uv sync --python 3.11 --extra dev`.

## 2. Train: one command

There are two ways to start the full training. Both run the same training and store the same results.

| Command | What it does | Python equivalent |
| --- | --- | --- |
| `make training` | Full training in the terminal: stage 1 → 2 → 3, then the showcase test. | `.venv/bin/python -m swarmflow pipeline` |
| `make run train` (same as `make run-train`) | The same training, **plus the viewer** in your browser. It shows live progress, flies the best model found so far, and switches to the final best model at the end. | `.venv/bin/python -m swarmflow pipeline --view` |

The training keeps the computer awake by itself while it runs (macOS, Windows and Linux), so `caffeinate` is no longer needed; pass `--allow-sleep` to turn that off. Keep a laptop on its charger.

### What happens, in order

1. **Stage 1, formation keeping** (hold, follow, turn), saved to `models/stage1-seed0/`:
   - behavior cloning from the hand-coded controller, with 3 DAgger rounds;
   - 1,000,000 PPO steps.
2. **Stage 2, moving obstacles** (adds crossing and head-on objects), saved to `models/stage2-seed0/`:
   - starts from stage 1's `best.pt`;
   - behavior cloning + 3 DAgger rounds on these obstacle scenarios, so the network learns avoidance from the teacher;
   - 1,000,000 PPO steps with an imitation loss (fine-tuning; the best of the cloned start and the PPO checkpoints is kept).
3. **Stage 3, the full benchmark** (adds repeated crossings and wind), saved to `models/stage3-seed0/`: the same recipe, starting from stage 2's `best.pt`, 1,000,000 PPO steps. Slalom, with static spheres on the formation slots, is not part of the learned-policy benchmark; see [training.md](training.md) Part D.
4. **Showcase**:
   - picks the best final model (across all trained seeds);
   - tests it against the hand-coded baseline on 7 scenarios × 20 seeds it has never seen;
   - tests the same model again behind the optional safety filter (section 4), reported separately;
   - stores everything in `models/showcase/`.
5. **Quantization**: small copies of the showcase model in `models/showcase/quantized/` (section 5).

Every 100,000 steps the model is validated on 20 seeds per scenario. Whenever it beats its previous best it is saved as `best.pt`; the score asks for at least 0.15 m clearance from moving objects, not only for no crash (training.md Part F). Each stage continues from the previous stage's `best.pt`.

### What you see while it runs

- **Terminal:** one line per update:
  - progress %, speed and ETA;
  - reward, share of failed episodes and slot error (last 100 episodes);
  - KL and value loss.

  Every validation prints its score and says when a new `best.pt` was saved.
- **Browser (`make run train`):** the section **"Did the model learn correctly?"** refreshes every 10 s:
  - a progress bar (stage, %, steps/s, time left in the stage, last and best validation score);
  - training curves for every stage;
  - the drones flown by the best model saved so far.

  Press **Run** to watch them. After the pipeline finishes, the page shows the final verdict and the learned-vs-baseline table, and the viewer keeps running until you press Ctrl+C.

### Time (run 3 measured about 550 PPO steps per second on a Mac; slower laptops take longer)

| Part | Time |
| --- | --- |
| Stage 1 (cloning ~8 min + 1M PPO steps) | about 40 min |
| Stage 2 (cloning + DAgger on obstacle scenarios ~30 min + 1M PPO steps) | about 1 h 10 min |
| Stage 3 (the same on all 7 scenarios) | about 1 h 15 min |
| Showcase test (policy alone and with the safety filter) + quantization | about 20 min |
| **Total** | **about 3.5–4 h** (a slower laptop can take twice as long) |

You do not have to train it in one go: see "Stop, test, continue" below.

This is version 5 of the training. The earlier versions are explained in [training.md](training.md):

- **Part C:** the fixes after the first run.
- **Part D:** after the second run, a cautious teacher and slalom removed.
- **Part E:** after the third run, a teacher margin of 0.7 / 0.3 m, recovery counted after overlapping objects, and 1M PPO steps in stages 2–3 (was 2M).
- **Part F:** crash-proof long runs, exact stop and continue, a clearance-aware choice of `best.pt` on 20 seeds, the safety filter and quantization.

When you start it, the older runs' folders are moved to `models/<name>-old-<date>` automatically, and everything trains again. Nothing is deleted.

### Stop, test, continue

You can train for an hour, look at the model, and train another hour, as often as you like.

1. **Stop:** press **Ctrl+C** once. The training finishes the update it is on (a few seconds; during behavior cloning, the current round, up to a few minutes), saves, and exits with "Stopped after the last update ... run the same command to continue". Pressing Ctrl+C a second time exits at once; the last saved update is kept.
2. **Look:** `make status` shows every stage, how far it got and its best validation score, and prints the exact commands to test and watch its `best.pt`:

   ```sh
   make status                       # python -m swarmflow status
   make eval-stage1                  # 20 development seeds; or the eval command printed by status
   make run                          # the viewer picks the best model so far
   ```

3. **Continue:** run the **same command** you started with (`make training`, `make run train`, `make train-stageN` or the Python equivalents). It continues exactly where it stopped:
   - finished stages are skipped;
   - an unfinished stage continues from its `latest.pt` with its original settings, the same learning-rate and imitation schedules, its counters, its random state and its saved teacher data; a stop during behavior cloning continues at the next cloning round;
   - only the few episodes that were in flight when it stopped are started again.

A crash, a power cut, closing the terminal or the laptop sleeping loses at most the update (or cloning round) that was in progress; run the same command again. Checkpoints and logs are written so that an interruption can never corrupt them, and a file briefly locked by the viewer, an antivirus scanner or cloud sync (common on Windows) is retried instead of stopping the run.

### More seeds (recommended for the report)

```sh
make train-full-multi                      # seeds 0 1 2, about 12 h  (python -m swarmflow pipeline --seeds 0 1 2)
make training SEED=1                       # or add one seed at a time
```

The showcase always keeps the best seed, which is the most direct way to pass the zero-failure check (one graze in 140 episodes fails it). Report the others too, because RL results vary between seeds. Thanks to stop and continue, the 12 hours can be spread over several sessions.

## 3. Where the results are stored

```text
models/
  stage1-seed0/ stage2-seed0/ stage3-seed0/   one folder per stage and seed
    best.pt               best validated checkpoint of that stage
    latest.pt             last update, only for continuing an interrupted run
    checkpoints/          snapshots every 500,000 steps
    training.jsonl        every cloning round and PPO update
    evaluations.jsonl     every validation (the learning curve)
    best.json             metrics of best.pt
    config.json           all settings;  finished.json appears when the stage is done
    teacher-data/         the cloning data, kept only until the stage finishes (for an exact continue)
  showcase/                                   THE RESULT used by the viewer and the presentation
    best.pt               the selected model
    SUMMARY.md            verdict (policy alone, and with the safety filter), per-stage table, checks, learned vs baseline
    summary.json          the same, machine-readable
    test_learned.json, test_learned_filter.json, test_baseline.json, test_compare.json   held-out test reports
    history/              training logs of every stage behind the selected model
    quantized/            fp32 / fp16 / int8 / int8-dynamic copies, QUANTIZATION.md and report.json
  showcase-archive/<date>/                    the previous showcase, each time a new one is built
```

`models/best.pt` and `models/latest.pt` directly inside `models/` were not produced by this code, and the viewer ignores them.

## 4. Show the result

```sh
make run              # or double-click "Launch SwarmFlow.command"   (python -m swarmflow view)
```

Open <http://127.0.0.1:8765>. The viewer uses the best model automatically: `models/showcase/best.pt`, or, before the showcase exists, the best stage checkpoint. It keeps checking `models/`, so a newer best model is picked up without restarting.

1. **Active controller** reads `learned checkpoint`, and the yellow note shows which file is flying.
2. Pick a scenario and press **Run**. Only the network flies the drones; no baseline or safety filter runs behind it.
3. Switch **Controller** to *Predictive baseline* and run the same seed to compare. *Slot-only* shows what happens with no avoidance.
4. **"Did the model learn correctly?"**:
   - **Verdict**: `LEARNED CORRECTLY` only if every acceptance check passed on the held-out seeds (no failures; at least 95% of disturbances recovered within 5 s; slot error below 0.35 m on hold, follow and turn).
   - **With safety filter**: the verdict of the same model behind the safety filter (below), and how often the filter took over.
   - **Curves**: a rising validation score and falling failures show learning; ◆ marks the saved `best.pt`.
   - **Table**: learned policy vs baseline on the held-out seeds.

Other viewer commands: `make view-baseline` (no trained model), and `python -m swarmflow view --checkpoint <file> --port 8766` for one fixed checkpoint (a quantized `model.pt` works too).

### The safety filter (optional, reported separately)

The policy alone has to pass the verdict with no help. Separately, the showcase also tests it behind a **safety filter**: before each step, every drone's commanded motion is predicted 1 s ahead; if it would come within 0.1 m of a neighbour or an object, that drone flies the hand-coded controller for 0.5 s instead. In development tests it took over 0.04–0.06% of drone decisions and removed the remaining graze (0/140 failures, 80/80 disturbances recovered). A result with the filter is "policy + safety layer", never "the policy learned it"; report both lines from `SUMMARY.md`.

```sh
make eval-filter      # python -m swarmflow eval --policy learned --safety-filter --checkpoint models/showcase/best.pt ...
make view-filter      # python -m swarmflow view --safety-filter --checkpoint models/showcase/best.pt
```

`--filter-margin` and `--filter-horizon` change the 0.1 m and 1 s.

## 5. Quantized models

After the showcase, the pipeline builds small copies of `models/showcase/best.pt` in `models/showcase/quantized/` and checks each one against the original. They keep only the network part that flies the drones (no critic, no optimizer state), so even `fp32` is about 6 times smaller than the checkpoint.

```sh
make quantize                                            # the default presets below
make quantize VARIANTS="int8 int8-per-tensor"            # choose presets
python -m swarmflow quantize --list                      # all presets
python -m swarmflow quantize --checkpoint models/stage2-seed0/best.pt --seeds 20 --safety-filter
python -m swarmflow quantize --seed-start 100000         # check on the held-out seeds for the report
```

| Preset | What it stores | Size (run 3 model) |
| --- | --- | --- |
| `fp32` (default) | the actor in float32 | about 1.1 MB (17% of the checkpoint) |
| `fp16` (default) | half-precision weights, computed in float32 | about 0.55 MB |
| `int8` (default) | 8-bit weights, one scale per output channel, computed in float32 | about 0.3 MB |
| `int8-dynamic` (default) | the same 8-bit weights run by PyTorch's dynamic INT8 kernels (inputs quantized too); falls back to `int8` behaviour where the machine has no quantized engine | about 0.3 MB |
| `int8-per-tensor` | 8-bit weights with one scale per matrix | about 0.3 MB |
| `int8-keep-head` | like `int8`, but the final action layer stays float32 | about 0.3 MB |

`QUANTIZATION.md` lists, for the original and every preset: file size, the largest change in the commanded velocity, time per team step, and the flight check on 7 scenarios × 20 development seeds (failures, recovered disturbances, slot error, clearance and the three verdict checks). Every file loads on any computer (`torch.load(..., weights_only=True)`), and a preset is used like any checkpoint:

```sh
python -m swarmflow view --checkpoint models/showcase/quantized/int8/model.pt --port 8766
python -m swarmflow eval --policy learned --checkpoint models/showcase/quantized/int8/model.pt --seeds 20
```

Quantized files are for flying and testing only; they cannot be trained further. Skip the step with `python -m swarmflow pipeline --no-quantize`.

## 6. Stage by stage (optional)

These use the same settings as `make training`:

```sh
make train-stage1     make eval-stage1        # python -m swarmflow train --stage 1
make train-stage2     make eval-stage2        # starts from models/stage1-seed0/best.pt
make train-stage3     make eval-stage3        # starts from models/stage2-seed0/best.pt
make showcase                                 # choose, test on held-out seeds, store in models/showcase
make continue-stage2                          # finish an interrupted stage (re-running make train-stage2 does the same)
```

The numbered tests `make test-9`, `test-10` and `test-11` compare each stage's `best.pt` with the baseline on the same 20 seeds. They reuse trained models and only train when one is missing.

## 7. Do and don't

**Do**

- Run `make test` once before a long training run.
- Keep a laptop on its charger (sleep is blocked automatically during training).
- Stop with Ctrl+C whenever you like and re-run the same command; it continues exactly where it stopped.
- Judge the model by the **viewer verdict**, the **validation curve** and `models/showcase/SUMMARY.md`.
- Train more than one seed for the report (`make train-full-multi`).
- Zip `models/showcase/` to share the result; it holds the model, its test reports and its training history.

**Don't**

- **Don't delete or edit files in a `models/stageN-seedS/` folder while it trains.** `latest.pt` is needed to continue, and `config.json` and `best.json` drive the resume and the model choice.
- **Don't run two trainings into the same folder at the same time**, for example `make training` in two terminals. Train different seeds instead (`SEED=1`).
- **Don't change `--drones`, `--duration`, `--noise` or `--dropout` between stages.** A stage can only continue from a checkpoint with the same environment settings.
- **Don't start a stage before the previous one has finished.** Stage 2 starts from stage 1's `best.pt`; `make training` handles the order for you.
- **Don't judge the model by training reward or loss alone.** Training uses exploration noise; the validation score and the held-out verdict are the real measures.
- **Don't tune anything on the held-out test seeds (100000+).** They are only for the final showcase test; use seeds 1000+ (`make eval-stageN`) while experimenting.
- **Don't try to resume old checkpoints from `runs/`** (format `swarmflow-v2`). They were trained with the old broken reward and network. They can still be *viewed* with `--checkpoint`, but not continued.
- **Don't shorten the stage budgets to make training finish faster.** The earlier 2–5 minute runs are exactly what failed (see section 2 of training.md).
- **Don't show `models/best.pt` directly inside `models/` as our result.** It was not produced by this code.
- **Don't present the with-filter verdict as the policy's own result.** It is the policy plus a safety layer.

## 8. Troubleshooting

| Message | Meaning and fix |
| --- | --- |
| `... already holds a trained model; use --continue` | That folder already has a run. Use `make training` or `make train-stageN` (both continue automatically), `make continue-stageN`, or another `--output`. |
| `Warning: could not save latest.pt (...)` | Another program held the file for over a minute (antivirus, cloud sync, a file viewer). Training keeps going and saves again after the next update; close that program if it repeats. |
| `PPO produced non-finite values in 3 updates in a row` | The update was rolled back twice with a halved learning rate and still failed. `latest.pt` holds the last good update; run the same command to continue. |
| `... is an actor-only (deployment or quantized) model` | Quantized files cannot be trained; continue from the stage's `latest.pt` or `best.pt`. |
| `No interpreter found for Python 3.12` (make install) | Use the installed version: `make install PY_VERSION=3.11`. |
| `Stage 2 starts from models/stage1-seed0/best.pt; train stage 1 first` | Run the stages in order, or just use `make training`. |
| `... is a swarmflow-v2 checkpoint trained with the old reward and network` | An old run cannot be resumed; train fresh. |
| `Resume checkpoint has incompatible environment configuration` | The environment settings differ from the checkpoint; use the same ones in every stage. |
| `Port 8765 is already in use` (with `make run train`) | A viewer is already running; it follows the training too. Or close it and run again. |
| Verdict `NOT LEARNED YET` | Read the failing check and the curves, then see section 6.3 of [training.md](training.md). |
