# Orchestrator brief (shared by all five agents)

Workflow: Agent 1 -> Agent 2 -> Agent 3 -> Agent 4 -> Agent 5, strictly sequential.
Every agent reads all earlier files in `Report/kb/` and writes its own numbered file.

Shared paths (project root = `/Users/rmayank/Documents/cursor/BTP/Drone-Swarm`):

- Knowledge base: `Report/kb/`
  - `01_project_knowledge_base.md` (Agent 1)
  - `02_theory_math_literature.md` + `references.bib` draft (Agent 2)
  - `03_visual_inventory.md` + generated figures in `Report/latex/figures/` (Agent 3)
  - `04_report_outline.md` (Agent 4)
  - `05_qa_audit.md` (Agent 5) and the LaTeX project in `Report/latex/`
- Reference (structure only) demo report: `Report/demo/7th_midterm (3).pdf`
- Reference repo context (team-written literature notes for this same project, priority 3/4):
  `Report/kb/reference_context/` (copied from github.com/Webrookie0/BEETEEPEE, branch `sumitkr`, folder `context/`).

## Report metadata (from the user; must be exact)

- Title: "Reinforcement Learning for Swarm Technologies of Drone"
- Institution: Netaji Subhas University of Technology (NSUT), New Delhi-110078
- Programme: B.Tech.
- Department: to be verified from nsut.ac.in. Roll numbers are 2023UEC..., and the demo report (same
  university, same UEC roll prefix) uses "Department of Electronics & Communication Engineering" and
  "B.Tech. in Electronics & Communication Engineering".
- Supervisor: Prof. Jyotsna Singh (designation to be verified)
- Team: Nishant Raj (2023UEC2606), Sumit Kumar (2023UEC2554), Dheeraj Sharma (2023UEC2599),
  Tarun Kumar (2023UEC2551)
- Page limit: 20-25 pages total.

## Facts the orchestrator already verified

- Local folder == GitHub `GreatNerve/Drone-Swarm` branch `nishant` (HEAD `bf4eb92`, identical to remote).
  Other branches: `sumit` (adds run 3 models, `f66c7f6`), `dheeraj`, `Tarun`, `main` (older). No branch
  contains a completed pipeline-v4 training run.
- Project name in code: **SwarmFlow** ("follow, yield, recover"). Python 3.12, NumPy, Gymnasium,
  PettingZoo, PyTorch (torch 2.14.0 in `.venv`). No matplotlib in `.venv`; no LaTeX installed on this Mac.
- Existing detailed internal docs: `knowledge.md`, `sofar.md`, `training.md`, `guide.md`, `README.md`,
  `docs/implementation.md`, `docs/research-plan.md`, `docs/validation.md`, `presentation.md`.
- Code checked by the orchestrator against docs (all consistent): `env.py` (reward, dynamics, swept
  collisions, recovery), `config.py` (defaults), `policy.py` (SetEncoder / ActorCritic, width 128 in
  pipeline >= v2), `baseline.py` (PredictiveController, SlotController), `train.py` (DART+DAgger clone,
  Worker with crash replay, GAE, RewardScaler, PPO loss with imitation term, KL early stop),
  `curriculum.py` (STAGES, COMMON, showcase), `evaluate.py` (selection_score, learning_checks).
- **Best trained model = run 3 (pipeline v3)**: `models/previous-runs/run3-pipeline-v3/showcase/best.pt`.
  Held-out test (7 scenarios x 20 seeds, 100000-100019): 138/140 episodes without failure, score
  0.936 vs baseline 0.970; slot RMSE hold/follow/turn 0.023/0.027/0.045 m; 70/80 events recovered
  within 5 s (78/80 under the v4 overlap rule, per docs); 2 failures = head_on grazes by one drone.
  Built-in verdict: "NOT LEARNED YET" (zero-failure check fails). This must be reported honestly.
- `models/showcase/` (what `make run` loads by default) still holds **run 2** (148/160, score 0.815).
  To demonstrate run 3 in the viewer: `.venv/bin/python -m swarmflow view --checkpoint
  models/previous-runs/run3-pipeline-v3/showcase/best.pt --port 8766`.
- Safety-filter ("shield") experiment `runs/experiments/shield.py` + logs: 0/140 failures on
  development seeds 1000-1019 with 0.1 m / 1 s threshold; teacher used on 0.01% of drone-steps.
  It is an experiment, not part of the benchmark; must be labelled as such.
- Pipeline v4 (current code, `c4f7249`) is implemented and unit-tested but its full training run has
  NOT been performed. v4 must not be reported as a trained result.
- `models/best.pt` / `models/latest.pt` directly in `models/` were not produced by this code (per
  knowledge.md 10.6); do not present them.

## Reporting stance (user's instruction)

Focus on the final working system and its demonstrated behaviour; do not turn the report into a diary
of failed versions. Never fabricate success. The honest framing: the learned decentralized policy
flies the full benchmark with 98.6% (138/140) failure-free held-out episodes and sub-5 cm formation
error, close to the hand-coded predictive baseline, but it does not yet meet the strict zero-failure
acceptance criterion; a lightweight safety filter closes that gap on development seeds.
