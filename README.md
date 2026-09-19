# BTP1

Working repo for BTP1: decentralized drone-swarm control with reinforcement learning. Product: [docs/prd.md](./docs/prd.md). Technical: [docs/trd.md](./docs/trd.md).

## Start here

| File | What it is |
| --- | --- |
| [CONTEXT.md](./CONTEXT.md) | Domain glossary only. No implementation. |
| [AGENTS.md](./AGENTS.md) | How agents work in this repo. |
| [COMMAND.md](./COMMAND.md) | Install, command truth, local run sequence |

## Layout

```
docs/          committed, modular source of truth
no-push/       local-only (global gitignore). Extracts, scratch, token caches
graphify-out/  knowledge graph. Query it; do not re-scan the tree by default
.cursor/rules/ persistent agent rules
```

## Run

Full command list and local sequence: [COMMAND.md](./COMMAND.md).

```
make help
make sim              # Python 3D sim, trains until Ctrl+C
make view TEST=3      # drag the red target on the right panel
make view TEST=2      # 1/2/3 or click S/R/C to change geometry live
make view-train       # Tests on models/latest.pt (train dump)
make view-prod        # Tests on models/best.pt (prod dump)
make eval MODE=prod   # all 3 tests, writes results/submit
make sim-resume       # continue for hours/days
```

`total_steps: 0` in `configs/train.yaml` means no cap. Set a number to get an ETA and auto-stop.

`VIEW=python` draws `assets/drone/quadrotor.obj`. `VIEW=gazebo` needs a local Gazebo install; otherwise it falls back to Python. Physics never runs in Gazebo.

**Models:** `models/latest.pt` is the train dump (current net). `models/best.pt` is the prod dump (best gates). **Submit pack:** close a view window or run `make eval` → zip `results/submit/` (images, graphs, result.md, data, copied checkpoint).

## Agents

Read `AGENTS.md` before editing. Query `graphify-out/graph.json` before searching files. Put one-off extracts in `no-push/extracts/`. Never add AI authorship or `Co-authored-by` trailers.
