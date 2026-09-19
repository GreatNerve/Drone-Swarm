# train

PPO+GAE with truncated-horizon bootstrap. CPU `EnvWorkers` (spawn, round-robin tests), net on GPU if present.

A fresh run clones `GeometricPid` cmds into the actor (`warmstart_pid`), fits the critic on that clone, then PPO. Action noise is small. A PPO pack that makes Test 1 **or** Test 2 worse, or that collides / leaves the arena, is **rejected**. Test 2 clone and train cover square, rectangle, and circle slots. Reward punishes same-altitude crowding *before* a hit so the net learns the PID altitude pass-through instead of stacking.

The live train window cycles **Test 1 / 2 / 3 at random every ~8 s** (noisy collect). After Ctrl+C, `make view-train TEST=2` is the clean look at slots.

`--resume` skips the clone and continues `latest.pt`. Old metre-trained checkpoints are invalid — start `make sim` without resume after the centimetre plant. Probe errors are centimetres. There is no 2 m PPO-stop.

The live train window is **noisy exploration**, not the PID hover. After Ctrl+C, `make view-train` is the clean look at the net.

**Train vs prod**

- Train net: `models/latest.pt` — current weights, overwritten every PPO update and on Ctrl+C. `make view-train` / `make eval MODE=train`.
- Prod net: `models/best.pt` — best combined gate score so far. `make view-prod` / `make eval MODE=prod`.

`make view` defaults to the PID baseline. Close view / stop train / `make eval` writes `results/submit/` (images, graphs, result, data, copied checkpoint).

`total_steps: 0` runs until Ctrl+C. `--resume` continues from `latest.pt`.
