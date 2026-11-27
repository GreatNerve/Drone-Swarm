# Showcase model

- Model: `best.pt`, copied from `models/stage3-seed0/best.pt`
- Trained through: stage1-seed0 -> stage2-seed0 -> stage3-seed0
- Chosen at 1,601,536 environment steps with validation score 0.927 (selection seeds 5000+)
- Held-out test seeds: 100000..100019; test score learned 0.945 vs baseline 0.970
- **Learned correctly (policy alone): YES**
- With the opt-in safety filter (the teacher takes over a drone predicted within 0.1 m of contact in the next 1 s; active on 0.006% of drone decisions): **YES**. This is the policy plus a safety layer, not the policy alone.

| Stage | Scenarios | Steps at end | Best validation score | Best at step |
| --- | --- | --- | --- | --- |
| Stage 1 | hold, follow, turn | 1,000,000 | 0.996 | 800,768 |
| Stage 2 | hold, follow, turn, crossing, head_on | 1,800,768 | 0.931 | 1,601,536 |
| Stage 3 | hold, follow, crossing, head_on, double_crossing, turn, gust | 2,601,536 | 0.927 | 1,601,536 |

| Check (policy alone) | Result | Detail |
| --- | --- | --- |
| No collisions or other failures | PASS | 140/140 episodes completed |
| >= 95% of disturbances recovered within 5 s | PASS | 77/80 events |
| Slot RMSE below 0.35 m when undisturbed | PASS | hold 0.030 m, follow 0.029 m, turn 0.043 m |

| Check (with safety filter) | Result | Detail |
| --- | --- | --- |
| No collisions or other failures | PASS | 140/140 episodes completed |
| >= 95% of disturbances recovered within 5 s | PASS | 77/80 events |
| Slot RMSE below 0.35 m when undisturbed | PASS | hold 0.030 m, follow 0.029 m, turn 0.043 m |

| Scenario | Success (baseline / learned / filtered) | Slot RMSE m (baseline / learned) | Recovered events (baseline / learned) | Min clearance m (learned / filtered) |
| --- | --- | --- | --- | --- |
| hold | 1.000 / 1.000 / 1.000 | 0.014 / 0.030 | 0/0 / 0/0 | 0.764 / 0.764 |
| follow | 1.000 / 1.000 / 1.000 | 0.014 / 0.029 | 0/0 / 0/0 | 0.764 / 0.764 |
| crossing | 1.000 / 1.000 / 1.000 | 0.195 / 0.278 | 20/20 / 20/20 | 0.209 / 0.209 |
| head_on | 1.000 / 1.000 / 1.000 | 0.182 / 0.294 | 20/20 / 20/20 | 0.215 / 0.215 |
| double_crossing | 1.000 / 1.000 / 1.000 | 0.272 / 0.415 | 40/40 / 37/40 | 0.062 / 0.135 |
| turn | 1.000 / 1.000 / 1.000 | 0.033 / 0.043 | 0/0 / 0/0 | 0.764 / 0.764 |
| gust | 1.000 / 1.000 / 1.000 | 0.128 / 0.185 | 0/0 / 0/0 | 0.764 / 0.764 |

View it: `python -m swarmflow view` (loads this model automatically).
