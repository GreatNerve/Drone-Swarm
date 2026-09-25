# report

Writes a submission pack on view close, `make eval`, and train stop.

Also writes a **run log** after each view episode, each eval, and each train eval. Last **3** files only, in `no-push/runs/`. Each file lists per-drone `last_err_cm`: standoff-ring error on Tests 1 and 3, slot distance on Test 2. Use those when a demo looks wrong.

Public: `dump_pack(...) -> Path`, `write_run_log(...) -> Path | None`.

Layout:

```
results/submit/          zip this
  result.md              gates + where the model lives
  result.json
  images/testN.png
  graphs/distance.png
  data/trajectory.csv
  data/metrics.csv       copied from runs/ if present
  models/latest.pt or best.pt
results/archive/<stamp>_<mode>/
no-push/runs/run_*.json   last 3 view/eval logs only
```

Model dumps (training): `models/latest.pt` every PPO update; `models/best.pt` when combined gates improve.
