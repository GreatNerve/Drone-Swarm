# Quantized models

- Source: `models/showcase/best.pt`
- Action error: largest absolute difference from the original's command (each axis in [-1, 1]) over 13,440 drone states
- Latency: one 8-drone team step on this CPU with 4 PyTorch thread(s)
- Flight check: 7 scenarios x 20 seeds (1000..1019); verdict = the three acceptance checks

| Variant | Size | vs checkpoint | Mean / max action error | ms per step | Compute | Failures | Recovered within 5 s | Slot RMSE m | Min obstacle clearance m | Verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| original | 6728 KB | 100.0% | 0.00000 / 0.0000 | 2.13 | float32 | 1/140 | 74/80 | 0.184 | -0.011 | 1/3 checks |
| fp32 | 1122 KB | 16.7% | 0.00000 / 0.0000 | 2.12 | float32 | 1/140 | 74/80 | 0.184 | -0.011 | 1/3 checks |
| fp16 | 567 KB | 8.4% | 0.00004 / 0.0003 | 2.11 | float32 | 1/140 | 74/80 | 0.184 | -0.011 | 1/3 checks |
| int8 | 306 KB | 4.6% | 0.00110 / 0.0098 | 2.17 | float32 | 1/140 | 76/80 | 0.187 | -0.007 | 2/3 checks |
| int8-dynamic | 306 KB | 4.6% | 0.00255 / 0.0586 | 3.30 | int8 kernels (onednn) | 3/140 | 71/80 | 0.186 | -0.007 | 1/3 checks |

Use a variant like any checkpoint: `python -m swarmflow view --checkpoint <variant>/model.pt` or `python -m swarmflow eval --policy learned --checkpoint <variant>/model.pt`.
