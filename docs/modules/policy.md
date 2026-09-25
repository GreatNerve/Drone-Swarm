# policy

Shared actor-critic. Deep-sets neighbor encoder. Obs → 4-cmd `[roll, pitch, yaw_rate, climb]` + value.

Gaussian on unbounded `u`, then `tanh(u)` with Jacobian log-prob correction. PPO must call `evaluate`, not a raw Normal on squashed actions. Value is `critic(h.detach())` so value loss cannot move actor features. Action `log_std` is clamped to `[-4.5, -2]` (quiet hover noise).

Actor mean is initialized at zeros (level hover). PPO and the PID clone learn the cmd, not motors.

Public: `ActorCritic`, `NeighborEncoder`. Depends on PyTorch.
