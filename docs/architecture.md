# Architecture

Runtime shape for v1. Details: [trd.md](./trd.md). Product: [prd.md](./prd.md).

```
src/swarm/          installable package (not the repo name)
  dynamics/         quadrotor body, motors
  driver/           cmd → motors (PX4/micro-ROS stand-in)
  env/              SwarmEnv reset/step
  obs/              local obs
  tasks/            three tests
  gates/            pass gates
  baseline/         PID (outer loop → cmd)
  policy/           shared actor-critic
  train/            PPO workers
  viewer/           python | gazebo (draw only)
configs/
assets/drone/       visual mesh (OBJ) + Gazebo model.sdf
assets/worlds/      swarm.sdf (draw-only)
models/             *.pt (gitignored)
results/            submit pack (images, graphs, data)
tests/
```

Train: CPU workers run `SwarmEnv`; GPU (else CPU) runs the net. Viewer is optional and is not the integrator.

Physical later: a ROS 2 or micro-ROS node *imports* `swarm` and republishes the same flight cmd; it does not fork dynamics.
