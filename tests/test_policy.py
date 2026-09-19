import pytest


def test_tanh_actions_and_logprob():
    try:
        import torch
    except OSError as exc:
        pytest.skip(f"torch unavailable: {exc}")
    from swarm.policy.actor_critic import ActorCritic

    policy = ActorCritic(neighbor_k=3, hidden=32, hover_action=-0.4)
    policy.eval()
    obs = torch.zeros(2, 18 + 6 * 3)
    with torch.no_grad():
        act, logp, value = policy.act(obs, deterministic=True)
        _ = act.detach().cpu().numpy()
    assert act.shape == (2, 4)
    assert torch.allclose(act, torch.full_like(act, -0.4), atol=0.05)
    assert torch.all(act.abs() <= 1.0)
    assert torch.isfinite(logp).all()
    assert value.shape == (2,)
    logp2, ent, _ = policy.evaluate(obs, act.detach())
    assert torch.isfinite(logp2).all()
    assert torch.isfinite(ent).all()
    assert torch.allclose(logp, logp2, atol=1e-4)
    sampled, slog, _ = policy.act(obs, deterministic=False)
    assert torch.isfinite(slog).all()
    assert torch.all(sampled.abs() <= 1.0)


def test_value_loss_does_not_move_actor_mean():
    try:
        import torch
    except OSError as exc:
        pytest.skip(f"torch unavailable: {exc}")
    from swarm.policy.actor_critic import ActorCritic

    policy = ActorCritic(neighbor_k=3, hidden=32)
    before = policy.actor_mean.weight.data.clone()
    obs = torch.randn(4, 18 + 6 * 3)
    opt = torch.optim.Adam(policy.parameters(), lr=0.05)
    _mean, _std, value = policy.forward(obs)
    loss = value.pow(2).mean()
    opt.zero_grad()
    loss.backward()
    opt.step()
    assert torch.equal(before, policy.actor_mean.weight.data)


def test_action_std_can_go_quieter_than_old_floor():
    try:
        import torch
    except OSError as exc:
        pytest.skip(f"torch unavailable: {exc}")
    from swarm.policy.actor_critic import ActorCritic

    policy = ActorCritic(neighbor_k=3, hidden=32)
    policy.log_std.data.fill_(-4.2)
    obs = torch.zeros(1, 18 + 6 * 3)
    _mean, std, _value = policy.forward(obs)
    assert float(std.mean().detach()) < 0.03


def test_pid_warmstart_reduces_action_error():
    try:
        import torch
    except OSError as exc:
        pytest.skip(f"torch unavailable: {exc}")
    from swarm.config import load
    from swarm.policy.actor_critic import ActorCritic
    from swarm.train.loop import warmstart_pid

    cfg = load(train=True)
    device = torch.device("cpu")
    policy = ActorCritic(int(cfg["neighbor_k"]), 32)
    opt = torch.optim.Adam(policy.parameters(), lr=0.01)
    mse = warmstart_pid(cfg, policy, opt, device, ticks=40)
    assert mse < 0.25
