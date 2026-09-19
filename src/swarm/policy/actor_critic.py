from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


def _atanh(x: torch.Tensor) -> torch.Tensor:
    x = x.clamp(-0.999999, 0.999999)
    return 0.5 * (torch.log1p(x) - torch.log1p(-x))


def tanh_logprob(dist: torch.distributions.Normal, u: torch.Tensor) -> torch.Tensor:
    # ponytail: Gaussian entropy on u; exact tanh entropy if KL diagnostics matter
    logp = dist.log_prob(u).sum(-1)
    logp = logp - (2.0 * (math.log(2.0) - u - F.softplus(-2.0 * u))).sum(-1)
    return logp


class NeighborEncoder(nn.Module):
    def __init__(self, neighbor_k: int, hidden: int):
        super().__init__()
        self.neighbor_k = neighbor_k
        self.embed = nn.Sequential(nn.Linear(6, hidden), nn.Tanh(), nn.Linear(hidden, hidden), nn.Tanh())

    def forward(self, neighbor_flat: torch.Tensor) -> torch.Tensor:
        b = neighbor_flat.shape[0]
        n = neighbor_flat.view(b, self.neighbor_k, 6)
        dist = n[..., :3].norm(dim=-1, keepdim=True)
        mask = (dist > 1e-6).float()
        h = self.embed(n)
        w = mask / mask.sum(dim=1, keepdim=True).clamp_min(1.0)
        return (h * w).sum(dim=1)


class ActorCritic(nn.Module):
    def __init__(self, neighbor_k: int, hidden: int = 128, hover_action: float = 0.0):
        super().__init__()
        self.neighbor_k = neighbor_k
        self.self_n = 18
        self.neighbors = NeighborEncoder(neighbor_k, hidden)
        self.body = nn.Sequential(nn.Linear(18 + hidden, hidden), nn.Tanh(), nn.Linear(hidden, hidden), nn.Tanh())
        self.actor_mean = nn.Linear(hidden, 4)
        nn.init.zeros_(self.actor_mean.weight)
        hover = max(-0.999, min(0.999, float(hover_action)))
        hover_u = math.atanh(hover)
        self.actor_mean.bias.data.fill_(hover_u)
        self.register_buffer("_hover_u", torch.full((4,), hover_u))
        # ponytail: clamp min -2 forced std≥0.135; PPO then could not tighten hover
        self.log_std = nn.Parameter(torch.full((4,), -3.5))
        self.critic = nn.Linear(hidden, 1)

    def features(self, obs: torch.Tensor) -> torch.Tensor:
        self_obs = obs[:, : self.self_n]
        nei = obs[:, self.self_n :]
        x = torch.cat([self_obs, self.neighbors(nei)], dim=-1)
        return self.body(x)

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        h = self.features(obs)
        mean = self.actor_mean(h)
        std = self.log_std.clamp(-4.5, -2.0).exp().expand_as(mean)
        # detach: value loss must not move the cloned actor features
        return mean, std, self.critic(h.detach()).squeeze(-1)

    def act(self, obs: torch.Tensor, deterministic: bool = False):
        mean, std, value = self.forward(obs)
        dist = torch.distributions.Normal(mean, std)
        u = mean if deterministic else dist.sample()
        action = torch.tanh(u)
        return action, tanh_logprob(dist, u), value

    def evaluate(self, obs: torch.Tensor, action: torch.Tensor):
        mean, std, value = self.forward(obs)
        dist = torch.distributions.Normal(mean, std)
        u = _atanh(action)
        entropy = dist.entropy().sum(-1)
        return tanh_logprob(dist, u), entropy, value
