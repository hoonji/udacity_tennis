import numpy as np
import torch
import torch.nn as nn
from torch.distributions.normal import Normal


class OrthogonalInitLinear(nn.Linear):

  def __init__(self, *args, scale=1.0, **kwargs):
    super().__init__(*args, **kwargs)
    torch.nn.init.orthogonal_(self.weight)
    self.weight.data.mul_(scale)
    torch.nn.init.constant_(self.bias, 0)


class Agent(nn.Module):

  def __init__(self, n_observations, n_actions, hidden_size=256):
    super().__init__()
    self.std = nn.Parameter(torch.ones(1, n_actions))
    self.actor_means = nn.Sequential(
        OrthogonalInitLinear(n_observations, hidden_size), nn.LeakyReLU(),
        OrthogonalInitLinear(hidden_size, hidden_size), nn.LeakyReLU(),
        OrthogonalInitLinear(hidden_size, n_actions, scale=1e-3), nn.Tanh())
    self.critic = nn.Sequential(
        OrthogonalInitLinear(n_observations * 2 + n_actions * 2, hidden_size),
        nn.LeakyReLU(),
        OrthogonalInitLinear(hidden_size, hidden_size),
        nn.LeakyReLU(),
        OrthogonalInitLinear(hidden_size, 1, scale=1e-3),
    )

  def pi(self, x, actions=None):
    """Returns a sampled action and its probability distribution."""
    means = self.actor_means(x)
    distribution = Normal(means, self.std)
    if actions == None:
      actions = distribution.sample()
    logprobs = distribution.log_prob(actions).sum(-1)

    return actions, logprobs
