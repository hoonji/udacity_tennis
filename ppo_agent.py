import numpy as np
import torch
import torch.nn as nn
from torch.distributions.normal import Normal


class OrthogonalInitLinear(nn.Linear):

  def __init__(self, *args, stddev=np.sqrt(2), constant_bias=0.0, **kwargs):
    super().__init__(*args, **kwargs)
    torch.nn.init.orthogonal_(self.weight, stddev)
    torch.nn.init.constant_(self.bias, constant_bias)


class Agent(nn.Module):

  def __init__(self, n_observations, n_actions, hidden_size=256):
    super().__init__()
    self.critic = nn.Sequential(
        OrthogonalInitLinear(n_observations, hidden_size),
        nn.ReLU(),
        OrthogonalInitLinear(hidden_size, hidden_size),
        nn.ReLU(),
        OrthogonalInitLinear(hidden_size, 1),
        nn.Tanh(),
    )
    self.actor_means = nn.Sequential(
        OrthogonalInitLinear(n_observations, hidden_size), nn.ReLU(),
        OrthogonalInitLinear(hidden_size, hidden_size), nn.ReLU(),
        OrthogonalInitLinear(hidden_size, n_actions))
    self.actor_logstd = nn.Parameter(torch.zeros(1, n_actions))

  def pi(self, x):
    """Returns a sampled action and its probability distribution."""
    action_means = self.actor_means(x)
    action_stddevs = torch.exp(self.actor_logstd.expand_as(action_means))
    probs = Normal(action_means, action_stddevs)
    return probs.sample(), probs
