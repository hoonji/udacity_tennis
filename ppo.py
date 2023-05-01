# Implementation based on Costa Huang's PPO implementation: https://github.com/vwxyzjn/ppo-implementation-details/blob/main/ppo_continuous_action.py
# Updated to support and solve the multi agent Reacher environment.

import time
import pickle
import random
from collections import namedtuple
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from ppo_agent import Agent

LEARNING_RATE = 3e-4
# ADAM_EPS = 1e-5
WEIGHT_DECAY = 1e-4
GAMMA = .99
UPDATE_EPOCHS = 3
CLIP_COEF = .2
MAX_GRAD_NORM = 5
GAE_LAMBDA = .95
V_COEF = .5
HIDDEN_LAYER_SIZE = 64
ENTROPY_COEF = .01
N_EPISODES = 20000


class Rollout:
  """Stores rollouts and yields minibatches for training."""

  # Batch for PPO training loop.
  Batch = namedtuple(
      'Batch', ['observations', 'obsactions', 'actions', 'advantages', 'logprobs', 'values'])

  def __init__(self):
    self.reset()

  def __len__(self):
    return len(self.actions)

  def reset(self):
    self.observations = []
    self.actions = []
    self.rewards = []
    self.obsactions = []
    self.dones = []
    self.logprobs = []
    self.values = []
    self.advantages = None

  def gen_minibatches(self, batch_size=256):
    """Generates shuffled minibatches using rollout data."""
    batch_indices = np.arange(len(self))
    np.random.shuffle(batch_indices)

    observations = torch.tensor(self.observations, dtype=torch.float32)
    actions = torch.tensor(self.actions, dtype=torch.float32)
    obsactions = torch.tensor(self.obsactions, dtype=torch.float32)
    logprobs = torch.tensor(self.logprobs, dtype=torch.float32)
    values = torch.tensor(self.values, dtype=torch.float32)

    for start in range(1 + len(self) // batch_size):
      indices = batch_indices[start:start + batch_size]
      minibatch = self.Batch(
          observations=observations[indices],
          obsactions=obsactions[indices],
          actions=actions[indices],
          advantages=self.advantages[indices],
          logprobs=logprobs[indices],
          values=values[indices],
      )
      yield minibatch


def run_ppo(env, seed=123):
  """Trains a ppo agent in an environment.

  Saves model and learning curve checkpoints.
  """
  # set seeds
  torch.manual_seed(seed)
  torch.cuda.manual_seed_all(seed)
  np.random.seed(seed)
  random.seed(seed)

  brain_name = env.brain_names[0]
  brain = env.brains[brain_name]
  env_info = env.reset(train_mode=True)[brain_name]
  observations = env_info.vector_observations
  n_agents = len(env_info.agents)
  n_observations = env_info.vector_observations.shape[1]
  n_actions = brain.vector_action_space_size
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

  agent = Agent(n_observations, n_actions, HIDDEN_LAYER_SIZE).to(device)
  optimizer = optim.Adam(agent.parameters(),
                         lr=LEARNING_RATE,
                         weight_decay=WEIGHT_DECAY)

  rollout = Rollout()

  next_dones = torch.zeros(n_agents).to(device)
  current_returns = np.zeros(n_agents)
  scores = []
  time_checkpoint = time.time()

  for episode in range(1, N_EPISODES + 1):
    rollout.reset()
    env_info = env.reset(train_mode=True)[brain_name]
    observations = env_info.vector_observations

    while True:
      with torch.no_grad():
        actions, logprobs = agent.pi(
            torch.tensor(observations, dtype=torch.float32))
        actions, logprobs = actions.numpy(), logprobs.numpy()

      env_info = env.step(np.clip(actions, -1, 1))[brain_name]
      next_observations = env_info.vector_observations

      rollout.observations.append(observations)
      rollout.actions.append(actions)
      rollout.logprobs.append(logprobs)
      obsactions = []  # combined and normalized observation + action
      obsactions.append(
          np.concatenate(
              (observations[0], observations[1], actions[0], actions[1])))
      obsactions.append(
          np.concatenate(
              (observations[1], observations[0], actions[1], actions[0])))
      rollout.obsactions.append(obsactions)
      rollout.rewards.append(env_info.rewards)
      rollout.dones.append(np.array(env_info.local_done))
      with torch.no_grad():
        values = agent.critic(torch.tensor(obsactions,
                                           dtype=torch.float32)).squeeze(-1)
        values = values.numpy()
      rollout.values.append(values)

      # Record agent returns
      current_returns += env_info.rewards
      scores.extend(current_returns[env_info.local_done])
      current_returns[env_info.local_done] = 0

      observations = next_observations

      if np.any(env_info.local_done):
        break

    z = np.zeros(n_agents)
    rollout_len = len(rollout)
    rollout.advantages = torch.zeros((rollout_len, n_agents),
                                     dtype=torch.float32)
    for t in reversed(range(rollout_len)):
      next_values = rollout.values[t + 1] if t != rollout_len - 1 else 0
      td_errors = rollout.rewards[t] + (
          1 - rollout.dones[t]) * GAMMA * next_values - rollout.values[t]
      z = td_errors + (1 - rollout.dones[t]) * GAMMA * GAE_LAMBDA * z
      rollout.advantages[t] = torch.from_numpy(z)

    rollout.advantages = (rollout.advantages - rollout.advantages.mean()
                          ) / rollout.advantages.std()

    for epoch in range(UPDATE_EPOCHS):
      for observations, obsactions, actions, advantages, logprobs, values in rollout.gen_minibatches(
      ):
        _, new_logprobs = agent.pi(observations, actions=actions)
        new_values = agent.critic(obsactions).squeeze(-1)
        entropy = -(new_logprobs.exp() * logprobs)
        ratios = (new_logprobs - logprobs).exp()

        # Surrogate objective
        surrogate_loss1 = -advantages * ratios
        surrogate_loss2 = -advantages * torch.clamp(ratios, 1 - CLIP_COEF,
                                                    1 + CLIP_COEF)
        surrogate_loss = torch.max(surrogate_loss1, surrogate_loss2).mean()
        value_loss = V_COEF * (((advantages + values) - new_values)**2).mean()
        entropy_loss = ENTROPY_COEF * entropy.mean()
        loss = surrogate_loss + value_loss - entropy_loss

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(agent.parameters(), MAX_GRAD_NORM)
        optimizer.step()

    if episode % 1000 == 0:
      torch.save(agent.state_dict(), f'{brain_name}_model_checkpoint.pickle')

    last_100_returns = np.array(scores[-100:]).mean()
    if episode % 50 == 0:
      print(
          f"episode {episode}. Last update in {time.time() - time_checkpoint}s")
      time_checkpoint = time.time()
      print(f'last 100 returns: {last_100_returns}')

    if last_100_returns > .5:
      print(f'last 100 returns: {last_100_returns}')
      print(f'solved after {episode} episodes')

      print(f'saving brain to {brain_name}_model.pickle')
      torch.save(agent.state_dict(), f'{brain_name}_model.pickle')

      print(f'saving scores to {brain_name}_scores.pickle')
      with open(f'{brain_name}_scores.pickle', 'wb') as f:
        pickle.dump(scores, f)
