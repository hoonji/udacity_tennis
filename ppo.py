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
from tensorboardX import SummaryWriter

from ppo_agent import Agent

LEARNING_RATE = 3e-4
ADAM_EPS = 1e-5
GAMMA = .99
LAMBDA = .95
UPDATE_EPOCHS = 3
N_MINIBATCHES = 10
CLIP_COEF = .2
MAX_GRAD_NORM = 5
GAE_LAMBDA = .95
V_COEF = .5
HIDDEN_LAYER_SIZE = 32
ROLLOUT_LEN = 1024
N_ROLLOUTS = 50000
ENTROPY_COEF = 0


@dataclass
class Rollout:
  """Stores rollouts and yields minibatches for training."""
  batch_size: int
  observations: torch.Tensor
  actions: torch.Tensor
  rewards: torch.Tensor
  dones: torch.Tensor
  logprobs: torch.Tensor
  values: torch.Tensor
  advantages: torch.Tensor

  # Tensors required for PPO training loop.
  Batch = namedtuple(
      'Batch', ['observations', 'actions', 'advantages', 'logprobs', 'values'])

  def gen_minibatches(self):
    """Generates shuffled minibatches using rollout data."""
    batch_indices = np.arange(self.batch_size)
    np.random.shuffle(batch_indices)
    minibatch_size = self.batch_size // N_MINIBATCHES
    for start in range(0, self.batch_size, minibatch_size):
      indices = batch_indices[start:start + minibatch_size]
      minibatch = self.Batch(
          observations=self.observations.reshape(
              (self.batch_size, -1))[indices],
          actions=self.actions.reshape((self.batch_size, -1))[indices],
          advantages=self.advantages.reshape(self.batch_size)[indices],
          logprobs=self.logprobs.reshape(self.batch_size)[indices],
          values=self.values.reshape(self.batch_size)[indices],
      )
      yield minibatch


def run_ppo(env):
  """Trains a ppo agent in an environment.

  Saves model and learning curve checkpoints.
  """
  writer = SummaryWriter()

  brain_name = env.brain_names[0]
  brain = env.brains[brain_name]
  env_info = env.reset(train_mode=True)[brain_name]
  num_agents = len(env_info.agents)
  n_observations = env_info.vector_observations.shape[1]
  n_actions = brain.vector_action_space_size
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  batch_size = ROLLOUT_LEN * num_agents

  agent = Agent(n_observations, n_actions, HIDDEN_LAYER_SIZE).to(device)
  optimizer = optim.Adam(agent.parameters(), lr=LEARNING_RATE, eps=ADAM_EPS)

  rollout = Rollout(batch_size=batch_size,
                    observations=torch.zeros(
                        (ROLLOUT_LEN, num_agents, n_observations)).to(device),
                    actions=torch.zeros(
                        (ROLLOUT_LEN, num_agents, n_actions)).to(device),
                    logprobs=torch.zeros(ROLLOUT_LEN, num_agents).to(device),
                    rewards=torch.zeros(ROLLOUT_LEN, num_agents).to(device),
                    dones=torch.zeros(ROLLOUT_LEN, num_agents).to(device),
                    values=torch.zeros(ROLLOUT_LEN, num_agents).to(device),
                    advantages=torch.zeros(ROLLOUT_LEN, num_agents).to(device))

  next_observations = torch.Tensor(env_info.vector_observations).to(device)
  next_dones = torch.zeros(num_agents).to(device)
  current_returns = np.zeros(num_agents)
  scores = []
  time_checkpoint = time.time()
  n_episodes = 0
  max_return = 0
  solved = False

  print('Beginning training loop')
  for update in range(1, N_ROLLOUTS + 1):
    lr_frac = 1.0 - (update - 1) / N_ROLLOUTS
    optimizer.param_groups[0]["lr"] = lr_frac * LEARNING_RATE

    for t in range(ROLLOUT_LEN):
      observations = next_observations

      with torch.no_grad():
        actions, probs = agent.pi(observations)
        values = agent.critic(observations)
      env_info = env.step(actions.cpu().numpy())[brain_name]
      dones = env_info.local_done
      rewards = np.array(env_info.rewards)

      rollout.observations[t] = observations
      rollout.actions[t] = actions
      rollout.rewards[t] = torch.Tensor(rewards).to(device)
      rollout.dones[t] = next_dones  # for this step, record previous dones
      rollout.logprobs[t] = probs.log_prob(actions).sum(1)
      rollout.values[t] = values.flatten()

      # Record agent returns
      current_returns += env_info.rewards
      scores.extend(current_returns[dones])
      current_returns[dones] = 0

      if any(dones):
        writer.add_scalar("ppo/agent0_returns", scores[-2], n_episodes)
        writer.add_scalar("ppo/agent1_returns", scores[-1], n_episodes)
        env_info = env.reset(train_mode=True)[brain_name]
        n_episodes += 1

      next_observations = torch.Tensor(env_info.vector_observations).to(device)
      next_dones = torch.Tensor([dones]).to(device)

    last_100_avg = np.array(scores[-100:]).mean()
    writer.add_scalar("ppo/last_100_returns", last_100_avg, update)
    if not solved and last_100_avg > .5:
      print(f'Solved in {n_episodes} episodes')
      solved = True

    if last_100_avg > 1.0:
      torch.save(agent.state_dict(), f'{brain_name}_model_final.pickle')
      return

    z = 0
    for t in reversed(range(ROLLOUT_LEN)):
      if t == ROLLOUT_LEN - 1:
        next_nonterminal = 1.0 - next_dones
        with torch.no_grad():
          next_values = agent.critic(next_observations).flatten()
      else:
        next_nonterminal = 1.0 - rollout.dones[t + 1]
        next_values = rollout.values[t + 1]
      td_errors = rollout.rewards[
          t] + next_nonterminal * GAMMA * next_values - rollout.values[t]
      z = td_errors + next_nonterminal * GAMMA * GAE_LAMBDA * z
      rollout.advantages[t] = z

    clipped = []
    ratios_list = []
    for epoch in range(UPDATE_EPOCHS):
      for observations, actions, advantages, logprobs, values in rollout.gen_minibatches(
      ):
        _, probs = agent.pi(observations)
        next_values = agent.critic(observations)
        next_logprobs = probs.log_prob(actions).sum(1)
        next_entropy = probs.entropy().sum(1)
        ratios = (next_logprobs - logprobs).exp()
        # Add to tracking variables
        ratios_list.append(ratios.mean().item())
        clipped.append(((1.0 - ratios) > CLIP_COEF).float().mean().item())

        # Surrogate objective
        surrogate_loss1 = -advantages * ratios
        surrogate_loss2 = -advantages * torch.clamp(ratios, 1 - CLIP_COEF,
                                                    1 + CLIP_COEF)
        surrogate_loss = torch.max(surrogate_loss1, surrogate_loss2).mean()
        value_loss = V_COEF * (((advantages + values) - next_values)**2).mean()
        entropy_loss = ENTROPY_COEF * next_entropy.mean()
        loss = surrogate_loss + value_loss - entropy_loss

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(agent.parameters(), MAX_GRAD_NORM)
        optimizer.step()

    writer.add_scalar('ppo/advantages', rollout.advantages.mean(), update)
    writer.add_scalar('ppo/prob_ratios', np.mean(ratios_list), update)
    writer.add_scalar('ppo/policy_loss', surrogate_loss, update)
    writer.add_scalar('ppo/clipped_ratio', np.mean(clipped), update)
    writer.add_scalar('ppo/values', rollout.values.mean(), update)
    writer.add_scalar('ppo/value_loss', value_loss, update)
    writer.add_scalar('ppo/probs', torch.exp(rollout.logprobs).mean(), update)
    writer.add_scalar('ppo/stddevs', torch.exp(agent.actor_logstd).mean(), update)

    # if update % 5000 == 0:
    if update % 10 == 0:
      print(
          f"update {update}/{N_ROLLOUTS}. finished {n_episodes} episodes. Last update in {time.time() - time_checkpoint}s"
      )
      time_checkpoint = time.time()
      print(f'average of last 100 returns: {last_100_avg}')
      print(f'max of last 100: {max(scores[-100:])}')

    if update % 1000 == 0:
      torch.save(agent.state_dict(), f'checkpoints/{brain_name}_model_checkpoint_update_{update}.pickle')

  torch.save(agent.state_dict(), f'{brain_name}_model_final.pickle')
