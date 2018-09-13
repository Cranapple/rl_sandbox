# Mostly taken from the following github and modified.
# https://github.com/ASzot/ppo-pytorch

import os
import shutil
import argparse
import gym
import numpy as np
from itertools import count
from collections import namedtuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler

from tensorboardX import SummaryWriter

###########################################################################
# HYPERPARAMS AND GLOBALS
###########################################################################

EPS = 1e-5  # Standard definition of epsilon
CLIP_PARAM = 0.2  # Hyperparams of the PPO equation
N_EPOCH = 4  # Number of update epochs
N_MINI_BATCH = 32  # Number of mini batches to use in updating
VALUE_LOSS_COEFF = 0.5  # Coefficients for the loss term. (all relative to the action loss which is 1.0)
ENTROPY_COEFF = 0 # 0.01
LR = 7e-4  # Learning rate of the optimizer
MAX_GRAD_NORM = 0.5  # Clip gradient norm
N_STEPS = 80  # Number of steps to generate actions
N_FRAMES = 10e6  # Total number of frames to train on
GAMMA = 0.99  # Discounted reward factor
ENV_NAME = 'MountainCarContinuous-v0'  # Environment we are going to use. Make sure it is a continuous action space.
SAVE_INTERVAL = 500
LOG_INTERVAL = 10
MODEL_DIR = 'ppo_model'

env = gym.make(ENV_NAME)

###########################################################################
# Memory
###########################################################################

"""
Stores the data from a rollout that will be used later in genenerating samples
"""
class RolloutStorage(object):
    def __init__(self, num_steps, obs_shape, action_space, start_obs):
        self.observations = torch.zeros(num_steps + 1, *obs_shape)
        self.rewards = torch.zeros(num_steps, 1)
        self.value_preds = torch.zeros(num_steps + 1, 1)
        self.returns = torch.zeros(num_steps + 1, 1)
        self.action_log_probs = torch.zeros(num_steps, 1)
        action_shape = action_space.shape[0]
        self.actions = torch.zeros(num_steps, action_shape)
        self.masks = torch.ones(num_steps + 1, 1)

        self.num_steps = num_steps
        self.step = 0
        self.observations[0].copy_(start_obs)

    def insert(self, current_obs, action, action_log_prob, value_pred, reward, mask):
        self.observations[self.step + 1].copy_(current_obs)
        self.actions[self.step].copy_(action)
        self.action_log_probs[self.step].copy_(action_log_prob)
        self.value_preds[self.step].copy_(value_pred)
        self.rewards[self.step].copy_(reward)
        self.masks[self.step + 1].copy_(mask)

        self.step = (self.step + 1) % self.num_steps

    def after_update(self):
        self.observations[0].copy_(self.observations[-1])
        self.masks[0].copy_(self.masks[-1])

    def compute_returns(self, next_value, gamma):
        self.returns[-1] = next_value
        for step in reversed(range(self.rewards.size(0))):
            self.returns[step] = self.returns[step + 1] * gamma * self.masks[step + 1] + self.rewards[step]

    def sample(self, advantages, num_mini_batch):
        num_steps = self.rewards.size()[0:1]
        batch_size = num_steps[0]
        # Make sure we have at least enough for a bunch of batches of size 1.
        assert batch_size >= num_mini_batch

        mini_batch_size = batch_size // num_mini_batch
        sampler = BatchSampler(SubsetRandomSampler(range(batch_size)), mini_batch_size, drop_last=False)
        for indices in sampler:
            observations_batch = self.observations[:-1].view(-1, *self.observations.size()[1:])[indices]
            actions_batch = self.actions.view(-1, self.actions.size(-1))[indices]
            return_batch = self.returns[:-1].view(-1, 1)[indices]
            masks_batch = self.masks[:-1].view(-1, 1)[indices]
            old_action_log_probs_batch = self.action_log_probs.view(-1, 1)[indices]
            adv = advantages.view(-1, 1)[indices]

            yield observations_batch, actions_batch, return_batch, masks_batch, old_action_log_probs_batch, adv

###########################################################################
# MODEL
###########################################################################

# Init layer to have the proper weight initializations.
def init_layer(m):
    weight = m.weight.data
    weight.normal_(0, 1)
    weight *= 1.0 / torch.sqrt(weight.pow(2).sum(1, keepdim=True))
    nn.init.constant_(m.bias.data, 0)
    return m

# Standard feed forward network for actor and critic with tanh activations
class Mlp(nn.Module):
    def __init__(self, num_inputs):
        super().__init__()
        # We do not want to select action yet as that will be probablistic.
        self.actor_hidden = nn.Sequential(
                init_layer(nn.Linear(num_inputs, 64)),
                nn.Tanh(),
                init_layer(nn.Linear(64, 64)),
                nn.Tanh(),
            )
        self.critic = nn.Sequential(
                init_layer(nn.Linear(num_inputs, 64)),
                nn.Tanh(),
                init_layer(nn.Linear(64, 64)),
                nn.Tanh(),
                init_layer(nn.Linear(64, 1)),
            )
        self.train()

    def forward(self, inputs):
        return self.actor_hidden(inputs), self.critic(inputs)

class Policy(nn.Module):
    def __init__(self, obs_shape, action_space):
        super().__init__()
        self.actor_critic = Mlp(obs_shape[0])
        num_outputs = action_space.shape[0]
        # How we will define our normal distribution to sample action from
        self.action_mean = init_layer(nn.Linear(64, num_outputs))
        self.action_log_std = nn.Parameter(torch.zeros(1, num_outputs))

    def __get_dist(self, actor_features):
        action_mean = self.action_mean(actor_features)
        # action_log_std = self.action_log_std.expand_as(action_mean)
        return torch.distributions.Normal(action_mean, self.action_log_std.exp())

    def act(self, inputs, deterministic=False):
        actor_features, value = self.actor_critic(inputs)
        dist = self.__get_dist(actor_features)
        if deterministic:
            action = dist.mean[0]
        else:
            action = dist.sample()[0]
        action_log_probs = dist.log_prob(action).sum(-1, keepdim=True)[0]
        return value, action, action_log_probs

    def get_value(self, inputs):
        _, value = self.actor_critic(inputs)
        return value

    def evaluate_actions(self, inputs, action):
        actor_features, value = self.actor_critic(inputs)
        dist = self.__get_dist(actor_features)
        action_log_probs = dist.log_prob(action).sum(-1, keepdim=True)
        dist_entropy = dist.entropy().sum(-1).mean()
        return value, action_log_probs, dist_entropy

###########################################################################
# TRAINING
###########################################################################

def update_params(rollouts, policy, optimizer):
    advantages = rollouts.returns[:-1] - rollouts.value_preds[:-1]
    # Normalize advantages. (0 mean. 1 std)
    advantages = (advantages - advantages.mean()) / (advantages.std() + EPS)
    for _ in range(N_EPOCH):
        samples = rollouts.sample(advantages, N_MINI_BATCH)
        value_losses = []
        action_losses = []
        entropy_losses = []
        losses = []
        for obs, actions, returns, masks, old_action_log_probs, adv_targ in samples:
            values, action_log_probs, dist_entropy = policy.evaluate_actions(obs, actions)
            # This is where we apply the PPO equation.
            ratio = torch.exp(action_log_probs - old_action_log_probs)
            surr1 = ratio * adv_targ
            surr2 = torch.clamp(ratio, 1.0 - CLIP_PARAM, 1.0 + CLIP_PARAM) * adv_targ
            action_loss = -torch.min(surr1, surr2).mean()
            value_loss = F.mse_loss(returns, values)

            optimizer.zero_grad()
            loss = (value_loss * VALUE_LOSS_COEFF + action_loss - dist_entropy * ENTROPY_COEFF)
            loss.backward()
            nn.utils.clip_grad_norm_(policy.parameters(), MAX_GRAD_NORM)
            optimizer.step()

            value_losses.append(value_loss.item())
            action_losses.append(action_loss.item())
            entropy_losses.append(dist_entropy.item())
            losses.append(loss.item())

    return np.mean(value_losses), np.mean(action_losses), np.mean(entropy_losses), np.mean(losses)

def main():
    # Create our model output path if it does not exist.
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)
    else:
        shutil.rmtree(MODEL_DIR)
        os.makedirs(MODEL_DIR)
    # Create logging directory
    if os.path.exists('runs'):
        shutil.rmtree('runs')
    writer = SummaryWriter()

    obs_shape = env.observation_space.shape
    policy = Policy(obs_shape, env.action_space)
    optimizer = optim.Adam(policy.parameters(), lr=LR, eps=EPS)
    obs = env.reset()
    current_obs = torch.from_numpy(obs).float()
    
    # Intialize our rollouts
    rollouts = RolloutStorage(N_STEPS, obs_shape, env.action_space, current_obs)

    # episode_rewards = torch.zeros([N_ENVS, 1])
    # final_rewards = torch.zeros([N_ENVS, 1])
    episode_reward = 0
    final_reward = 0

    n_updates = int(N_FRAMES // N_STEPS)
    for update_i in range(n_updates):
        # Generate samples
        for step in range(N_STEPS):
            # Generate and take an action
            with torch.no_grad():
                value, action, action_log_prob = policy.act(rollouts.observations[step])
            take_actions = action.squeeze().numpy()
            # if len(take_actions.shape) == 1:
            #     take_actions = np.expand_dims(take_actions, axis=-1)
            # obs, reward, done, info = envs.step(take_actions)
            obs, reward, done, info = env.step(action.numpy())
            # convert to pytorch tensor
            # reward = torch.from_numpy(np.expand_dims(np.stack(reward), 1)).float()
            # masks = torch.FloatTensor([[0.0] if d else [1.0] for d in done])
            mask = 0.0 if done else 1.0
            # update reward info for logging
            # episode_rewards += reward
            # final_rewards *= masks
            # final_rewards += (1 - masks) * episode_rewards
            # episode_rewards *= masks
            episode_reward += reward
            final_reward *= mask
            final_reward += (1 - mask) * episode_reward
            episode_reward *= mask
            # Update our current observation tensor
            current_obs *= mask
            # update_current_obs(obs)
            obs = torch.from_numpy(obs).float()
            current_obs[:] = obs
            rollouts.insert(current_obs, action, action_log_prob, value, 
                            torch.FloatTensor([reward]), torch.FloatTensor([mask]))
        with torch.no_grad():
            next_value = policy.get_value(rollouts.observations[-1]).detach()
        rollouts.compute_returns(next_value, GAMMA)
        value_loss, action_loss, entropy_loss, overall_loss = update_params(rollouts, policy, optimizer)
        rollouts.after_update()

        # Log to tensorboard
        writer.add_scalar('data/action_loss', action_loss, update_i)
        writer.add_scalar('data/value_loss', value_loss, update_i)
        writer.add_scalar('data/entropy_loss', entropy_loss, update_i)
        writer.add_scalar('data/overall_loss', overall_loss, update_i)
        # writer.add_scalar('data/avg_reward', final_rewards.mean(), update_i)
        writer.add_scalar('data/avg_reward', final_reward, update_i)

        if update_i % LOG_INTERVAL == 0:
            print('Reward: %.3f' % (final_reward))

        if update_i % SAVE_INTERVAL == 0:
            save_model = policy
            torch.save(save_model, os.path.join(MODEL_DIR, 'model_%i.pt' % update_i))

if __name__ == '__main__':
    main()