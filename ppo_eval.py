# Mostly taken from the following github and modified.
# https://github.com/ASzot/ppo-pytorch

import torch
import gym
import numpy as np
import types

ENV_NAME = 'MountainCarContinuous-v0'
load_path = 'ppo_model/model_1000.pt'
policy = torch.load(load_path)


def make_env():
    return gym.make(ENV_NAME)
env = make_env

obs_shape = env.observation_space.shape
current_obs = torch.zeros(1, *obs_shape)
def update_current_obs(obs):
    obs = torch.from_numpy(obs).float()
    current_obs[:, :] = obs


for i in range(1000):
    obs = env.reset()
    update_current_obs(obs)
    done = False
    episode_reward = 0.0
    while not done:
        with torch.no_grad():
            _, action, _ = policy.act(current_obs, deterministic=True)
        action = action.squeeze(1).numpy()
        obs, reward, done, _ = env.step(action)

        episode_reward += reward

        update_current_obs(obs)
        env.render()

    print(reward)