import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import gym
from torch.distributions import Categorical
import random

env = gym.make('CartPole-v0')
# env.seed(0)
gym.spaces.prng.seed(1337)

print(env.action_space.n)
print(env.action_space.sample())
print(env.action_space.sample())
print(env.action_space.sample())
print(env.action_space.sample())