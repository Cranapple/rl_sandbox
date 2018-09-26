# import argparse
import gym
import numpy as np
# from itertools import count
# from collections import namedtuple

# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import torch.optim as optim
# from torch.distributions import Categorical

NUM_SAMPLES = 100
NUM_STEPS = 50000
MAX_SIM_TIME = 100000
STDDEV = 10000
STDEV_ANNEAL_RT = 0.01
LRT = 1
OBSERVE_PERIOD = 10

env = gym.make('CartPole-v0')

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)

class Policy():
    def __init__(self):
        self.hidden = np.random.normal(loc=0, scale=0.01, size=(4, 32))
        self.output = np.random.normal(loc=0, scale=0.01, size=(32, 2))
        self.stddev = STDDEV

    def set_params(self, hidden, output):
        self.hidden = np.copy(hidden)
        self.output = np.copy(output)

    def perterb_params(self):
        noise_hidden = np.random.normal(loc=0, scale=self.stddev, size=(4, 32))
        noise_output = np.random.normal(loc=0, scale=self.stddev, size=(32, 2))
        np.add(self.hidden, noise_hidden, self.hidden)
        np.add(self.output, noise_output, self.output)
        return noise_hidden, noise_output

    def forward(self, x):
        x = np.dot(x, self.hidden)
        np.maximum(x, 0, x)  # Relu
        x = np.dot(x, self.output)
        return softmax(x)

    def select_action(self, x):
        x = self.forward(x)
        return np.random.choice(len(x), 1, p=x)[0]

def main():
    model = Policy()
    sample_model = Policy()
    running_reward = 20
    for i_episode in range(NUM_STEPS):
        state = env.reset()
        avg_reward = 0
        sum_hidden = np.zeros(shape=model.hidden.shape)
        sum_output = np.zeros(shape=model.output.shape)
        for _ in range(NUM_SAMPLES):
            sample_model.set_params(model.hidden, model.output)
            noise_hidden, noise_output = sample_model.perterb_params()
            sample_reward = 0
            env.reset()
            for _ in range(MAX_SIM_TIME):  # Don't infinite loop while learning
                action = sample_model.select_action(state)
                state, reward, done, _ = env.step(action)
                sample_reward += reward
                if done:
                    break
            avg_reward += sample_reward
            sum_hidden += noise_hidden * sample_reward
            sum_output += noise_output * sample_reward
        avg_reward /= NUM_SAMPLES
        model.hidden = np.add(model.hidden, LRT * sum_hidden / (NUM_SAMPLES * model.stddev * avg_reward))
        model.stddev = model.stddev * (1.0 - STDEV_ANNEAL_RT)
        running_reward = running_reward * 0.99 + avg_reward * 0.01
        print('Episode {}\treward: {:2f}\tavg reward: {:.2f}'.format(i_episode, avg_reward, running_reward))
        if i_episode % OBSERVE_PERIOD == 0:
            env.reset()
            for _ in range(MAX_SIM_TIME):  # Don't infinite loop while learning
                action = model.select_action(state)
                state, reward, done, _ = env.step(action)
                env.render()
                if done:
                    break

if __name__ == '__main__':
    main()