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
STDDEV = 10
STDEV_ANNEAL_RT = 0.01
LRT = 1
OBSERVE_PERIOD = 10

# env = gym.make('CartPole-v1')
# env = gym.make('MountainCar-v0')
# env = gym.make('Acrobot-v1')
env = gym.make('SpaceInvaders-ram-v0')

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)

class Policy():
    def __init__(self, input_size, output_size):
        # self.input_size = input_size
        # self.output_size = output_size
        self.hidden = np.random.normal(loc=0, scale=0.01, size=(input_size, 128))
        self.output = np.random.normal(loc=0, scale=0.01, size=(128, output_size))
        self.stddev = STDDEV

    def set_params(self, hidden, output):
        self.hidden = np.copy(hidden)
        self.output = np.copy(output)

    def perterb_params(self):
        noise_hidden = np.random.normal(loc=0, scale=self.stddev, size=self.hidden.shape)
        noise_output = np.random.normal(loc=0, scale=self.stddev, size=self.output.shape)
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
    model = Policy(env.observation_space.shape[0], env.action_space.n)
    sample_model = Policy(env.observation_space.shape[0], env.action_space.n)
    running_reward = None
    print("input size: ", env.observation_space.shape)
    print("ouput size: ", env.action_space.n)
    print("Training...")
    for i_episode in range(NUM_STEPS):
        state = env.reset()
        tot_reward = 0
        sum_hidden = np.zeros(shape=model.hidden.shape)
        sum_output = np.zeros(shape=model.output.shape)
        for w in range(NUM_SAMPLES):
            sample_model.set_params(model.hidden, model.output)
            noise_hidden, noise_output = sample_model.perterb_params()
            sample_reward = 0
            env.reset()
            for _ in range(MAX_SIM_TIME):  # Don't infinite loop while learning
                action = sample_model.select_action(state)
                state, reward, done, _ = env.step(action)
                sample_reward += reward
                if w == 0:
                    env.render()
                if done:
                    break
            tot_reward += sample_reward
            sum_hidden += noise_hidden * sample_reward
            sum_output += noise_output * sample_reward
        avg_reward = tot_reward / NUM_SAMPLES
        model.hidden = np.add(model.hidden, LRT * sum_hidden / (NUM_SAMPLES * abs(tot_reward)))
        model.output = np.add(model.output, LRT * sum_output / (NUM_SAMPLES * abs(avg_reward)))
        model.stddev = model.stddev * (1.0 - STDEV_ANNEAL_RT)
        running_reward = running_reward * 0.99 + avg_reward * 0.01 if running_reward else avg_reward
        print('Episode {}\treward: {:2f}\tavg reward: {:.2f}\tstddev: {:.2f}'.format(i_episode, avg_reward, running_reward, model.stddev))
        # if i_episode % OBSERVE_PERIOD == 0:
        #     env.reset()
        #     for _ in range(MAX_SIM_TIME):  # Don't infinite loop while learning
        #         action = model.select_action(state)
        #         state, reward, done, _ = env.step(action)
        #         env.render()
        #         if done:
        #             break

if __name__ == '__main__':
    main()