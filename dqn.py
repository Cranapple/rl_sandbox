# Mostly taken from pytorch example github
# https://github.com/pytorch/examples/tree/master/reinforcement_learning

import argparse
import gym
import numpy as np
from itertools import count

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical


parser = argparse.ArgumentParser(description='PyTorch DQN example')
parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                    help='discount factor (default: 0.99)')
parser.add_argument('--seed', type=int, default=543, metavar='N',
                    help='random seed (default: 543)')
parser.add_argument('--render', action='store_true',
                    help='render the environment')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='interval between training status logs (default: 10)')
args = parser.parse_args()


env = gym.make('CartPole-v0')
env.seed(args.seed)
torch.manual_seed(args.seed)


class Policy(nn.Module):
    def __init__(self):
        super(Policy, self).__init__()
        self.affine1 = nn.Linear(4, 128)
        self.affine2 = nn.Linear(128, 2)

        self.saved_log_probs = []
        self.rewards = []

    def forward(self, x):
        x = F.relu(self.affine1(x))
        action_scores = self.affine2(x)
        return F.softmax(action_scores, dim=1)


policy = Policy()
optimizer = optim.Adam(policy.parameters(), lr=1e-2)
eps = np.finfo(np.float32).eps.item()


def select_action(state):
    state = torch.from_numpy(state).float().unsqueeze(0)
    probs = policy(state)
    m = Categorical(probs)
    action = m.sample()
    policy.saved_log_probs.append(m.log_prob(action))
    return action.item()


def finish_episode():
    R = 0
    policy_loss = []
    rewards = []
    for r in policy.rewards[::-1]:
        R = r + args.gamma * R
        rewards.insert(0, R)
    rewards = torch.tensor(rewards)
    rewards = (rewards - rewards.mean()) / (rewards.std() + eps)
    for log_prob, reward in zip(policy.saved_log_probs, rewards):
        policy_loss.append(-log_prob * reward)
    optimizer.zero_grad()
    policy_loss = torch.cat(policy_loss).sum()
    policy_loss.backward()
    optimizer.step()
    del policy.rewards[:]
    del policy.saved_log_probs[:]


def main():
    running_reward = 10
    for i_episode in count(1):
        state = env.reset()
        for t in range(10000):  # Don't infinite loop while learning
            action = select_action(state)
            state, reward, done, _ = env.step(action)
            if args.render:
                env.render()
            policy.rewards.append(reward)
            if done:
                break

        running_reward = running_reward * 0.99 + t * 0.01
        finish_episode()
        if i_episode % args.log_interval == 0:
            print('Episode {}\tLast length: {:5d}\tAverage length: {:.2f}'.format(
                i_episode, t, running_reward))
        if running_reward > env.spec.reward_threshold:
            print("Solved! Running reward is now {} and "
                  "the last episode runs to {} time steps!".format(running_reward, t))
            break


if __name__ == '__main__':
    main()

# import gym
# import numpy as np
# import tensorflow as tf
# import matplotlib.pyplot as plt

# env = gym.make('CartPole-v0')
# # env = gym.make('Acrobot-v1')
# # env = gym.make('MountainCar-v0')
# # env = gym.make('MountainCarContinuous-v0')
# # env = gym.make('Pendulum-v0')
# print(env.observation_space.sample())
# print(env.action_space.sample())

# # Hyperparams
# num_ep = 100
# ep_time = 10000
# hidden_layer_size = 100
# eps = 0.1
# lmbd = 0.99

# # Graph params
# reward_list = []
# loss_list = []
# time_list = []

# # TF Variables.
# inputs = tf.placeholder(shape=env.observation_space.sample().shape, dtype=tf.float32)
# in_flat = tf.reshape(inputs, [1, -1])
# hw = tf.Variable(tf.random_uniform([in_flat.shape[1].value, hidden_layer_size], 0, 0.01))
# hb = tf.Variable(tf.zeros([1, hidden_layer_size]))
# hw_ac = tf.nn.relu(hw + hb)
# w = tf.Variable(tf.random_uniform([hw_ac.shape[1].value, env.action_space.n], 0, 0.01))
# b = tf.Variable(tf.zeros([1, env.action_space.n]))
# q = tf.matmul(hw_ac, w) + b
# # w = tf.Variable(tf.random_uniform([in_flat.shape[1].value, env.action_space.n], 0, 0.01))
# # b = tf.Variable(tf.zeros([1, env.action_space.n]))
# # q = tf.matmul(in_flat, w) + b
# a_max = tf.argmax(q, 1)
# new_q = tf.placeholder(shape=q.shape, dtype=tf.float32)
# loss = tf.reduce_sum(tf.square(new_q - q))
# trainer = tf.train.GradientDescentOptimizer(learning_rate=0.1)
# updateModel = trainer.minimize(loss)

# with tf.Session() as sess:
#     for i_episode in range(num_ep):
#         observation = env.reset()
#         sess.run(tf.global_variables_initializer())
#         total_reward = 0
#         total_loss = 0
#         for t in range(ep_time):
#             env.render()
#             # Choose state
#             a_t, q_t = sess.run([a_max, q], feed_dict={inputs: observation})
#             if np.random.rand(1) < eps:
#                 a_t[0] = env.action_space.sample()
#             observation_next, reward, done, info = env.step(a_t[0])
#             # Get value of the state we just entered
#             qn_t = sess.run([q], feed_dict={inputs: observation_next})
#             qn_max = np.max(qn_t)
#             # Update model according to TD
#             target_q = q_t
#             target_q[0][a_t[0]] = reward + lmbd * qn_max
#             _, loss_t= sess.run([updateModel, loss], feed_dict={inputs: observation, new_q: target_q})
#             # Move to next state
#             observation = observation_next
#             # Record reward
#             total_reward += reward
#             total_loss += loss_t
#             if done:
#                 print("Timesteps: {} ".format(t+1))
#                 print("Total loss: {} ".format(total_loss))
#                 eps = 1./(1000 * (i_episode/num_ep) + 10)
#                 print(eps)
#                 reward_list.append(total_reward)
#                 loss_list.append(total_loss)
#                 time_list.append(t)
#                 break
# plt.subplot(311)
# plt.plot(reward_list)
# plt.subplot(312)
# plt.plot(time_list)
# plt.subplot(313)
# plt.plot(loss_list)
# plt.show()