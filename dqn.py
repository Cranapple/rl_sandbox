import gym
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

env = gym.make('CartPole-v0')
# env = gym.make('Acrobot-v1')
# env = gym.make('MountainCar-v0')
# env = gym.make('MountainCarContinuous-v0')
# env = gym.make('Pendulum-v0')
print(env.observation_space.sample())
print(env.action_space.sample())

# Hyperparams
num_ep = 100
ep_time = 10000
hidden_layer_size = 100
eps = 0.1
lmbd = 0.99

# Graph params
reward_list = []
loss_list = []
time_list = []

# TF Variables.
inputs = tf.placeholder(shape=env.observation_space.sample().shape, dtype=tf.float32)
in_flat = tf.reshape(inputs, [1, -1])
hw = tf.Variable(tf.random_uniform([in_flat.shape[1].value, hidden_layer_size], 0, 0.01))
hb = tf.Variable(tf.zeros([1, hidden_layer_size]))
hw_ac = tf.nn.relu(hw + hb)
w = tf.Variable(tf.random_uniform([hw_ac.shape[1].value, env.action_space.n], 0, 0.01))
b = tf.Variable(tf.zeros([1, env.action_space.n]))
q = tf.matmul(hw_ac, w) + b
# w = tf.Variable(tf.random_uniform([in_flat.shape[1].value, env.action_space.n], 0, 0.01))
# b = tf.Variable(tf.zeros([1, env.action_space.n]))
# q = tf.matmul(in_flat, w) + b
a_max = tf.argmax(q, 1)
new_q = tf.placeholder(shape=q.shape, dtype=tf.float32)
loss = tf.reduce_sum(tf.square(new_q - q))
trainer = tf.train.GradientDescentOptimizer(learning_rate=0.1)
updateModel = trainer.minimize(loss)

with tf.Session() as sess:
    for i_episode in range(num_ep):
        observation = env.reset()
        sess.run(tf.global_variables_initializer())
        total_reward = 0
        total_loss = 0
        for t in range(ep_time):
            env.render()
            # Choose state
            a_t, q_t = sess.run([a_max, q], feed_dict={inputs: observation})
            if np.random.rand(1) < eps:
                a_t[0] = env.action_space.sample()
            observation_next, reward, done, info = env.step(a_t[0])
            # Get value of the state we just entered
            qn_t = sess.run([q], feed_dict={inputs: observation_next})
            qn_max = np.max(qn_t)
            # Update model according to TD
            target_q = q_t
            target_q[0][a_t[0]] = reward + lmbd * qn_max
            _, loss_t= sess.run([updateModel, loss], feed_dict={inputs: observation, new_q: target_q})
            # Move to next state
            observation = observation_next
            # Record reward
            total_reward += reward
            total_loss += loss_t
            if done:
                print("Timesteps: {} ".format(t+1))
                print("Total loss: {} ".format(total_loss))
                eps = 1./(1000 * (i_episode/num_ep) + 10)
                print(eps)
                reward_list.append(total_reward)
                loss_list.append(total_loss)
                time_list.append(t)
                break
plt.subplot(311)
plt.plot(reward_list)
plt.subplot(312)
plt.plot(time_list)
plt.subplot(313)
plt.plot(loss_list)
plt.show()