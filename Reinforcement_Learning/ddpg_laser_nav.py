import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, Model
import random
from collections import deque

from Reinforcement_Learning.robile_gym_env import RobileEnv

# Define the Actor Network (Policy)
class Actor(tf.keras.Model):
    def __init__(self, model_path):
        super(Actor, self).__init__()
        self.model = tf.keras.models.load_model(model_path, compile=False)

    def call(self, laser, goal):
        return self.model([laser, goal])

# Define the Critic Network (Q-value function)
class Critic(tf.keras.Model):
    def __init__(self, laser_dim, goal_dim, action_dim):
        super(Critic, self).__init__()

        # Laser input processing
        self.laser_input = layers.Input(shape=(laser_dim,))
        self.laser_fc = layers.Dense(256, activation='relu')(self.laser_input)
        self.laser_fc = layers.BatchNormalization()(self.laser_fc)
        self.laser_fc = layers.LeakyReLU()(self.laser_fc)
        self.laser_fc = layers.Dense(128, activation='relu')(self.laser_fc)

        # Goal input processing
        self.goal_input = layers.Input(shape=(goal_dim,))
        self.goal_fc = layers.Dense(128, activation='relu')(self.goal_input)
        self.goal_fc = layers.BatchNormalization()(self.goal_fc)
        self.goal_fc = layers.LeakyReLU()(self.goal_fc)
        self.goal_fc = layers.Dense(64, activation='relu')(self.goal_fc)

        # Action input processing
        self.action_input = layers.Input(shape=(action_dim,))
        self.action_fc = layers.Dense(64, activation='relu')(self.action_input)
        self.action_fc = layers.BatchNormalization()(self.action_fc)
        self.action_fc = layers.LeakyReLU()(self.action_fc)

        # Concatenate processed laser, goal, and action outputs
        merged_input = layers.Concatenate()([self.laser_fc, self.goal_fc, self.action_fc])

        # Feed-forward layers for the critic after merging
        hidden = layers.Dense(256, activation='relu')(merged_input)
        hidden = layers.BatchNormalization()(hidden)
        hidden = layers.LeakyReLU()(hidden)
        hidden = layers.Dense(128, activation='relu')(hidden)
        hidden = layers.BatchNormalization()(hidden)
        hidden = layers.LeakyReLU()(hidden)
        hidden = layers.Dense(64, activation='relu')(hidden)
        hidden = layers.BatchNormalization()(hidden)
        hidden = layers.LeakyReLU()(hidden)

        # Output layer (Q-value)
        self.q_value = layers.Dense(1)(hidden)

        # Create the model
        self.model = Model(inputs=[self.laser_input, self.goal_input, self.action_input], outputs=self.q_value)

    def call(self, laser, goal, action):
        return self.model([laser, goal, action])

        

# Experience Replay Buffer
class ReplayBuffer:
    def __init__(self, buffer_size, batch_size):
        self.buffer = deque(maxlen=buffer_size)
        self.batch_size = batch_size

    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state[0], state[1], action, reward, next_state[0], next_state[1], done))

    def sample(self):
        return random.sample(self.buffer, self.batch_size)

    def size(self):
        return len(self.buffer)

# DDPG Agent
class DDPGAgent:
    def __init__(self, laser_dim, goal_dim, action_dim, pretrained_model_path, buffer_size=100000, batch_size=3, gamma=0.99, tau=0.005, lr=1e-3):
        self.actor = Actor(pretrained_model_path)
        self.critic = Critic(laser_dim, goal_dim, action_dim)
        self.target_actor = Actor(pretrained_model_path)
        self.target_critic = Critic(laser_dim, goal_dim, action_dim)

        # Initialize target networks to be the same as the original networks
        self.target_actor.set_weights(self.actor.get_weights())
        self.target_critic.set_weights(self.critic.get_weights())

        self.actor_optimizer = optimizers.Adam(lr)
        self.critic_optimizer = optimizers.Adam(lr)
        
        self.buffer = ReplayBuffer(buffer_size, batch_size)
        self.gamma = gamma  # Discount factor for future rewards
        self.tau = tau  # Target network update rate

    def update(self):
        if self.buffer.size() < self.buffer.batch_size:
            return

        # Sample a batch of transitions from the replay buffer
        state_laser, state_goal, action, reward, next_state_laser, next_state_goal, done = zip(*self.buffer.sample())
        state = [np.array(state_laser), np.array(state_goal)]
        action = np.array(action)
        reward = np.array(reward)
        next_state = [np.array(next_state_laser), np.array(next_state_goal)]
        done = np.array(done)
        
        target_action = self.target_actor(next_state[0], next_state[1])
        target_q_value = self.target_critic(next_state[0], next_state[1], target_action)
        target = reward + (1 - done) * self.gamma * target_q_value
        # print('target', target)
        

        # Update Critic
        with tf.GradientTape() as tape:        
            current_q_value = self.critic(state[0], state[1], action)
            critic_loss = tf.reduce_mean(tf.square(current_q_value - target))

        critic_grads = tape.gradient(critic_loss, self.critic.trainable_variables)
        # print('trainable params')
        # print([v.name for v in self.critic.trainable_variables])
        # print('current_q_value', current_q_value)
        # print("critic_loss:", critic_loss)
        # print("critic_grads:", critic_grads)
        
        # self.critic_optimizer.apply_gradients(zip(critic_grads, self.critic.trainable_variables))

        # Update Actor
        with tf.GradientTape() as tape:
            actor_loss = -tf.reduce_mean(self.critic(state[0], state[1], self.actor(state[0], state[1])))

        actor_grads = tape.gradient(actor_loss, self.actor.trainable_variables)
        self.actor_optimizer.apply_gradients(zip(actor_grads, self.actor.trainable_variables))

        # Soft update target networks
        self.soft_update(self.actor, self.target_actor)
        self.soft_update(self.critic, self.target_critic)

    def soft_update(self, source, target):
        source_weights = source.get_weights()
        target_weights = target.get_weights()

        new_weights = []
        for source_w, target_w in zip(source_weights, target_weights):
            new_weights.append(self.tau * source_w + (1.0 - self.tau) * target_w)

        target.set_weights(new_weights)

        

# Training Loop
def train_ddpg(env, agent, num_episodes=1000, max_timesteps=20):
    for episode in range(num_episodes):
        state = env.reset()
        episode_reward = 0
        for t in range(max_timesteps):
            state_vector = np.expand_dims(state[0], axis=0), np.expand_dims(state[1], axis=0)
            action = agent.actor(state_vector[0], state_vector[1])[0]
            next_state, reward, done, _ = env.step(action)

            # Store transition in the replay buffer
            agent.buffer.add(state, action, reward, next_state, done)
            print('action', action)

            # Update agent
            agent.update()

            state = next_state
            episode_reward += reward

            if done:
                break

        print(f"Episode {episode}, Reward: {episode_reward}")

laser_shape = 513
goal_shape = 2
motion_command_shape = 3
model_path= 'models/model_laser_corr07112024_rescaled_outputs_a_model_epoch_200.keras'

agent = DDPGAgent(laser_dim=laser_shape, goal_dim=goal_shape, action_dim=motion_command_shape, 
                pretrained_model_path=model_path) 

# Train the agent
env = RobileEnv()
train_ddpg(env, agent)
