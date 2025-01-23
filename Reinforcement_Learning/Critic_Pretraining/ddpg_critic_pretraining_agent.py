import numpy as np
import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.layers import BatchNormalization, LeakyReLU, Add, Flatten, Dense, concatenate, Normalization, Conv2D, MaxPooling2D, Dropout
from tensorflow.keras.initializers import HeNormal
from tensorflow.keras import Model, Input, regularizers, layers, optimizers


# Define the Actor Network (Policy)
class Actor(tf.keras.Model):
    def __init__(self, model_path):
        super(Actor, self).__init__()
        self.model = tf.keras.models.load_model(model_path, compile=False)

    def call(self, laser, goal):
        return self.model([laser, goal])

# Define the Critic Network (Q-value function)
class Critic(tf.keras.Model):
    def __init__(self, laser_shape, goal_shape, motion_command_shape, reward_shape):
        super(Critic, self).__init__()
        self.laser_shape = laser_shape
        self.goal_shape = goal_shape
        self.motion_command_shape = motion_command_shape
        self.reward_shape = reward_shape

        # Define normalization layers
        self.laser_normalization = Normalization()
        self.goal_normalization = Normalization()
        self.reward_normalization = Normalization()

        # Initialize the model
        self.model = self._create_model()
        
        
    
    def _create_model(self):
        """
        Creates a TensorFlow model for processing laser scans, goals, motion commands, and predicting rewards.
        
        :return: tf.keras.Model
        """
        # Define inputs
        laser_input = Input(shape=(self.laser_shape,), name='laser_input')
        goal_input = Input(shape=(self.goal_shape,), name='goal_input')
        motion_command_input = Input(shape=(self.motion_command_shape,), name='motion_command_input')
    
        # Optional normalization
        laser_normalized = self.laser_normalization(laser_input)
        goal_normalized = self.goal_normalization(goal_input)
        motion_command_normalized = self.reward_normalization(motion_command_input)
    
        # Laser processing
        laser_hidden = Dense(64, kernel_initializer=HeNormal(), kernel_regularizer=regularizers.l2(1e-4))(laser_normalized)
        laser_hidden = BatchNormalization()(laser_hidden)
        laser_hidden = LeakyReLU()(laser_hidden)
        laser_hidden = Dropout(0.2)(laser_hidden)
        laser_hidden = Dense(32, kernel_initializer=HeNormal(), kernel_regularizer=regularizers.l2(1e-4))(laser_hidden)
        laser_hidden = BatchNormalization()(laser_hidden)
        laser_hidden = LeakyReLU()(laser_hidden)

         # Goal processing
        goal_hidden = Dense(16, kernel_initializer=HeNormal(), kernel_regularizer=regularizers.l2(1e-4))(goal_normalized)
        goal_hidden = BatchNormalization()(goal_hidden)
        goal_hidden = LeakyReLU()(goal_hidden)
        goal_hidden = Dropout(0.2)(goal_hidden)
        goal_hidden = Dense(32, kernel_initializer=HeNormal(), kernel_regularizer=regularizers.l2(1e-4))(goal_hidden)
        goal_hidden = BatchNormalization()(goal_hidden)
        goal_hidden = LeakyReLU()(goal_hidden)
    
        # Motion command processing
        motion_command_hidden = Dense(8, kernel_initializer=HeNormal(), kernel_regularizer=regularizers.l2(1e-4))(motion_command_normalized)
        motion_command_hidden = BatchNormalization()(motion_command_hidden)
        motion_command_hidden = LeakyReLU()(motion_command_hidden)
        motion_command_hidden = Dropout(0.2)(motion_command_hidden)
        motion_command_hidden = Dense(32, kernel_initializer=HeNormal(), kernel_regularizer=regularizers.l2(1e-4))(motion_command_hidden)
        motion_command_hidden = BatchNormalization()(motion_command_hidden)
        motion_command_hidden = LeakyReLU()(motion_command_hidden)
    
        # Concatenate processed features
        concatenated = concatenate([laser_hidden, goal_hidden, motion_command_hidden])
    
        # Fully connected layers after concatenation
        hidden = Dense(32, kernel_initializer=HeNormal(), kernel_regularizer=regularizers.l2(1e-4))(concatenated)
        hidden = BatchNormalization()(hidden)
        hidden = LeakyReLU()(hidden)
        hidden = Dropout(0.2)(hidden)
        hidden = Dense(16, kernel_initializer=HeNormal(), kernel_regularizer=regularizers.l2(1e-4))(hidden)
        hidden = BatchNormalization()(hidden)
        hidden = LeakyReLU()(hidden)
        hidden = Dropout(0.2)(hidden)
    
        # Output layer for reward
        output = Dense(1, activation='linear', name='reward_output')(hidden)
    
        # Create and return the model
        model = Model(inputs=[laser_input, goal_input, motion_command_input], outputs=output)
        return model

    def call(self, laser, goal, action):
        return self.model([laser, goal, action])


# DDPG Agent
class DDPGAgent:
    def __init__(self, laser_shape, goal_shape, motion_command_shape, reward_shape, pretrained_actor_model_path, gamma=0.95, tau=0.001, lr=1e-3):
        self.actor = Actor(pretrained_actor_model_path)
        self.critic = Critic(laser_shape, goal_shape, motion_command_shape, reward_shape)
        self.target_actor = Actor(pretrained_actor_model_path)
        self.target_critic = Critic(laser_shape, goal_shape, motion_command_shape, reward_shape)

        # Initialize target networks to be the same as the original networks
        self.target_actor.set_weights(self.actor.get_weights())
        self.target_critic.set_weights(self.critic.get_weights())

        self.critic_optimizer = optimizers.Adam(lr)
        
        self.gamma = gamma  # Discount factor for future rewards
        self.tau = tau  # Target network update rate      

            
    def update(self, train_batch):

        # Sample a batch of transitions from the replay buffer
        state_laser, state_goal, action, reward, next_state_laser, next_state_goal = train_batch
        state = [np.array(state_laser), np.array(state_goal)]
        action = np.array(action)
        reward = np.array(reward)
        next_state = [np.array(next_state_laser), np.array(next_state_goal)]
        done = 0
        
        target_action = self.target_actor(next_state[0], next_state[1])
        target_q_value = self.target_critic(next_state[0], next_state[1], target_action)
        target = reward + (1 - done) * self.gamma * target_q_value

        # Update Critic
        with tf.GradientTape() as tape:        
            current_q_value = self.critic(state[0], state[1], action)
            critic_loss = tf.reduce_mean(tf.square(current_q_value - target))

        critic_grads = tape.gradient(critic_loss, self.critic.trainable_variables)
        
        self.critic_optimizer.apply_gradients(zip(critic_grads, self.critic.trainable_variables))


        

        return critic_loss.numpy()

    def soft_update(self, source, target):
        source_weights = source.get_weights()
        target_weights = target.get_weights()

        new_weights = []
        for source_w, target_w in zip(source_weights, target_weights):
            new_weights.append(self.tau * source_w + (1.0 - self.tau) * target_w)

        target.set_weights(new_weights)

    def validate(self, val_dataset, val_steps, batch_size):
        
        val_dataset_= val_dataset.shuffle(buffer_size=100000).batch(batch_size)
        val_dataset_ = val_dataset_.prefetch(tf.data.AUTOTUNE)
        val_critic_loss = []
        step = 0
        for val_batch in val_dataset_:
            state_laser, state_goal, action, reward, next_state_laser, next_state_goal = val_batch
            state = [np.array(state_laser), np.array(state_goal)]
            action = np.array(action)
            reward = np.array(reward)
            next_state = [np.array(next_state_laser), np.array(next_state_goal)]
            done = 0
            target_action = self.target_actor(next_state[0], next_state[1])
            target_q_value = self.target_critic(next_state[0], next_state[1], target_action)
            target = reward + (1 - done) * self.gamma * target_q_value
    
            current_q_value = self.critic(state[0], state[1], action)
            critic_loss = tf.reduce_mean(tf.square(current_q_value - target))
            val_critic_loss.append(critic_loss)
            step = step + 1
            if step >= val_steps:
                break
            
        val_critic_loss = sum(val_critic_loss)/len(val_critic_loss)
        return val_critic_loss.numpy()
 

