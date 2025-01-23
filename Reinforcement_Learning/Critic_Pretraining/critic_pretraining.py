import numpy as np
import tensorflow as tf
import os
import tensorflow.summary as tf_summary
import datetime
import yaml
import argparse
from critic_data_loader import DatasetLoader
from ddpg_critic_pretraining_agent import DDPGAgent

# Training Loop
def train_critic(agent, train_dataset, val_dataset, batch_size, model_save_path, epochs=100, train_steps=1000, val_steps=200):
    print('starting training', flush = True)
    for epoch in range(epochs):
        train_dataset_= train_dataset.shuffle(buffer_size=10000).batch(batch_size)
        train_dataset_ = train_dataset_.prefetch(tf.data.AUTOTUNE)
        step = 0 
        for train_batch in train_dataset_:
            critic_loss = agent.update(train_batch)            
            if step%10==0:
                agent.soft_update(agent.critic, agent.target_critic)

            if epoch%10==0:            
                with writer.as_default():
                    tf_summary.scalar(f"Episode_{epoch} Critic Loss", critic_loss, step=step)
            step = step+1
            if step >= train_steps:
                break
            

        if epoch%25==0:
            model_save_path = os.path.join(model_save_path, f"{model_name}_model_episode_{epoch+1}.keras")
            agent.critic.model.save(model_save_path)
            print('model saved at epoch ', epoch, flush = True)
        if epoch%10==0:
            val_critic_loss = agent.validate(val_dataset, val_steps, batch_size)
            print('validation loss at epoch',epoch, flush = True)
            print(val_critic_loss, flush = True)
            with writer.as_default():
                tf_summary.scalar("Validation Critic Loss", val_critic_loss, step=epoch)
        
        # Log metrics
        with writer.as_default():
            tf_summary.scalar("Critic Loss", critic_loss, step=epoch)

def load_config(config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Train model with configuration from YAML.")
    parser.add_argument("--config", type=str, required=True, help="Path to the YAML configuration file.")
    args = parser.parse_args()
    
    # Load the configuration
    config = load_config(args.config)
    
    root_dir = config['root_dir']
    tf_file = os.path.join(root_dir, config['tfrecord_file'])
    
    log_dir = config['log_dir'] +datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = os.path.join(root_dir, log_dir)
    os.makedirs(log_dir, exist_ok=True)
    
    model_save_path = os.path.join(root_dir, config['model_save_path'])
    model_name = config['model_name']
    os.makedirs(model_save_path, exist_ok=True)
    
    pretrained_actor_model_path = config['pretrained_actor_model_path']
    pretrained_actor_model_path = os.path.join(root_dir, pretrained_actor_model_path)
    
    epochs=config['epochs']
    train_steps = config['train_steps']
    batch_size = config['batch_size']
    val_steps = config['val_steps']
    test_steps = config['test_steps']
    
    laser_shape = 513
    goal_shape = 2
    motion_command_shape = 3
    reward_shape = 1
    
    loader = DatasetLoader(tf_file)
    train_dataset, val_dataset, test_dataset = loader.get_prepared_datasets(train_size = config['train_size'], val_size = config['val_size'])
    
    writer = tf_summary.create_file_writer(log_dir)

    print('Init agent')
    
    agent = DDPGAgent(laser_shape, goal_shape, motion_command_shape, reward_shape, pretrained_actor_model_path=pretrained_actor_model_path) 
    train_critic(agent, train_dataset, val_dataset, batch_size,model_save_path, epochs, train_steps, val_steps)
    test_critic_loss = agent.validate(test_dataset, test_steps, batch_size)
    with writer.as_default():
        tf_summary.scalar("Test Critic Loss", test_critic_loss, step=epochs)
