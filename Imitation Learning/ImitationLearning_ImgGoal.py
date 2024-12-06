from sklearn.model_selection import train_test_split
import pickle
import os
import h5py
from PIL import Image
import numpy as np
import datetime
import math
import random
import yaml
import argparse
import transforms3d
from tensorflow.keras.models import Model
import tensorflow as tf
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.layers import GlobalAveragePooling2D, BatchNormalization, LeakyReLU, Add, Flatten, Dense, concatenate, Rescaling, Normalization, Conv2D, MaxPooling2D, Dropout
from tensorflow.keras.initializers import HeNormal
from tensorflow.keras import Model, Input, regularizers
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from sklearn.model_selection import train_test_split

class DatasetLoader:
    def __init__(self, tfrecord_file, image_shape=[224, 224, 3], lasers_shape=513, goal_shape=2, motion_command_shape=3):
        self.tfrecord_file = tfrecord_file
        self.image_shape = image_shape
        self.lasers_shape = lasers_shape
        self.goal_shape = goal_shape
        self.motion_command_shape = motion_command_shape
        self.dataset_length = self._get_dataset_length()

    def _parse_function_with_images(self, proto):
        features = {
            'image': tf.io.FixedLenFeature([], tf.string),  # Raw serialized image
            'goal': tf.io.FixedLenFeature([self.goal_shape], tf.float32),
            'motion_command': tf.io.FixedLenFeature([self.motion_command_shape], tf.float32)
        }
        
        # Parse the input `proto` based on the defined features
        parsed_features = tf.io.parse_single_example(proto, features)    

        # Deserialize and reshape the raw image tensor
        image = tf.io.parse_tensor(parsed_features['image'], out_type=tf.uint8)
        image = tf.reshape(image, self.image_shape)
        
        return (image, parsed_features['goal']), parsed_features['motion_command']

    def _get_dataset_length(self):
        dataset = tf.data.TFRecordDataset(self.tfrecord_file)
        metadata_features = {
            'metadata': tf.io.FixedLenFeature([], tf.string, default_value=''),
            'length': tf.io.FixedLenFeature([], tf.int64, default_value=-1)
        }
        
        for record in dataset.take(1):
            parsed_features = tf.io.parse_single_example(record, metadata_features)
            dataset_length = parsed_features['length'].numpy()
            return dataset_length
        
        print("Metadata not found. Setting length to None.")
        return None

    def load_dataset(self):
        dataset = tf.data.TFRecordDataset(self.tfrecord_file)
        dataset = dataset.skip(1).map(self._parse_function_with_images, num_parallel_calls=tf.data.AUTOTUNE)
        return dataset

    def split_dataset(self, dataset, train_size=0.7, val_size=0.2):
        num_elements = self.dataset_length
        if num_elements is None:
            raise ValueError("Dataset length is not set. Ensure the metadata is properly included in the TFRecord.")
        
        train_end = int(train_size * num_elements)
        val_end = int((train_size + val_size) * num_elements)
        
        train_dataset = dataset.take(train_end)
        val_dataset = dataset.skip(train_end).take(val_end - train_end)
        test_dataset = dataset.skip(val_end).take(num_elements - val_end)
        
        return train_dataset, val_dataset, test_dataset

    def preprocess_and_augment(self, dataset_, batch_size=128):
        dataset_= dataset_.shuffle(buffer_size=10000).batch(batch_size).repeat()
        dataset_ = dataset_.prefetch(tf.data.AUTOTUNE)
        return dataset_

    def get_prepared_datasets(self, train_size=0.7, val_size=0.2, batch_size=128):
        dataset = self.load_dataset()
        
        train_dataset, val_dataset, test_dataset = self.split_dataset(dataset, train_size, val_size)
        del dataset
        train_dataset = self.preprocess_and_augment(train_dataset, batch_size)
        val_dataset = self.preprocess_and_augment(val_dataset, batch_size)
        test_dataset = self.preprocess_and_augment(test_dataset, batch_size)
        
        return train_dataset, val_dataset, test_dataset

class CustomExponentialDecay(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, initial_learning_rate, decay_steps, decay_rate, minimum_learning_rate, staircase=True):
        self.initial_learning_rate = initial_learning_rate
        self.decay_steps = decay_steps
        self.decay_rate = decay_rate
        self.minimum_learning_rate = minimum_learning_rate
        self.staircase = staircase
        self.base_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=initial_learning_rate,
            decay_steps=decay_steps,
            decay_rate=decay_rate,
            staircase=staircase
        )

    def __call__(self, step):
        lr = self.base_schedule(step)
        return tf.maximum(lr, self.minimum_learning_rate)

    def get_config(self):
        return {
            'initial_learning_rate': self.initial_learning_rate,
            'decay_steps': self.decay_steps,
            'decay_rate': self.decay_rate,
            'minimum_learning_rate': self.minimum_learning_rate,
            'staircase': self.staircase
        }

class ModelCheckpointEveryN(Callback):
    def __init__(self, save_dir, model_name, save_freq=25):
        """
        A custom callback that saves the model every 'save_freq' epochs.
        
        :param save_dir: Directory to save the model.
        :param save_freq: Number of epochs between each save.
        """
        super(ModelCheckpointEveryN, self).__init__()
        self.save_dir = save_dir
        self.save_freq = save_freq
        self.model_name = model_name

    def on_epoch_end(self, epoch, logs=None):
        # Save the model every 'save_freq' epochs
        if (epoch + 1) % self.save_freq == 0:
            model_save_path = os.path.join(self.save_dir, f"{self.model_name}_model_epoch_{epoch+1}.keras")
            print(f"\nEpoch {epoch+1}: Saving model to {model_save_path}")
            self.model.save(model_save_path)

class MotionCommandModel:
    def __init__(self, image_shape, goal_shape, motion_command_shape, log_dir):
        self.image_shape = image_shape
        self.goal_shape = goal_shape
        self.motion_command_shape = motion_command_shape
        self.log_dir = log_dir

        # Define normalization layers
        self.image_rescaling = Rescaling(1./255)
        self.goal_normalization = Normalization()

        # Initialize the model
        self.model = self._create_model()
        
    
    def residual_block(self, x, units):
        shortcut = x
        x = Dense(units, kernel_initializer=HeNormal())(x)
        x = BatchNormalization()(x)
        x = LeakyReLU()(x)

        # Adjust shortcut if dimensions differ
        if shortcut.shape[-1] != units:
            shortcut = Dense(units, kernel_initializer=HeNormal())(shortcut)
        
        x = Add()([x, shortcut])
        return x

    def _create_model(self):
        # Define inputs
        goal_input = Input(shape=(self.goal_shape,), name='goal_input')
        image_input = Input(shape=self.image_shape, name='image_input')

        # Normalize inputs
        goal_normalized = self.goal_normalization(goal_input)
        image_rescaled = self.image_rescaling(image_input)

        # Process goal with dense and residual layers
        goal_hidden = Dense(8, kernel_initializer=HeNormal())(goal_normalized)
        goal_hidden = BatchNormalization()(goal_hidden)
        goal_hidden = LeakyReLU()(goal_hidden)
        goal_hidden = self.residual_block(goal_hidden, 16)
        goal_hidden = Dense(16, kernel_initializer=HeNormal())(goal_hidden)  # Original third layer
        goal_hidden = BatchNormalization()(goal_hidden)
        goal_hidden = LeakyReLU()(goal_hidden)

        # Load and freeze base model
        base_model = MobileNetV2(include_top=False, weights='imagenet', input_tensor=image_rescaled)
        base_model.trainable = False
        
        # Choose an intermediate layer from MobileNetV2
        intermediate_layer = base_model.get_layer('block_15_expand_relu').output
    
        # Add custom convolutional layers on top of the intermediate output
        x = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', 
                   padding='same')(intermediate_layer)
        x = BatchNormalization()(x)
        x = LeakyReLU()(x)
        x = Dropout(0.2)(x)
        
        x = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', 
                   padding='same')(x)
        x = BatchNormalization()(x)
        x = LeakyReLU()(x)
        x = Dropout(0.2)(x)
        
        x = Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', 
                   padding='same')(x)
        x = BatchNormalization()(x)
        x = LeakyReLU()(x)

    
        # Global Average Pooling to reduce spatial dimensions
        image_hidden = GlobalAveragePooling2D()(x)
    
        # Process the pooled features with dense layers
        image_hidden = Dense(128, kernel_initializer='he_normal')(image_hidden)
        image_hidden = Dropout(0.2)(image_hidden)
        image_hidden = BatchNormalization()(image_hidden)
        image_hidden = LeakyReLU()(image_hidden)
        image_hidden = Dropout(0.2)(image_hidden)
        image_hidden = Dense(64, kernel_initializer='he_normal')(image_hidden)
        image_hidden = BatchNormalization()(image_hidden)
        image_hidden = LeakyReLU()(image_hidden)
        image_hidden = Dropout(0.2)(image_hidden)
        image_hidden = Dense(32, kernel_initializer='he_normal')(image_hidden)
        image_hidden = BatchNormalization()(image_hidden)
        image_hidden = LeakyReLU()(image_hidden)

        # Concatenate processed features
        concatenated = concatenate([goal_hidden, image_hidden])
        hidden = Dense(128, kernel_initializer=HeNormal())(concatenated)
        hidden = BatchNormalization()(hidden)
        hidden = LeakyReLU()(hidden)
        hidden = Dense(64, kernel_initializer=HeNormal())(hidden)
        hidden = BatchNormalization()(hidden)
        hidden = LeakyReLU()(hidden)
        hidden = Dense(16, kernel_initializer=HeNormal())(hidden)
        hidden = BatchNormalization()(hidden)
        hidden = LeakyReLU()(hidden)

        # Output layer
        output = Dense(self.motion_command_shape, activation='linear', name='motion_command_output')(hidden)
        
        # Compile the model
        model = Model(inputs=[image_input, goal_input], outputs=output)
        return model

    def compile_model(self, initial_learning_rate=0.005, decay_steps=5000, decay_rate=0.96, minimum_learning_rate=0.001):
        lr_schedule = CustomExponentialDecay(
            initial_learning_rate=initial_learning_rate,
            decay_steps=decay_steps,
            decay_rate=decay_rate,
            minimum_learning_rate=minimum_learning_rate)
        
        optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
        self.model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])

    class PrintLearningRateCallback(Callback):
        def on_epoch_end(self, epoch, logs=None):
            lr = self.model.optimizer.learning_rate
            if isinstance(lr, tf.keras.optimizers.schedules.LearningRateSchedule):
                lr = lr(self.model.optimizer.iterations)
            print(f"\nEpoch {epoch+1}: Learning rate is {tf.keras.backend.eval(lr)}")
    

    def train_model(self, train_dataset, val_dataset, epochs, train_steps, val_steps, initial_learning_rate, decay_steps,
                    decay_rate, minimum_learning_rate, model_save_path, model_name):
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=self.log_dir, histogram_freq=1)
        self.compile_model(initial_learning_rate, decay_steps, decay_rate, minimum_learning_rate)
        model_checkpoint_callback = ModelCheckpointEveryN(save_dir=model_save_path, model_name=model_name, save_freq=25)

        history = self.model.fit(
            train_dataset,
            epochs=epochs,
            validation_data=val_dataset,
            steps_per_epoch=train_steps,  # Specify steps per epoch for training
            validation_steps=val_steps,
            verbose=1,
            callbacks=[self.PrintLearningRateCallback(), tensorboard_callback, model_checkpoint_callback]
        )
        return history

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
    model_save_path = os.path.join(root_dir, config['model_save_path'])
    model_name = config['model_name']

    loader = DatasetLoader(tf_file)
    train_dataset, val_dataset, test_dataset = loader.get_prepared_datasets(train_size = config['train_size'], val_size = config['val_size'],
                                                                           batch_size = config['batch_size'])

    print('Training, validation, and test datasets created and preprocessed.')

    # Usage example
    image_shape = (224, 224, 3)
    goal_shape = 2
    motion_command_shape = 3

    model_instance = MotionCommandModel(image_shape, goal_shape, motion_command_shape, log_dir)
    history = model_instance.train_model(train_dataset, val_dataset, epochs=config['epochs'],
                                         train_steps = config['train_steps'], val_steps = config['val_steps'],
                                        initial_learning_rate = config['initial_learning_rate'], decay_steps = config['decay_steps'],
                                         decay_rate = config['decay_rate'], minimum_learning_rate = config['minimum_learning_rate'],
                                        model_save_path = model_save_path, model_name = model_name)

    print("Model training complete.")



