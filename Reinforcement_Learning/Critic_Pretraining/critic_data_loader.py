import tensorflow as tf
from sklearn.model_selection import train_test_split

class DatasetLoader:
    def __init__(self, tfrecord_file, image_shape=[224, 224, 3], lasers_shape=513, goal_shape=2, motion_command_shape=3, reward_shape = 1):
        self.tfrecord_file = tfrecord_file
        self.image_shape = image_shape
        self.lasers_shape = lasers_shape
        self.goal_shape = goal_shape
        self.motion_command_shape = motion_command_shape
        self.reward_shape = reward_shape
        self.dataset_length = self._get_dataset_length()
        print(self.dataset_length)

    def _parse_with_lasers_function(self,proto):
        features = {
            'laser': tf.io.FixedLenFeature([self.lasers_shape], tf.float32),
            'goal': tf.io.FixedLenFeature([self.goal_shape], tf.float32),
            'next_laser': tf.io.FixedLenFeature([self.lasers_shape], tf.float32),
            'next_goal': tf.io.FixedLenFeature([self.goal_shape], tf.float32),
            'motion_command': tf.io.FixedLenFeature([self.motion_command_shape], tf.float32),
            'reward': tf.io.FixedLenFeature([self.reward_shape], tf.float32),
        }
        parsed_features = tf.io.parse_single_example(proto, features)
        return  (parsed_features['laser'], parsed_features['goal'], parsed_features['motion_command'], parsed_features['reward'],
                 parsed_features['next_laser'], parsed_features['next_goal'])

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
        dataset = dataset.skip(1).map(self._parse_with_lasers_function, num_parallel_calls=tf.data.AUTOTUNE)
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

    def get_prepared_datasets(self, train_size=0.7, val_size=0.2):
        dataset = self.load_dataset()
        
        train_dataset, val_dataset, test_dataset = self.split_dataset(dataset, train_size, val_size)
        
        return train_dataset, val_dataset, test_dataset
