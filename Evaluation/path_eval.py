import rclpy
from rclpy.node import Node
from nav_msgs.msg import Path, OccupancyGrid
from tf2_msgs.msg import TFMessage
from std_msgs.msg import Header
import numpy as np
import tensorflow as tf
import math
import transforms3d
import os
import time
import math
from tf2_ros import Buffer, TransformListener, LookupException, ConnectivityException, ExtrapolationException
import tf2_geometry_msgs

class ExtractPath():
    def __init__(self, bag_file, output_dir):
        self.bag_file = bag_file
        self.output_dir = output_dir
        

        # Containers for data
        self.path = None
        self.map = None
        self.map_resolution = None
        self.map_origin = None

        # Process messages
        self.process_bag(bag_file)

    def process_bag(self, bag_file):
        storage_options = rosbag2_py.StorageOptions(uri=bag_file, storage_id='sqlite3')
        converter_options = rosbag2_py.ConverterOptions(input_serialization_format='cdr', output_serialization_format='cdr')
        reader = rosbag2_py.SequentialReader()
        reader.open(storage_options, converter_options)
        while reader.has_next():
            topic, msg, t = reader.read_next()

            if topic == "/robot_path":
                self.process_path(msg, t)
            elif topic == "/map":
                self.process_map(msg, t)

    def process_path(self, msg, timestamp):
        path_msg = deserialize_message(msg, Path)
        self.path = path_msg

    def process_map(self, msg, timestamp):
        msg_ = deserialize_message(msg, OccupancyGrid)
        # Extract metadata
        self.map_resolution = msg.info.resolution
        width = msg.info.width
        height = msg.info.height
        self.map_origin = msg.info.origin

        # Extract the occupancy data
        self.map = np.array(msg.data).reshape((height, width))
        
    
