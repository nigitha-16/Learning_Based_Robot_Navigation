import gymnasium as gym
from gymnasium import spaces
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist, PoseStamped
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist, PoseStamped
from nav_msgs.msg import Path
from tf2_msgs.msg import TFMessage
from std_msgs.msg import Header
import tensorflow as tf
import math
import transforms3d
import os
import time
import math
import time


# Function to invert a transformation (translation + rotation)
def invert_transform(translation, rotation):
    # Convert the quaternion to a rotation matrix (3x3)
    rotation_matrix = transforms3d.quaternions.quat2mat([rotation[3], rotation[0], rotation[1], rotation[2]])[:3, :3]
    # Invert the rotation matrix (transpose of a rotation matrix is its inverse)
    rotation_matrix_inv = np.transpose(rotation_matrix)
    
    # Invert the translation (apply the inverse rotation to the negative translation)
    translation_inv = -np.dot(rotation_matrix_inv, [translation[0], translation[1], translation[2]])
    
    # Create the inverse quaternion (negate the vector part, keep the scalar part the same)
    rotation_inv = transforms3d.quaternions.qinverse([rotation[3], rotation[0], rotation[1], rotation[2]])
    return translation_inv, rotation_inv

# Function to apply the transformation (translation + rotation) to a point
def transform_point(translation, rotation, point):
    # Convert the point to a homogeneous vector (x, y, z, 1)
    point_homogeneous = np.array([point[0], point[1], 0.0, 1.0])
    
    # Create the translation matrix (4x4)
    translation_matrix = np.identity(4)
    translation_matrix[0, 3] = translation[0]
    translation_matrix[1, 3] = translation[1]
    translation_matrix[2, 3] = translation[2]
    # Create the transformation matrix from translation and rotation
    rotation_matrix = np.identity(4)
    rotation_matrix[:3, :3] = transforms3d.quaternions.quat2mat(rotation)[:3, :3]
    # Combine translation and rotation into a single transformation matrix
    transformation_matrix = np.dot(translation_matrix, rotation_matrix)
    # Apply the transformation to the point
    transformed_point = np.dot(transformation_matrix, point_homogeneous)
    
    
    # Return the transformed point (x, y, z)
    return transformed_point[:3]

def transform_pose(translation, rotation, point_in_odom):
    # Get the inverse of the transformation
    translation_inv, rotation_inv = invert_transform(translation, rotation)
    
    # Convert the point from odom to base_link using the inverted transformation
    point_transformed = transform_point(translation_inv, rotation_inv, point_in_odom)
    
    return point_transformed

class RobileNode(Node):
    def __init__(self):
        super().__init__("RobileNode")

        # ROS2 Subscribers
        self.laser_sub = self.create_subscription(
            LaserScan,
            '/scan',
            self.laser_callback,
            1)
        self.goal_sub = self.create_subscription(
            PoseStamped,
            '/goal_pose',
            self.goal_callback,
            10)
        
        self.odom_sub = self.create_subscription(
            Odometry,
            '/odom',
            self.odom_callback,
            1)
        
        self.tf_sub = self.create_subscription(
            TFMessage,
            '/tf',
            self.tf_callback,
            1)

        # Publisher for the velocity
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        # Publisher for the path
        self.path_pub = self.create_publisher(Path, '/robot_path', 10)

        self.laser_data = None
        self.robot_position = None
        self.robot_orientation = None
        self.goal_data = None  
        self.translation = None
        self.rotation =None
        self.translation_odom_map = None
        self.rotation_odom_map =None
        self.goal_data_rel = None
        self.path = []  # Store robot's path
        self.path_msg = Path()  # Path message to be published
        self.path_msg.header.frame_id = 'map'  # Assuming the path is in the "map" frame

    def laser_callback(self, msg):
        """Callback for laser scan data"""
        # Convert laser scan to numpy array
        ranges = np.array(msg.ranges, dtype=np.float32)
        ranges = np.nan_to_num(ranges, nan=100.0, posinf=100.0, neginf=100.0)
        self.laser_data = ranges
        print('laser', self.laser_data[0])
        
        
    def tf_callback(self, msg):
        """Callback for tf data"""
        for transform in msg.transforms:
            # Check if the transform is from odom to base_link
            if transform.child_frame_id == 'odom' and transform.header.frame_id == 'map':
                # Extract translation and rotation
                translation = transform.transform.translation
                self.translation_odom_map = [translation.x, translation.y, translation.z]
                rotation = transform.transform.rotation
                self.rotation_odom_map = [rotation.x, rotation.y, rotation.z, rotation.w]
            if self.translation_odom_map is not None:
                if transform.child_frame_id == 'base_link' and transform.header.frame_id == 'odom':
                    # Extract translation and rotation
                    translation = transform.transform.translation
                    self.translation = [translation.x, translation.y, translation.z]
                    rotation = transform.transform.rotation
                    self.rotation = [rotation.x, rotation.y, rotation.z, rotation.w]

        
    def odom_callback(self, msg):
        """Callback for odometry data"""
        # Extract position and orientation from odometry message
        self.robot_position = (msg.pose.pose.position.x, msg.pose.pose.position.y)
        self.robot_orientation = msg.pose.pose.orientation
        # self.linear_velocity = msg.twist.twist.linear
        # self.angular_velocity = msg.twist.twist.angular   

    def goal_callback(self, msg):
        """Callback for goal position in odom frame (PoseStamped)"""
        self.goal_data = [msg.pose.position.x, msg.pose.position.y]
        print('goal_data', self.goal_data)

    def send_command_to_robot(self, linear_x, linear_y, angular_z):
        """
        Publish velocity commands to the robot.
        """
        twist_msg = Twist()
        twist_msg.linear.x = float(linear_x) # Predicted linear velocity
        twist_msg.linear.y = float(linear_y)# Predicted linear velocity
        twist_msg.angular.z = float(angular_z)  # Predicted angular velocity        
        self.cmd_vel_pub.publish(twist_msg)
        self.record_path()
        self.path_pub.publish(self.path_msg)
        
    def record_path(self):
        """Record the robot's position and orientation in the path message"""
        if self.robot_position and self.robot_orientation:
            pose_stamped = PoseStamped()
            pose_stamped.header.frame_id = "odom"
            pose_stamped.header.stamp = self.get_clock().now().to_msg()
            pose_stamped.pose.position.x = self.robot_position[0]
            pose_stamped.pose.position.y = self.robot_position[1]
            pose_stamped.pose.position.z = 0.0  # Assuming a 2D path
            pose_stamped.pose.orientation = self.robot_orientation
            pose_stamped.pose = self.transform_pose_to_map(pose_stamped.pose)
            self.path_msg.poses.append(pose_stamped)

            # Optionally, limit the path size to avoid excessive memory usage
            max_path_length = 10000
            if len(self.path_msg.poses) > max_path_length:
                self.path_msg.poses.pop(0)
                
    def transform_pose_to_map(self, odom_pose):
        try:
            # Lookup the transform from map to odom
            transform = self.tf_buffer.lookup_transform(
                'map',  # Target frame
                'odom',  # Source frame
                rclpy.time.Time(),  # Use the latest transform
                timeout=rclpy.duration.Duration(seconds=1.0)
            )
    
            # Transform the odom pose into the map frame
            map_pose = tf2_geometry_msgs.do_transform_pose(odom_pose, transform)
            return map_pose
        except (LookupException, ConnectivityException, ExtrapolationException) as e:
            self.get_logger().error(f"Failed to transform pose: {e}")


class RobileEnv(gym.Env):
    def __init__(self):
        super(RobileEnv, self).__init__()
        
        # Initialize ROS2
        rclpy.init()
        self.node = RobileNode()

        # Define observation space 
        laser_dim = 513
        goal_dim = 2
        self.laser_space = spaces.Box(low=-np.inf, high=np.inf, shape=(laser_dim,), dtype=np.float32)
        self.goal_space = spaces.Box(low=-np.inf, high=np.inf, shape=(goal_dim,), dtype=np.float32)


        # Define action space (linear_x, linear_y, angular_z)
        self.action_space = spaces.Box(
            low=np.array([-1.0, -1.0, -1.0]),  # Define action limits
            high=np.array([1.0, 1.0, 1.0]),
            dtype=np.float32
        )
        
        self.empty_laser = np.full(self.laser_space.shape, np.nan, dtype=np.float32)
        self.empty_goal = np.full(self.goal_space.shape, np.nan, dtype=np.float32)

    def step(self, action):
        """
        Perform an action, collect the next state, reward, and done flag.
        """
        # Send action to the robot
        linear_x, linear_y, angular_z = action
        self.node.send_command_to_robot(linear_x, linear_y, angular_z)
        time.sleep(2)
        # Wait for new data
        self.node.laser_data = None
        self.node.translation = None
        observation = [self.empty_laser, self.empty_goal]
        while np.any(np.isnan(observation[0])) or np.any(np.isnan(observation[1])):
            rclpy.spin_once(self.node, timeout_sec=1)
            observation = self.get_robot_state()
            

        # Compute reward
        reward = self.compute_reward(observation)

        # Check if the episode is done
        done = self.check_done(observation)

        # Additional debug info
        info = {}
        return observation, reward, done, info

    def reset(self):
        """
        Give new goal to robot.
        """
        print('Give Goal Data')
        self.node.send_command_to_robot(0.0, 0.0, 0.0)
        self.node.goal_data = None
        observation = [self.empty_laser, self.empty_goal]
        
        while np.any(np.isnan(observation[0])) or np.any(np.isnan(observation[1])):           
            rclpy.spin_once(self.node, timeout_sec=1)
            observation = self.get_robot_state()
        print('received Goal')
        return observation

    def get_robot_state(self):
        """
        Get the combined state of the robot (laser + goal).
        """
        if self.node.laser_data is None or self.node.goal_data is None:
            
            return [self.empty_laser, self.empty_goal]
        if self.node.translation is not None:
            # Compute goal position relative to robot
            goal_position = transform_pose(self.node.translation_odom_map, self.node.rotation_odom_map, self.node.goal_data)
            goal_position = transform_pose(self.node.translation, self.node.rotation, goal_position)
            goal_distance = np.sqrt(goal_position[0] ** 2 + goal_position[1]** 2)
            goal_angle = math.atan2(goal_position[1],goal_position[0])
            self.node.goal_data_rel = np.array([goal_distance, goal_angle])
            print('relative goal', self.node.goal_data_rel)
        else:
            return [self.empty_laser, self.empty_goal]
        return [self.node.laser_data, self.node.goal_data_rel]

    def compute_reward(self, observation):
        """
        Define a reward function based on the robot's state.
        """
        # minimize distance to the goal
        reward = -self.node.goal_data_rel[0]
        return reward

    def check_done(self, observation):
        """
        Check if the episode is done.
        """
        # Example: Done if goal is reached or robot collides
        if self.node.goal_data_rel[0] < 0.3:  # Goal reached
            return True
        if np.any(self.node.laser_data < 0.2):  # Collision threshold
            return True
        return False


    def close(self):
        """
        Clean up ROS2 resources.
        """
        self.node.destroy_node()
        rclpy.shutdown()
