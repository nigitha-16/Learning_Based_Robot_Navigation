import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist, PoseStamped
from tf2_msgs.msg import TFMessage
import numpy as np
import tensorflow as tf
import math
import transforms3d

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
    point_homogeneous = np.array([point[0], point[1], point[2], 1.0])
    
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

class VelocityPredictorNode(Node):
    def __init__(self):
        super().__init__('velocity_predictor')

        # Load the trained TensorFlow model
        self.model = tf.keras.models.load_model('path_to_your_model.h5')

        # Subscribe to laser scan, odometry, and goal position data
        self.laser_sub = self.create_subscription(
            LaserScan,
            '/scan',
            self.laser_callback,
            10)
        
        self.odom_sub = self.create_subscription(
            Odometry,
            '/odom',
            self.odom_callback,
            10)
        
        self.tf_sub = self.create_subscription(
            TFMessage,
            '/tf',
            self.tf_callback,
            10)
        
        self.goal_sub = self.create_subscription(
            PoseStamped,
            '/goal',
            self.goal_callback,
            10)
        
        # Publisher for the velocity
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)

        # Data storage
        self.laser_data = None
        self.odom_data = None
        self.goal_position = None  # (x, y) relative to the robot's frame
    
    def laser_callback(self, msg):
        """Callback for laser scan data"""
        # Convert laser scan to numpy array
        ranges = np.array(msg.ranges, dtype=np.float32)
        ranges = np.nan_to_num(ranges, nan=10.0, posinf=10.0, neginf=0.0)
        self.laser_data = ranges
        self.predict_velocity()
        
    def tf_callback(self, msg):
        """Callback for tf data"""
        for transform in msg.transforms:
            # Check if the transform is from odom to base_link
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
        self.linear_velocity = msg.twist.twist.linear
        self.angular_velocity = msg.twist.twist.angular

        # Create odom data array
        self.odom_data = np.array([
            self.robot_position[0], self.robot_position[1],
            self.robot_orientation.x, self.robot_orientation.y, self.robot_orientation.z, self.robot_orientation.w,
            self.linear_velocity.x, self.linear_velocity.y, self.linear_velocity.z,
            self.angular_velocity.x, self.angular_velocity.y, self.angular_velocity.z
        ], dtype=np.float32)
        
        self.predict_velocity()

    def goal_callback(self, msg):
        """Callback for goal position in odom frame (PoseStamped)"""
        goal_x = msg.pose.position.x
        goal_y = msg.pose.position.y

        # Compute goal position relative to robot
        self.goal_position = self.transform_goal_to_robot_frame(goal_x, goal_y)
        self.predict_velocity()

   



    # Function to transform a point from odom to base_link frame
    def transform_pose_to_base_link(translation, rotation, point_in_odom):
        # Get the inverse of the transformation
        translation_inv, rotation_inv = invert_transform(translation, rotation)
        
        # Convert the point from odom to base_link using the inverted transformation
        point_in_base_link = transform_point(translation_inv, rotation_inv, point_in_odom)
        
        return point_in_base_link
    def predict_velocity(self):
        """Predict velocity using the model"""
        # Ensure we have all necessary data: laser, odom, and goal
        if self.laser_data is None or self.odom_data is None or self.goal_position is None:
            return
        
        # Combine laser, odom, and goal data into a single input for the model
        input_data = np.concatenate([self.laser_data, self.odom_data, self.goal_position])
        
        # Add batch dimension (as TensorFlow models expect batches of data)
        input_data = np.expand_dims(input_data, axis=0)
        
        # Predict the velocity (linear and angular)
        predicted_velocity = self.model.predict(input_data)
        
        # Prepare and publish Twist message
        twist_msg = Twist()
        twist_msg.linear.x = predicted_velocity[0, 0]  # Predicted linear velocity
        twist_msg.angular.z = predicted_velocity[0, 1]  # Predicted angular velocity
        
        self.cmd_vel_pub.publish(twist_msg)
        self.get_logger().info(f"Published Velocity - Linear: {twist_msg.linear.x}, Angular: {twist_msg.angular.z}")

def main(args=None):
    rclpy.init(args=args)
    node = VelocityPredictorNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
