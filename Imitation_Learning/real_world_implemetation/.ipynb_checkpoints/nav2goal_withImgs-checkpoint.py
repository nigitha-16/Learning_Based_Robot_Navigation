import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist, PoseStamped
from tf2_msgs.msg import TFMessage
from nav_msgs.msg import Path
from std_msgs.msg import Header
import numpy as np
import tensorflow as tf
import math
import transforms3d
import os
import time
from cv_bridge import CvBridge
import cv2
from PIL import Image as pil_image
from tf2_ros import Buffer, TransformListener, LookupException, ConnectivityException, ExtrapolationException
import tf2_geometry_msgs


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

class VelocityPredictorNode(Node):
    def __init__(self, model_path):
        super().__init__('velocity_predictor')
        print(os.getcwd())
        # Load the trained TensorFlow model
        self.model = tf.keras.models.load_model(model_path, compile=False)
        print('loaded model')
        # Subscribe to laser scan, odometry, and goal position data
        self.img_sub = self.create_subscription(
            Image,
            '/camera/camera/color/image_raw',
            self.img_callback,
            1)
        
        self.odom_sub = self.create_subscription(
            Odometry,
            '/odom',
            self.odom_callback,
            1)
        
        self.tf_sub = self.create_subscription(
            TFMessage,
            '/tf',
            self.tf_callback,
            10)
        
        self.goal_sub = self.create_subscription(
            PoseStamped,
            '/goal_pose',
            self.goal_callback,
            10)
        # Publisher for the path
        self.path_pub = self.create_publisher(Path, '/robot_path', 10)
        # Publisher for the velocity
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)

        # Data storage
        self.image = None
        self.robot_position = None
        self.robot_orientation = None
        self.goal_data = None  
        self.translation = None
        self.rotation =None
        self.translation_odom_map = None
        self.rotation_odom_map =None
        self.goal_data_rel = None
        self.bridge = CvBridge()
        self.path = []  # Store robot's path
        self.path_msg = Path()  # Path message to be published
        self.path_msg.header.frame_id = 'map'  # Assuming the path is in the "map" frame
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

    
    def img_callback(self, msg):
        cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        self.image = np.array(pil_image.fromarray(cv_image).resize((224, 224)))
        
        
        
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

                    self.predict_velocity()
                    
                    

        
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
        

    
    def predict_velocity(self):
        """Predict velocity using the model"""
        # Ensure we have all necessary data: laser, odom, and goal
    
        if self.image is None or self.goal_data is None:
            return
        if self.translation is not None:
            # Compute goal position relative to robot
            goal_position = transform_pose(self.translation_odom_map, self.rotation_odom_map, self.goal_data)
            goal_position = transform_pose(self.translation, self.rotation, goal_position)
            goal_distance = np.sqrt(goal_position[0] ** 2 + goal_position[1]** 2)
            goal_angle = math.atan2(goal_position[1],goal_position[0])
            self.goal_data_rel = [goal_distance, goal_angle]
        else:
            return
        print('goal ', self.goal_data_rel)
        image_input = np.expand_dims(self.image, axis=0)  # Add batch dimension
        goal_input = np.expand_dims(self.goal_data_rel, axis=0)    # Add batch dimension

        # Pass the inputs to the model
        predicted_velocity = self.model.predict([image_input, goal_input])[0]*1.5
        
                
        print('predicted_velocity ', predicted_velocity)
        # Prepare and publish Twist message
        twist_msg = Twist()
        twist_msg.linear.x = float(predicted_velocity[0])  # Predicted linear velocity
        twist_msg.linear.y = float(predicted_velocity[1])  # Predicted linear velocity
        twist_msg.angular.z = float(predicted_velocity[2])  # Predicted angular velocity
        
        self.cmd_vel_pub.publish(twist_msg)
        self.record_path()
        self.path_pub.publish(self.path_msg)
        time.sleep(0.3)
        
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
            
def main(args=None):
    rclpy.init(args=args)
    node = VelocityPredictorNode(model_path= 'models/model_img_corr07112024_aug_a_model_epoch_500.keras')
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
