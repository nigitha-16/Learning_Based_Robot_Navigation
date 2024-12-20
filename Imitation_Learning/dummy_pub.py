import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist, PoseStamped, TransformStamped
from tf2_msgs.msg import TFMessage
import numpy as np
import transforms3d
import math
import time

class DataPublisherNode(Node):
    def __init__(self):
        super().__init__('data_publisher')

        # Publishers for laser scan, odometry, and goal
        self.scan_pub = self.create_publisher(LaserScan, '/scan', 10)
        self.odom_pub = self.create_publisher(Odometry, '/odom', 10)
        self.goal_pub = self.create_publisher(PoseStamped, '/goal', 10)
        self.tf_pub = self.create_publisher(TFMessage, '/tf', 10)

        self.timer = self.create_timer(0.5, self.publish_data)  # Publish data every 0.5 seconds

        # Initialize some data
        self.pose_x = 0.0
        self.pose_y = 0.0
        self.pose_theta = 0.0

    def publish_data(self):
        # Publish laser scan data
        # self.scan_data = np.linspace(0, 2 * np.pi, 513)  # Simulated scan data (360 degrees)
        # self.scan_ranges = np.random.uniform(0.5, 5.0, 513)
        self.scan_data = np.linspace(0, 2 * np.pi, 513)  # Simulated scan data (360 degrees)
        self.scan_ranges = np.random.uniform(0.01, 1, 513)
        laser_msg = LaserScan()
        laser_msg.header.stamp = self.get_clock().now().to_msg()
        laser_msg.header.frame_id = 'base_link'
        laser_msg.angle_min = -np.pi
        laser_msg.angle_max = np.pi
        laser_msg.angle_increment = np.pi / 180.0  # 1 degree increments
        laser_msg.time_increment = 0.0
        laser_msg.range_min = 0.0
        laser_msg.range_max = 10.0
        laser_msg.ranges = self.scan_ranges.tolist()  # Convert to list for ROS message
        laser_msg.intensities = []  # Not used
        self.scan_pub.publish(laser_msg)

        # Publish odometry data
        odom_msg = Odometry()
        odom_msg.header.stamp = self.get_clock().now().to_msg()
        odom_msg.header.frame_id = 'odom'
        odom_msg.child_frame_id = 'base_link'
        odom_msg.pose.pose.position.x = self.pose_x
        odom_msg.pose.pose.position.y = self.pose_y
        odom_msg.pose.pose.position.z = 0.0
        odom_msg.twist.twist.linear.x = 0.5  # Example linear velocity
        odom_msg.twist.twist.angular.z = 0.0  # Example angular velocity
        self.odom_pub.publish(odom_msg)

        # Publish goal data
        goal_msg = PoseStamped()
        goal_msg.header.stamp = self.get_clock().now().to_msg()
        goal_msg.header.frame_id = 'odom'
        goal_msg.pose.position.x = self.pose_x + 2.0  # Goal position 2 units ahead
        goal_msg.pose.position.y = self.pose_y
        goal_msg.pose.position.z = 0.0
        goal_msg.pose.orientation.w = 1.0  # No rotation
        self.goal_pub.publish(goal_msg)

        # Publish TF data
        tf_msg = TFMessage()
        tf_transform = self.create_tf_transform(self.pose_x, self.pose_y, self.pose_theta)
        tf_msg.transforms.append(tf_transform)
        self.tf_pub.publish(tf_msg)

        # Update pose for the next iteration
        self.pose_x += 0.1 * np.cos(self.pose_theta)
        self.pose_y += 0.1 * np.sin(self.pose_theta)
        self.pose_theta += 0.1  # Simulate turning

    def create_tf_transform(self, x, y, theta):
        transform = TransformStamped()
        transform.header.stamp = self.get_clock().now().to_msg()
        transform.header.frame_id = 'odom'
        transform.child_frame_id = 'base_link'
        transform.transform.translation.x = x
        transform.transform.translation.y = y
        transform.transform.translation.z = 0.0
        return transform

def main(args=None):
    rclpy.init(args=args)
    node = DataPublisherNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
