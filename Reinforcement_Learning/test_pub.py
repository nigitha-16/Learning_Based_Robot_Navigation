import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist, PoseStamped, TransformStamped
from nav_msgs.msg import Odometry
from tf2_msgs.msg import TFMessage
from std_msgs.msg import Header
import numpy as np
import math
import time


count = 0

class TestPublisher(Node):
    def __init__(self):
        super().__init__('test_publisher')

        # Publishers for testing
        self.laser_pub = self.create_publisher(LaserScan, '/scan', 10)
        # self.goal_pub = self.create_publisher(PoseStamped, '/goal_pose', 10)
        self.odom_pub = self.create_publisher(Odometry, '/odom', 10)
        self.tf_pub = self.create_publisher(TFMessage, '/tf', 10)

        # Timers to periodically publish messages
        self.laser_timer = self.create_timer(0.5, self.publish_laser_scan)
        # self.goal_timer = self.create_timer(3.0, self.publish_goal_pose)
        self.odom_timer = self.create_timer(0.1, self.publish_odometry)
        self.tf_timer = self.create_timer(0.1, self.publish_tf)

        # Initialize example data
        self.goal_x = 5.0
        self.goal_y = 5.0
        self.robot_x = 0.0
        self.robot_y = 0.0
        self.robot_theta = 0.0

    def publish_laser_scan(self):
        """Publish mock laser scan data."""
        global count
        msg = LaserScan()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'base_link'
        msg.angle_min = -math.pi
        msg.angle_max = math.pi
        msg.angle_increment = math.pi / 180.0  # 1 degree
        msg.range_min = 0.1
        msg.range_max = 10.0
        num_readings = int((msg.angle_max - msg.angle_min) / msg.angle_increment)
        msg.ranges = [5 + 0.5 * np.sin(i * 0.1) for i in range(513)]
        msg.ranges[0] = count
        count= count+1
        self.laser_pub.publish(msg)

    # def publish_goal_pose(self):
    #     """Publish a mock goal position."""
    #     msg = PoseStamped()
    #     msg.header.stamp = self.get_clock().now().to_msg()
    #     msg.header.frame_id = 'odom'
    #     msg.pose.position.x = self.goal_x
    #     msg.pose.position.y = self.goal_y
    #     msg.pose.position.z = 0.0
    #     msg.pose.orientation.w = 1.0
    #     self.goal_pub.publish(msg)

    def publish_odometry(self):
        """Publish mock odometry data."""
        msg = Odometry()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'odom'
        msg.child_frame_id = 'base_link'
        msg.pose.pose.position.x = self.robot_x
        msg.pose.pose.position.y = self.robot_y
        msg.pose.pose.orientation.z = math.sin(self.robot_theta / 2.0)
        msg.pose.pose.orientation.w = math.cos(self.robot_theta / 2.0)
        self.odom_pub.publish(msg)

    def publish_tf(self):
        """Publish mock TF data."""
        tf_msg = TFMessage()
        transform = TransformStamped()
        transform.header.stamp = self.get_clock().now().to_msg()
        transform.header.frame_id = 'map'
        transform.child_frame_id = 'odom'
        transform.transform.translation.x = 0.0
        transform.transform.translation.y = 0.0
        transform.transform.translation.z = 0.0
        transform.transform.rotation.w = 1.0
        tf_msg.transforms.append(transform)

        transform2 = TransformStamped()
        transform2.header.stamp = self.get_clock().now().to_msg()
        transform2.header.frame_id = 'odom'
        transform2.child_frame_id = 'base_link'
        transform2.transform.translation.x = self.robot_x
        transform2.transform.translation.y = self.robot_y
        transform2.transform.translation.z = 0.0
        transform2.transform.rotation.z = math.sin(self.robot_theta / 2.0)
        transform2.transform.rotation.w = math.cos(self.robot_theta / 2.0)
        tf_msg.transforms.append(transform2)

        self.tf_pub.publish(tf_msg)

def main(args=None):
    rclpy.init(args=args)
    node = TestPublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
