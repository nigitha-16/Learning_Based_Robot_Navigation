import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, LaserScan
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist, PoseStamped
from cv_bridge import CvBridge
import tensorflow as tf
import numpy as np

class ImitationLearningNavNode(Node):
    def __init__(self):
        super().__init__('imitation_learning_nav_node')
        
        # Load the trained model
        self.model = tf.keras.models.load_model('/path/to/your/saved/model')
        
        # Initialize CvBridge to convert ROS Image messages to OpenCV format
        self.bridge = CvBridge()
        
        # Subscriptions to relevant topics
        self.image_sub = self.create_subscription(Image, '/camera/color/image_raw', self.image_callback, 10)
        self.laser_sub = self.create_subscription(LaserScan, '/scan', self.laser_callback, 10)
        self.odom_sub = self.create_subscription(Odometry, '/odom', self.odom_callback, 10)
        self.goal_sub = self.create_subscription(PoseStamped, '/goal_pose', self.goal_callback, 10)

        # Publisher for the predicted velocity commands
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        
        # Initialize containers for the latest sensor data
        self.current_image = None
        self.current_laser = None
        self.current_pose = None
        self.goal_pose = None

    def image_callback(self, msg):
        # Convert the ROS image message to an OpenCV image
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            self.current_image = cv_image
        except Exception as e:
            self.get_logger().error(f"Image conversion failed: {str(e)}")
    
    def laser_callback(self, msg):
        # Store the latest laser scan data
        self.current_laser = np.array(msg.ranges)
    
    def odom_callback(self, msg):
        # Extract the current position (x, y) and orientation (z) from odometry
        position = msg.pose.pose.position
        orientation = msg.pose.pose.orientation
        self.current_pose = np.array([position.x, position.y, orientation.z])
    
    def goal_callback(self, msg):
        # Extract the goal position (x, y) and orientation (z) from the goal pose
        position = msg.pose.position
        orientation = msg.pose.orientation
        self.goal_pose = np.array([position.x, position.y, orientation.z])
    
    def process_inputs(self):
        # Check if all inputs are available
        if self.current_image is None or self.current_laser is None or self.current_pose is None or self.goal_pose is None:
            self.get_logger().info("Waiting for all inputs...")
            return None
        
        # Preprocess the image (resize, normalize, etc.) to fit model input
        input_image = cv2.resize(self.current_image, (224, 224))  # Example: Resize to (224, 224)
        input_image = input_image / 255.0  # Normalize the image to [0, 1]
        
        # Preprocess laser data (you may want to downsample or normalize)
        input_laser = self.current_laser.reshape(1, -1)  # Reshape to match model input
        
        # Preprocess pose data (current pose and goal pose)
        input_pose = self.current_pose.reshape(1, -1)
        input_goal = self.goal_pose.reshape(1, -1)
        
        return [np.expand_dims(input_image, axis=0), input_laser, input_pose, input_goal]

    def predict_velocities(self):
        # Process the inputs and make predictions using the trained model
        inputs = self.process_inputs()
        if inputs is None:
            return
        
        # Predict linear and angular velocities
        predicted_velocities = self.model.predict(inputs)
        
        # Extract linear and angular velocities
        predicted_linear = predicted_velocities[0][0]
        predicted_angular = predicted_velocities[1][0]
        
        # Create a Twist message
        cmd_vel_msg = Twist()
        cmd_vel_msg.linear.x = predicted_linear[0]
        cmd_vel_msg.linear.y = predicted_linear[1]
        cmd_vel_msg.angular.z = predicted_angular
        
        # Publish the predicted velocities
        self.cmd_vel_pub.publish(cmd_vel_msg)

def main(args=None):
    rclpy.init(args=args)
    
    # Create an instance of the ImitationLearningNode
    imitation_node = ImitationLearningNavNode()
    
    # Run the node at a fixed rate
    rate = imitation_node.create_rate(10)  # 10 Hz
    while rclpy.ok():
        rclpy.spin_once(imitation_node)
        imitation_node.predict_velocities()  # Make predictions and publish velocities
        rate.sleep()
    
    imitation_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
