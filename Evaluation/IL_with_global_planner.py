from std_msgs.msg import Header
from tf2_ros import LookupException, ConnectivityException, ExtrapolationException
import tf2_geometry_msgs
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSDurabilityPolicy
from tf_transformations import quaternion_from_euler, euler_from_quaternion
from tf2_geometry_msgs import do_transform_pose
from geometry_msgs.msg import PoseStamped, Twist
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseWithCovarianceStamped
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import OccupancyGrid
from nav_msgs.msg import Path
import tf2_ros
import numpy as np
from math import isinf, sin, cos, atan2, floor, ceil
import random
import pandas as pd
import cv2
import copy
import time
import threading
import sys
import select
import tensorflow as tf
import math


# Gives path given start and end position wrt  to map frame
class AStarPlanner:
    def __init__(self, init_position, goal_position, map_msg):
        self.init_position = init_position
        self.goal_position = goal_position
        self.init_state = None
        self.goal_state = None
        self.map = map_msg
        self.map_grid = None
        self.waypoint_res = 30
        self.waypoints = []
        self.robot_width = 0.3

    def find_path(self):
        # get start and end cells
        print('start state')
        self.init_state = self.get_nearest_mapcell(self.init_position)
        print('end state')
        self.goal_state = self.get_nearest_mapcell(self.goal_position)
        map_grid = np.array(self.map.data).reshape((self.map.info.height, self.map.info.width))
        self.map_grid = self.inflate_occupied_cells(map_grid)
        cv2.imwrite("tingg.png", self.map_grid)
        path = self.astar_search()
        # get significant waypoints. right now gets every nth point
        path = path[::self.waypoint_res]
        print('********PATH**************')
        # convert cells to position wrt map frame
        for cell in path:
            self.waypoints.append(self.get_mapcell_coord(cell))
        return self.waypoints

    # dilate the map to account for robot width
    def inflate_occupied_cells(self, map_grid):
        map_new = copy.deepcopy(map_grid)
        rows, cols = map_grid.shape
        n = ceil(self.robot_width / self.map.info.resolution)
        for i in range(rows):
            for j in range(cols):
                if map_grid[i][j] > 10:
                    for layer in range(1, n + 1):
                        for dx in range(-layer, layer + 1):
                            for dy in range(-layer, layer + 1):
                                new_i, new_j = i + dx, j + dy
                                if 0 <= new_i < rows and 0 <= new_j < cols:
                                    map_new[new_i][new_j] = 100
        return map_new

    # euclidian dist to goal
    def heuristic(self, state):
        return ((state[0] - self.goal_state[0]) ** 2 + (state[1] - self.goal_state[1]) ** 2) ** 0.5

    # get next possible cells the robot can move to
    def get_possible_moves(self, state):
        possible_moves = []
        x_lim = self.map_grid.shape[0]
        y_lim = self.map_grid.shape[1]
        # up
        if (state[1] + 1 < y_lim) and (state[1] + 1 > 0):
            if (self.map_grid[state[0]][state[1] + 1] < 10):
                possible_moves.append([state[0], state[1] + 1])
        # down
        if (state[1] - 1 < y_lim) and (state[1] - 1 > 0):
            if (self.map_grid[state[0]][state[1] - 1] < 10):
                possible_moves.append([state[0], state[1] - 1])
        # right
        if (state[0] + 1 < x_lim) and (state[0] + 1 > 0):
            if (self.map_grid[state[0] + 1][state[1]] < 10):
                possible_moves.append([state[0] + 1, state[1]])
        # left
        if (state[0] - 1 < x_lim) and (state[0] - 1 > 0):
            if (self.map_grid[state[0] - 1][state[1]] < 10):
                possible_moves.append([state[0] - 1, state[1]])
        # up right
        if (state[0] + 1 < x_lim) and (state[0] + 1 > 0) and (state[1] + 1 < y_lim) and (state[1] + 1 > 0):
            if (self.map_grid[state[0] + 1][state[1] + 1] < 10):
                possible_moves.append([state[0] + 1, state[1] + 1])
        # up left
        if (state[0] - 1 < x_lim) and (state[0] - 1 > 0) and (state[1] + 1 < y_lim) and (state[1] + 1 > 0):
            if (self.map_grid[state[0] - 1][state[1] + 1] < 10):
                possible_moves.append([state[0] - 1, state[1] + 1])
        # down right
        if (state[0] + 1 < x_lim) and (state[0] + 1 > 0) and (state[1] - 1 < y_lim) and (state[1] - 1 > 0):
            if (self.map_grid[state[0] + 1][state[1] - 1] < 10):
                possible_moves.append([state[0] + 1, state[1] - 1])
        # down left
        if (state[0] - 1 < x_lim) and (state[0] - 1 > 0) and (state[1] - 1 < y_lim) and (state[1] - 1 > 0):
            if (self.map_grid[state[0] - 1][state[1] - 1] < 10):
                possible_moves.append([state[0] - 1, state[1] - 1])
        return possible_moves

    # get the nearest map cell given the position wrt map frame
    def get_nearest_mapcell(self, position):
        # print('position', position)
        x_cell = floor((position[1] - self.map.info.origin.position.y) / self.map.info.resolution)
        y_cell = floor((position[0] - self.map.info.origin.position.x) / self.map.info.resolution)
        # print(x_cell, y_cell)
        return [x_cell, y_cell]

    # get coordinate of a cell wrt map frame
    def get_mapcell_coord(self, state):
        x = (state[1] * self.map.info.resolution) + self.map.info.origin.position.x
        y = (state[0] * self.map.info.resolution) + self.map.info.origin.position.y
        return (x, y)

    # search for path from start state to goal state
    def astar_search(self):
        open_nodes = []
        open_nodes_f = []
        open_nodes_state = []
        explored_states = []
        g = 0
        path = []
        # start with the initial state (cell)
        init_node = {'state': self.init_state, 'g': 0, 'parent': None}
        open_nodes_f.append(self.heuristic(self.init_state))
        open_nodes.append(init_node)
        open_nodes_state.append(init_node['state'])
        while True:
            # exit if there are no more nodes to explore
            if len(open_nodes) <= 0:
                break
            # get the node with least f value
            current_node = open_nodes.pop(0)
            current_f = open_nodes_f.pop(0)
            # exit if goal reached
            if self.goal_state == current_node['state']:
                break
            g = current_node['g'] + 1
            explored_states.append(current_node['state'])
            # get next possible states for robot to move
            possible_states = self.get_possible_moves(current_node['state'])
            possible_states = [state for state in possible_states if state not in explored_states]
            possible_states = [state for state in possible_states if state not in open_nodes_state]
            h_list = [self.heuristic(state) for state in possible_states]
            # f(n) = h(n)+ g(n)
            f_list = [h + g for h in h_list]
            # add all the next possible puzzle configs to open list
            for f, state in zip(f_list, possible_states):
                open_nodes.append({'state': state, 'g': g, 'parent': current_node})
                open_nodes_f.append(f)
                open_nodes_state.append(state)
            # sort based on f value
            open_nodes = pd.Series(data=open_nodes, index=open_nodes_f).sort_index().tolist()
            open_nodes_f = sorted(open_nodes_f)
        # backtrack the path
        path.append(current_node['state'])
        while True:
            parent = current_node['parent']
            if parent is None:
                break
            path.append(parent['state'])
            current_node = parent
        return path


class IL_Planner(Node):
    def __init__(self, model_path):
        super().__init__('il_planner')
        self.init_x = None
        self.init_y = None
        self.final_goal_x = None
        self.final_goal_y = None
        self.final_goal_x_wrt_map = None
        self.final_goal_y_wrt_map = None
        self.init_x_wrt_map = None
        self.init_y_wrt_map = None
        self.goal_pose = None

        self.map_msg = None
        self.way_points = None
        
        self.model = tf.keras.models.load_model(model_path, compile=False)

        qos_profile = QoSProfile(
            depth=10,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL
        )

        self.map_subscriber = self.create_subscription(OccupancyGrid, '/map', self.map_callback, qos_profile)
        self.amcl_pose_subscriber = self.create_subscription(PoseWithCovarianceStamped, '/amcl_pose',
                                                             self.amcl_pose_callback, qos_profile)
        self.goal_pose_subscriber = self.create_subscription(PoseStamped, '/goal_pose',
                                                             self.goal_pose_callback, 10)
        self.pose_subscriber = self.create_subscription(PoseWithCovarianceStamped, '/pose', self.pose_callback, 1)
        self.scan_subscriber = self.create_subscription(LaserScan, '/scan', self.scan_callback, 10)
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.astar_path_publisher = self.create_publisher(Path, '/astar', 10)
        self.odom_sub = self.create_subscription(Odometry, '/odom', self.odom_callback, 1)
        self.path_pub = self.create_publisher(Path, '/robot_path', 10)
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        
        self.goal_dist_error_thresh = 0.3

        self.got_amcl = False
        
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

    def odom_callback(self, msg):
        """Callback for odometry data"""
        # Extract position and orientation from odometry message
        self.robot_position = (msg.pose.pose.position.x, msg.pose.pose.position.y)
        self.robot_orientation = msg.pose.pose.orientation
        
    # create a goal pose msg wrt odom frame
    def create_goal_pose(self, x, y, theta=0):
        goal_quaternion = quaternion_from_euler(0, 0, theta)
        goal_pose = PoseStamped()
        goal_pose.header.frame_id = 'odom'
        goal_pose.pose.position.x = x
        goal_pose.pose.position.y = y
        goal_pose.pose.orientation.x = goal_quaternion[0]
        goal_pose.pose.orientation.y = goal_quaternion[1]
        goal_pose.pose.orientation.w = goal_quaternion[2]
        goal_pose.pose.orientation.z = goal_quaternion[3]
        return goal_pose


    def map_callback(self, msg):
        if self.map_msg is None:
            print("*****************got map")
        self.map_msg = msg
        # self.destroy_subscription(self.map_subscriber)

    def amcl_pose_callback(self, msg):
        # get the initial pose of robot after localization
        if self.init_x is None:
            print('got amcl position')
        self.init_x = msg.pose.pose.position.x
        self.init_y = msg.pose.pose.position.y
        self.init_x_wrt_map = self.init_x
        self.init_y_wrt_map = self.init_y
        self.got_amcl = True

    def pose_callback(self, msg):
        if self.tf_buffer.can_transform("map", "odom", rclpy.time.Time().to_msg()):
            # print('ab')
            # get the initial pose of robot after localization
            if not self.got_amcl:
                # print('got pose')
                self.init_x = msg.pose.pose.position.x
                self.init_y = msg.pose.pose.position.y
                odom_to_map_tf = self.tf_buffer.lookup_transform("map", "odom", rclpy.time.Time().to_msg())
                init_pose_wrt_map = do_transform_pose(msg.pose.pose, odom_to_map_tf)
                self.init_x_wrt_map = init_pose_wrt_map.position.x
                self.init_y_wrt_map = init_pose_wrt_map.position.y

    def goal_pose_callback(self, msg):
        # get the goal pose and then call AStar to get path(sequence of intermediate goal poses)
        print('new goal')
        self.final_goal_x = msg.pose.position.x
        self.final_goal_y = msg.pose.position.y
        self.final_goal_x_wrt_map = self.final_goal_x
        self.final_goal_y_wrt_map = self.final_goal_y
        if self.map_msg is not None:
            init_position = [self.init_x_wrt_map, self.init_y_wrt_map]
            goal_position = [self.final_goal_x_wrt_map, self.final_goal_y_wrt_map]
            astar_obj = AStarPlanner(init_position, goal_position, self.map_msg)
            way_points = astar_obj.find_path()
            self.way_points = []
            for wp in way_points:
                pose = PoseStamped()
                pose.header.frame_id = 'map'
                pose.header.stamp = self.get_clock().now().to_msg()
                pose.pose.position.x = wp[0]
                pose.pose.position.y = wp[1]
                pose.pose.orientation.w = 1.0
                self.way_points.append([wp[0], wp[1], 1.0])
            # orientation of final goal pose
            self.way_points[0][2] = msg.pose.orientation.w
            self.publish_astar_path()
            x, y, w = self.way_points.pop(-1)
            self.goal_pose = self.create_goal_pose(x, y, w)

    def publish_astar_path(self):
        path_msg = Path()
        path_msg.header.frame_id = 'map'
        for wp in self.way_points:
            pose = PoseStamped()
            pose.header.frame_id = 'map'
            pose.header.stamp = self.get_clock().now().to_msg()
            pose.pose.position.x = wp[0]
            pose.pose.position.y = wp[1]
            pose.pose.orientation.w = wp[2]
            path_msg.poses.append(pose)
        self.astar_path_publisher.publish(path_msg)
        
    def record_path(self):
        """Record the robot's position and orientation in the path message"""
        if self.robot_position and self.robot_orientation:
            pose_stamped = PoseStamped()
            pose_stamped.header = Header()
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

    def scan_callback(self, msg):
        
        ranges = np.array(msg.ranges, dtype=np.float32)
        ranges = np.nan_to_num(ranges, nan=100.0, posinf=100.0, neginf=100.0)
        self.laser_data = ranges
        self.predict_velocity()
        
    def predict_velocity(self):
        """Predict velocity using the model"""
        # Ensure we have all necessary data: laser, odom, and goal
    
        if self.laser_data is None:
            
            return
        if self.goal_pose is None:
            return
        if self.tf_buffer.can_transform("base_link", "map", rclpy.time.Time().to_msg()):
            map_to_base_link_tf = self.tf_buffer.lookup_transform("base_link", "map", rclpy.time.Time().to_msg())
            goal_wrt_base_link = do_transform_pose(self.goal_pose.pose, map_to_base_link_tf)
            
            goal_distance = np.sqrt(goal_wrt_base_link.position.x ** 2 + goal_wrt_base_link.position.y** 2)
            goal_angle = math.atan2(goal_wrt_base_link.position.y,goal_wrt_base_link.position.x)
            self.goal_data_rel = [goal_distance, goal_angle]
        else:
            return
        
        if goal_distance > self.goal_dist_error_thresh:
            
            # print('goal ', self.goal_data_rel)
            laser_input = np.expand_dims(self.laser_data, axis=0)  # Add batch dimension
            goal_input = np.expand_dims(self.goal_data_rel, axis=0)    # Add batch dimension

            # Pass the inputs to the model
            predicted_velocity = self.model.predict([laser_input, goal_input])[0]*2
                    
            # print('predicted_velocity ', predicted_velocity)
            # Prepare and publish Twist message
            twist_msg = Twist()
            twist_msg.linear.x = float(predicted_velocity[0]) # Predicted linear velocity
            twist_msg.linear.y = float(predicted_velocity[1])# Predicted linear velocity
            twist_msg.angular.z = float(predicted_velocity[2])  # Predicted angular velocity
            
            theta = math.atan2(predicted_velocity[1], predicted_velocity[0])        
            # Update the direction based on angular velocity
            theta_new = theta + predicted_velocity[2]         
            # Normalize theta_new to the range [-pi, pi]
            theta_new = math.atan2(math.sin(theta_new), math.cos(theta_new))
            # print('angle of movement', theta_new)
        else:
            
            if len(self.way_points) > 0:
                print('reached intermediate goal')
                x, y, w = self.way_points.pop(-1)
                self.goal_pose = self.create_goal_pose(x, y, w)
                print('next goal ', x, y)
            else:
                print('reached goal')
                
            twist_msg = Twist()
            twist_msg.linear.x = 0.0
            twist_msg.linear.y = 0.0
            twist_msg.angular.z = 0.0
            
        self.cmd_vel_pub.publish(twist_msg)
        self.record_path()
        self.path_pub.publish(self.path_msg)
        time.sleep(0.3)

def listen_for_keypress(node):
    """Check for keyboard input without blocking."""
    while rclpy.ok():
        # Check if there's input available
        if sys.stdin in select.select([sys.stdin], [], [], 0)[0]:
            key = sys.stdin.read(1)  # Read 1 character
            if key.lower() == 'q':  # If 'q' is pressed
                node.path_pub.publish(node.path_msg)  # Publish the final recorded path
                break
        time.sleep(0.1)  # Avoid high CPU usage
        
def main():
    rclpy.init()
    il_planner = IL_Planner('/home/nigitha/ros2_ws_rnd/src/ImitationLearning_models/model_laser_corr07112024_a_model_epoch_200.keras')
    
    key_listener_thread = threading.Thread(target=listen_for_keypress, args=(il_planner,), )
    key_listener_thread.daemon = True  # Daemonize the thread to ensure it stops with the main program
    key_listener_thread.start()

    # # Keep the node running
    rclpy.spin(il_planner)
    il_planner.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()