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
        print(path)
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
        print('position', position)
        x_cell = floor((position[1] - self.map.info.origin.position.y) / self.map.info.resolution)
        y_cell = floor((position[0] - self.map.info.origin.position.x) / self.map.info.resolution)
        print(x_cell, y_cell)
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


class PotentialFieldPlanner(Node):
    def __init__(self):
        super().__init__('potential_field_planner')
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
        self.vel_publisher = self.create_publisher(Twist, '/cmd_vel', 10)
        self.astar_path_publisher = self.create_publisher(Path, '/astar', 10)
        self.odom_sub = self.create_subscription(Odometry, '/odom', self.odom_callback, 1)
        self.path_pub = self.create_publisher(Path, '/robot_path', 10)
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self.ka = 0.05  # 0.05
        self.kr = 0.03  # 0.01

        self.max_velocity = 0.1  # 0.1
        self.min_velocity = 0.05  # 0.0
        self.max_angular_velocity = 0.05  # 0.05
        self.repulsive_threshold_dist = 0.3
        self.goal_dist_error_thresh = 0.3
        self.goal_theta_error_thresh = 0.8

        self.got_amcl = False
        
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

    def polar2cart(self, polar_coords):
        cartesian_coords = []
        for dist, theta in polar_coords:
            if not isinf(dist) and dist> 0.05:
                x = dist * cos(theta)
                y = dist * sin(theta)
                cartesian_coords.append(np.array([x, y]))
        return np.array(cartesian_coords)

    def attr_vel(self, goal_position):
        attr_vel = self.ka * goal_position / np.linalg.norm(goal_position)
        print('attraction', attr_vel)
        return attr_vel

    def rep_vel(self, obstacles):
        rep_vel = 0
        for obs in obstacles:
            dist = np.linalg.norm(obs)
            if dist < self.repulsive_threshold_dist:
                rep_vel -= self.kr * ((1 / dist) - (1 / self.repulsive_threshold_dist)) * (1 / dist ** 2) * (obs / dist)
        print('repulsion', rep_vel)
        return rep_vel

    def move_to_goal(self, obs_coords, goal_pose):
        goal_position = np.array([goal_pose.x, goal_pose.y])
        linear_vel = self.attr_vel(goal_position) + self.rep_vel(obs_coords)
        
        # print('total vel', linear_vel)
        # move randomly if total vel is very less and robot stops
        if abs(linear_vel[0]) < 0.01 and abs(linear_vel[1]) < 0.01:
            linear_vel[0] = random.uniform(0.2, self.max_velocity)
            linear_vel[1] = random.uniform(0.2, self.max_velocity)

        vel_command = Twist()
        vel_command.linear.x = np.clip(linear_vel[0], -self.max_velocity,
                                       self.max_velocity)
        vel_command.linear.y = np.clip(linear_vel[1], -self.max_velocity,
                                       self.max_velocity)
        vel_command.angular.z = np.clip(atan2(vel_command.linear.y, vel_command.linear.x), -self.max_angular_velocity,
                                        self.max_angular_velocity)
        # if the robot is facing an entirely diff direction than the one in which it has to move,
        # then first rotate the robot to that direction
        if abs(atan2(vel_command.linear.y, vel_command.linear.x)) > 0.8:
            vel_command.linear.x = min(self.min_velocity, vel_command.linear.x) if vel_command.linear.x > 0 else max(
                vel_command.linear.x, -self.min_velocity)
            vel_command.linear.y = min(self.min_velocity, vel_command.linear.y) if vel_command.linear.y > 0 else max(
                vel_command.linear.y, -self.min_velocity)

        print(vel_command.linear.x, vel_command.linear.y, vel_command.angular.z)
        return vel_command

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
            print('ab')
            # get the initial pose of robot after localization
            if not self.got_amcl:
                print('got pose')
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
        if self.goal_pose is None:
            return
        # transform goal pose from odom to base_link
        if self.tf_buffer.can_transform("base_link", "map", rclpy.time.Time().to_msg()):
            map_to_base_link_tf = self.tf_buffer.lookup_transform("base_link", "map", rclpy.time.Time().to_msg())
            goal_wrt_base_link = do_transform_pose(self.goal_pose.pose, map_to_base_link_tf)
            goal_theta_wrt_base_link = euler_from_quaternion(
                [goal_wrt_base_link.orientation.x, goal_wrt_base_link.orientation.y,
                 goal_wrt_base_link.orientation.z, goal_wrt_base_link.orientation.w])[2]
            # calculate distance to goal
            dist_to_goal = (goal_wrt_base_link.position.x ** 2 + goal_wrt_base_link.position.y ** 2) ** 0.5
            # if robot has not reached goal, calculate linear velocity and publish
            if dist_to_goal > self.goal_dist_error_thresh:
                # print('away from goal')
                # get obstacles cartesian coords
                laser_angles = np.arange(msg.angle_min, msg.angle_max, msg.angle_increment)
                laser_polar_coords = zip(msg.ranges, laser_angles)
                obs_cart_coords = self.polar2cart(laser_polar_coords)
                vel_command = self.move_to_goal(obs_cart_coords, goal_wrt_base_link.position)
            # if robot has reached goal, but orientation is misaligned, then calculate the angular velocity and publish
            elif (abs(goal_theta_wrt_base_link) > self.goal_theta_error_thresh) and (len(self.way_points) == 0):
                # print('should adjust pose')
                if goal_theta_wrt_base_link > 0:
                    angular_vel = min(goal_theta_wrt_base_link, self.max_angular_velocity)
                else:
                    angular_vel = max(goal_theta_wrt_base_link, -self.max_angular_velocity)
                vel_command = Twist()
                vel_command.linear.x = 0.0
                vel_command.linear.y = 0.0
                vel_command.angular.z = angular_vel
            # if goal pose reached, stop the robot
            else:
                if len(self.way_points) > 0:
                    print('reached intermediate goal')
                    x, y, w = self.way_points.pop(-1)
                    self.goal_pose = self.create_goal_pose(x, y, w)
                    print('next goal ', x, y)
                    vel_command = Twist()
                    vel_command.linear.x = 0.0
                    vel_command.linear.y = 0.0
                    vel_command.angular.z = 0.0
                else:
                    print('reached goal')
                    vel_command = Twist()
                    vel_command.linear.x = 0.0
                    vel_command.linear.y = 0.0
                    vel_command.angular.z = 0.0

            self.vel_publisher.publish(vel_command)
            self.record_path()

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
    potential_field_planner = PotentialFieldPlanner()
    
    key_listener_thread = threading.Thread(target=listen_for_keypress, args=(potential_field_planner,), )
    key_listener_thread.daemon = True  # Daemonize the thread to ensure it stops with the main program
    key_listener_thread.start()

    # # Keep the node running
    rclpy.spin(potential_field_planner)
    potential_field_planner.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()