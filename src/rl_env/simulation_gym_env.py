import rospy
import gym
import threading
from collections import deque
import numpy as np
import random
import matplotlib.pyplot as plt
from gym import spaces
from sensor_msgs.msg import Image
from std_msgs.msg import Float64
from kortex_driver.msg import BaseCyclic_Feedback
import time
from std_srvs.srv import Empty
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions.normal import Normal
from numpy import inf
import subprocess
from os import path
import os
from stable_baselines import PPO2
from stable_baselines.common.policies import MlpPolicy
from  stable_baselines.common.vec_env import DummyVecEnv
from tqdm import tqdm
from torch.distributions import MultivariateNormal, Normal
from collections import namedtuple,deque
import matplotlib
torch.autograd.set_detect_anomaly(True)
from torch.utils.tensorboard import SummaryWriter
import datetime 

import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image as PILImage


class KinovaEnv(gym.Env):
    """
    Custom Gym environment for the Kinova robotic arm.
    This environment interfaces with ROS and Gazebo to control the Kinova arm.
    """

    def __init__(self):
        super(KinovaEnv, self).__init__()
        # Define action and observation space
        self.action_dim = 5
       
        self.image_space = spaces.Box(
            low=0,
            high=255,
            shape=(84, 84, 3),
            dtype=np.uint8
        )
        
        self.end_eff_space = spaces.Box(
            low=np.array([-1.0]*3 + [-np.pi]*3 + [0.0]),  # x,y,z + theta_x,y,z + timestep
            high=np.array([1.0]*3 + [np.pi]*3 + [1.0]),
            dtype=np.float32
        )
        
        # Combined observation space
        self.observation_space = spaces.Dict({
            'image': self.image_space,
            'end_eff_space': self.end_eff_space
        })
        
        # Action space: [∆x, ∆y, ∆z, ∆θ, agripper]
        self.action_space = spaces.Box(
            low=np.array([-0.05, -0.05, -0.05, -np.pi/4, 0.0]),
            high=np.array([0.05, 0.05, 0.05, np.pi/4, 1.0]),
            dtype=np.float32
        )
        
        self.robot_name = "my_gen3"

        # Launch ROS core and Gazebo
        port = "11311"
        self.launfile = "spawn_kortex_robot.launch"
        subprocess.Popen(["roscore","-p",port])
        print("roscore launched!")
        rospy.init_node('ppo_controller')
        subprocess.Popen(["roslaunch","-p", "11311","kinova_gazebo","spawn_kortex_robot.launch"])
        print("Gazebo Launched")

        # Initialize ROS subscribers and publishers
        self.states = {image:[],pose:[]}
        self.cartesian_curr_sub = rospy.Subscriber(self.robot_name+'/base_feedback', BaseCyclic_Feedback, self.read_curr_state)
        self.camera_sub = rospy.Subscriber(self.robot+'/camera/image_raw', Image, self.read_image)
      
        self.clear_faults = rospy.ServiceProxy('/' + self.robot_name + '/base/clear_faults', Base_ClearFaults)
        rospy.wait_for_service('/' + self.robot_name + '/base/clear_faults')

        self.read_action = rospy.ServiceProxy('/' + self.robot_name + '/base/read_action', ReadAction)
        rospy.wait_for_service('/' + self.robot_name + '/base/read_action')

        self.execute_action = rospy.ServiceProxy('/' + self.robot_name + '/base/execute_action', ExecuteAction)
        rospy.wait_for_service('/' + self.robot_name + '/base/execute_action')

        self.set_cartesian_reference_frame = rospy.ServiceProxy('/' + self.robot_name + '/control_config/set_cartesian_reference_frame', SetCartesianReferenceFrame)
        rospy.wait_for_service('/' + self.robot_name + '/control_config/set_cartesian_reference_frame')

        self.activate_publishing_of_action_notification = rospy.ServiceProxy('/' + self.robot_name + '/base/activate_publishing_of_action_topic', OnNotificationActionTopic)
        rospy.wait_for_service('/' + self.robot_name + '/base/activate_publishing_of_action_topic')

        # Set up ROS services
        self.unpause = rospy.ServiceProxy("/gazebo/unpause_physics",Empty)
        self.pause = rospy.ServiceProxy("/gazebo/pause_physics",Empty)
        self.reset_proxy = rospy.ServiceProxy("/gazebo/reset_simulation",Empty)
        print("all service and topics called")

        
        # Initialize state variables
        self.states = {image:[],pose:[]}
        self.states_lock = threading.Lock()
        
        # Start ROS callback thread
        self.callback_thread = threading.Thread(target = self.run_callback)
        self.callback_thread.daemon = True
        self.callback_thread.start()

        self.last_action = 0
        self.last_distance = 0

        
        self.w1 = 0.3  # Increase importance of distance to goal
        self.w2 = 0.4  # Slightly reduce progress reward
        self.w3 = 0.1  # Increase action smoothness importance
        self.w4 = 0.05  # Increase energy penalty
        self.w5 = 0.05  # Increase joint limit penalty
        self.w6 = 0.05  # Increase velocity limit penalty
        log_metrics(filename,f"weights for rewards -- w1 : {self.w1}, w2 : {self.w2}, w3 : {self.w3}, w4 : {self.w4}, w5 : {self.w5}, w6 : {self.w6}\n")

    def read_curr_state(self,msg:BaseCyclic_Feedback):
        commanded_tool_pose = {
                'x': msg.base.commanded_tool_pose_x,
                'y': msg.base.commanded_tool_pose_y,
                'z': msg.base.commanded_tool_pose_z,
                'theta_x': msg.base.commanded_tool_pose_theta_x,
                'theta_y': msg.base.commanded_tool_pose_theta_y,
                'theta_z': msg.base.commanded_tool_pose_theta_z
            }

        self.states.pose = commanded_tool_pose

    def read_image(self,msg:Image):
        cv_image = bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        cv_image_rgb = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)

        # Convert to PIL Image
        pil_image = PILImage.fromarray(cv_image_rgb)
        self.states.image = pil_image
        
    # How can these both subscribers be synchronised? or are they already synchronised?



    def example_clear_faults(self):
        try:
            self.clear_faults()
        except rospy.ServiceException:
            rospy.logerr("Failed to call ClearFaults")
            return False
        rospy.loginfo("Cleared the faults successfully")
        rospy.sleep(2.5)
        return True

    def example_home_the_robot(self):
        req = ReadActionRequest()
        req.input.identifier = self.HOME_ACTION_IDENTIFIER
        self.last_action_notif_type = None
        try:
            res = self.read_action(req)
        except rospy.ServiceException:
            rospy.logerr("Failed to call ReadAction")
            return False
        req = ExecuteActionRequest()
        req.input = res.output
        rospy.loginfo("Sending the robot home...")
        try:
            self.execute_action(req)
        except rospy.ServiceException:
            rospy.logerr("Failed to call ExecuteAction")
            return False
        return self.wait_for_action_end_or_abort()

    def example_set_cartesian_reference_frame(self):
        req = SetCartesianReferenceFrameRequest()
        req.input.reference_frame = CartesianReferenceFrame.CARTESIAN_REFERENCE_FRAME_MIXED
        try:
            self.set_cartesian_reference_frame(req)
        except rospy.ServiceException:
            rospy.logerr("Failed to call SetCartesianReferenceFrame")
            return False
        rospy.loginfo("Set the cartesian reference frame successfully")
        rospy.sleep(0.25)
        return True

    def run_callback(self):
        """Run ROS callbacks in a separate thread."""
        rospy.spin()

    

    def get_current_state(self):
        """Get the current state of the robot."""
        with self.states_lock:
            return self.states

    def update_goal_position(self):
        """Update the goal position with a small random perturbation."""
        self.goal_position += np.random.uniform(low=-0.05, high=0.05, size=6)

    def detect_task_board(image):
        """
        Detects a black task board using grayscale thresholding and contour detection.

        Parameters:
        - image: BGR image from the camera.

        Returns:
        - Bounding box (x_min, y_min, x_max, y_max) if detected, else None.
        """
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Apply adaptive thresholding to highlight the dark areas
        thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                    cv2.THRESH_BINARY_INV, 15, 5)

        # Apply morphological operations to remove small noise
        kernel = np.ones((5, 5), np.uint8)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            return None  # No board detected

        # Find the largest contour
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)

        return (x, y, x + w, y + h)  # Bounding box format (x_min, y_min, x_max, y_max)


    def compute_match_score(image, template_path='task_board_template.png'):
        """
        Computes the match score between the input image and the template.
        The score is based on the number of good keypoint matches.

        Parameters:
        - image: The current image to check for the task board.
        - template_path: Path to the template image of the task board.

        Returns:
        - match_score: A float representing the match score (0 to 1).
        - bbox: The bounding box (x_min, y_min, x_max, y_max) of the detected task board.
        """
        # Load the template and convert it to grayscale
        template = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Use ORB detector to find keypoints and descriptors
        orb = cv2.ORB_create()
        kp1, des1 = orb.detectAndCompute(template, None)
        kp2, des2 = orb.detectAndCompute(gray_image, None)

        # Use BFMatcher to match descriptors between image and template
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(des1, des2)

        # Sort matches by distance
        matches = sorted(matches, key=lambda x: x.distance)

        # Calculate the match score (number of good matches / total number of template keypoints)
        good_matches = len(matches)
        total_keypoints = len(kp1)

        # Calculate match score as the ratio of good matches to total keypoints in the template
        match_score = good_matches / total_keypoints if total_keypoints > 0 else 0

        # Estimate the bounding box of the matched area
        if good_matches > 10:  # Check if we have enough matches
            src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

            # Find homography to align the template to the detected region
            M, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

            # Get the size of the template
            h, w = template.shape
            pts = np.float32([[0, 0], [0, h-1], [w-1, h-1], [w-1, 0]]).reshape(-1, 1, 2)
            dst = cv2.perspectiveTransform(pts, M)

            # Get bounding box from transformed points
            x_min, y_min = np.min(dst, axis=0).flatten()
            x_max, y_max = np.max(dst, axis=0).flatten()

            bbox = (int(x_min), int(y_min), int(x_max), int(y_max))
        else:
            bbox = None

        return match_score, bbox


    def compute_reward(image):
        """
        Computes the reward based on the match score and provides a reward value.
        The reward is higher when more of the task board is visible (better match).
        """
        match_score, bbox = compute_match_score(image)
        
        # Calculate the reward
        reward = match_score  # Match score directly used as reward (scaled 0-1)
        
        # If there is no match, reward is zero
        if match_score < 0.1:
            reward = 0
        # Plot results
        plt.figure(figsize=(8, 6))
        plt.imshow(image_rgb)  # Show image in RGB format
        plt.axis("off")

        # Draw bounding box if detected
        if bbox:
            x_min, y_min, x_max, y_max = bbox
            plt.gca().add_patch(plt.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min, 
                                            edgecolor='green', linewidth=3, fill=False))
            plt.title(f"Detected Task Board - Reward: {reward}")
        else:
            plt.title("Task board not detected!")

        plt.show()
        return reward, bbox

    
    def step(self, action):
        """
        Execute one time step within the environment.
        
        Args:
            action (np.array): Action to be executed
        
        Returns:
            tuple: (next_state, reward, done, info)
        """
        try:

            my_cartesian_speed = CartesianSpeed()
            my_cartesian_speed.translation = 0.1
            my_cartesian_speed.orientation = 15

            my_constrained_pose = ConstrainedPose()
            my_constrained_pose.constraint.oneof_type.speed.append(my_cartesian_speed)

            my_constrained_pose.target_pose.x = action[0]
            my_constrained_pose.target_pose.y = action[1]
            my_constrained_pose.target_pose.z = action[2]
            # my_constrained_pose.target_pose.theta_x = float(row['theta_x'])
            # my_constrained_pose.target_pose.theta_y = float(row['theta_y'])
            my_constrained_pose.target_pose.theta_z = action[3]

            req = ExecuteActionRequest()
            req.input.oneof_action_parameters.reach_pose.append(my_constrained_pose)
            req.input.name = f"pose_{i}"
            req.input.handle.action_type = ActionType.REACH_POSE
            req.input.handle.identifier = 1000 + i

            rospy.loginfo(f"Sending pose {i} ...")
            self.last_action_notif_type = None
            try:
                self.execute_action(req)
            except rospy.ServiceException:
                rospy.logerr(f"Failed to send pose {i}")
                return False
            rospy.loginfo(f"Waiting for pose {i} to finish...")
            self.wait_for_action_end_or_abort()

            self.unpause()
            time.sleep(TIME_DELTA)
            self.pause()

            # Update goal and get new state
            self.update_goal_position() 
            current_state = self.get_current_state()
            end_effector_position = current_state.pose
            next_state = np.concatenate((current_state, end_effector_position))

            # Calculate reward and check if done
            distance_to_goal = np.linalg.norm(end_effector_position - self.goal_position)     
            reward = self.compute_reward(self.states.image)
            done = distance_to_goal < 0.3

            self.last_distance = distance_to_goal
            self.last_action = action

            return next_state, reward, done, {}
    
        except Exception as e:
            rospy.logerr(f"Error reading poses from CSV: {e}")
            return False

    def reset(self):
        """
        Reset the environment to an initial state.
        
        Returns:
            np.array: Initial observation
        """
        # Reset simulation
        self.reset_proxy()
        rospy.sleep(1)

        # Set new random goal
        self.goal_position = np.random.uniform(low=-3, high=3, size=6)

        # Unpause physics, wait, then pause again
        self.unpause()
        time.sleep(TIME_DELTA)
        self.pause()
        self.example_home_the_robot()
        # Get initial state
        current_state = self.states.pose
        self.last_action = 0
        self.last_distance = 0

        # Set it to home position

        return current_state
    
    def close(self):
        """
        Clean up ROS1 resources.
        """
        rospy.signal_shutdown("Closing environment")
