import rospy
import threading
from collections import deque
import numpy as np
import random
import matplotlib.pyplot as plt
from gymnasium import spaces, Env
from stable_baselines3.common.env_checker import check_env
from sensor_msgs.msg import Image
from std_msgs.msg import Float64
from kortex_driver.msg import BaseCyclic_Feedback
import time
from std_srvs.srv import Empty as Empty_srv
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions.normal import Normal
from numpy import inf
import subprocess
from os import path
import os
from tqdm import tqdm
from torch.distributions import MultivariateNormal, Normal
from collections import namedtuple,deque
import matplotlib
torch.autograd.set_detect_anomaly(True)
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime 
from kortex_driver.srv import *
from kortex_driver.msg import *
import cv2
from cv_bridge import CvBridge
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image as PILImage
from stable_baselines3 import PPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.evaluation import evaluate_policy


class CustomCNN(BaseFeaturesExtractor):
    """
    Custom CNN feature extractor for processing both images and robot state
    """
    def __init__(self, observation_space: spaces.Dict, features_dim: int = 128):
        # Initialize the superclass with the correct parameters
        super().__init__(observation_space, features_dim)
        
        # Extract image dimensions from observation space
        self.image_dims = observation_space['image'].shape  # Should be (84, 84, 3)
        self.state_dims = observation_space['end_eff_space'].shape[0]  # Should be 7
        
        # Calculate CNN output dimension
        # Input: 84x84x3
        # After Conv2d(3, 32, 8, 4): 20x20x32
        # After Conv2d(32, 64, 4, 2): 9x9x64
        # After Conv2d(64, 32, 3, 1): 7x7x32
        # After Flatten: 7 * 7 * 32 = 1568
        
        # CNN layers for image processing
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 32, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten()
        )
        
        # FC layers for robot state processing
        self.state_net = nn.Sequential(
            nn.Linear(self.state_dims, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU()
        )
        
        # Combined processing
        self.combined = nn.Sequential(
            nn.Linear(1568 + 64, features_dim),  # 1568 from CNN + 64 from state_net
            nn.ReLU()
        )

    def forward(self, observations):
        # Process image through CNN
        # The image shape should be [batch_size, height, width, channels]
        # We need to permute to [batch_size, channels, height, width] for PyTorch

        image = observations['image']
        if len(image.shape) == 3:
            image = image.unsqueeze(0)  # Add batch dimension if needed
            
        # Ensure correct shape [batch, channels, height, width]
        if image.shape[-1] == 3:  # If channels are last
            image = image.permute(0, 3, 1, 2)
        
        # Normalize image
        image = image.float() / 255.0
        
        # Debug info
        rospy.logdebug(f"Image shape before CNN: {image.shape}")
        
        # Process through CNN
        image_features = self.cnn(image)
        
        # Process robot state
        state_features = self.state_net(observations['end_eff_space'])
        
        # Combine features
        combined = torch.cat([image_features, state_features], dim=1)
        rospy.logdebug(f"Combined features shape: {combined.shape}")
        
        return self.combined(combined)

class TensorboardCallback(BaseCallback):
    """
    Custom callback for logging to tensorboard.
    """
    def __init__(self, log_dir, verbose=0):
        super(TensorboardCallback, self).__init__(verbose)
        self.log_dir = log_dir
        
    def _init_callback(self) -> None:
        """
        Initialize callback parameters.
        """
        self.writer = SummaryWriter(self.log_dir)
        
    def _on_step(self) -> bool:
        """
        Log values and info on each step.
        """
        # Log episode reward
        if self.locals.get('done'):
            self.logger.record('train/episode_reward', 
                             self.training_env.get_attr('episode_reward')[0])
            self.logger.dump(self.num_timesteps)
        return True
        
    def _on_rollout_end(self) -> None:
        """
        Called when a rollout ends.
        """
        pass
        
    def _on_training_end(self) -> None:
        """
        Clean up when training ends.
        """
        self.writer.close()

class EvaluateCallback(BaseCallback):
    """
    Callback for evaluating the agent periodically during training.
    """
    def __init__(self, eval_env, log_dir, eval_freq=1000, n_eval_episodes=3, verbose=0):
        super(EvaluateCallback, self).__init__(verbose)
        self.eval_env = eval_env
        self.log_dir = log_dir
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes
        self.best_mean_reward = -np.inf
        
    def _init_callback(self):
        # Create folders if needed
        if self.log_dir is not None:
            os.makedirs(self.log_dir, exist_ok=True)
    
    def _on_step(self):
        if self.n_calls % self.eval_freq == 0:
            # Evaluate the agent
            mean_reward, std_reward = evaluate_policy(
                self.model,
                self.eval_env,
                n_eval_episodes=self.n_eval_episodes,
                deterministic=True
            )
            
            rospy.loginfo(f"Evaluation reward: {mean_reward:.2f} ± {std_reward:.2f}")
            
            # Log to tensorboard
            self.logger.record('eval/mean_reward', mean_reward)
            self.logger.record('eval/reward_std', std_reward)
            
            # Save best model
            if mean_reward > self.best_mean_reward:
                self.best_mean_reward = mean_reward
                self.model.save(f"{self.log_dir}/best_model")
                rospy.loginfo(f"New best model saved with reward {mean_reward:.2f}")
        
        return True

class KinovaEnv(Env):
    """
    Custom Gym environment for the Kinova robotic arm.
    This environment interfaces with ROS and Gazebo to control the Kinova arm.
    """

    def __init__(self):
        super(KinovaEnv, self).__init__()
        # Define action and observation space
        self.action_dim = 5
        self.HOME_ACTION_IDENTIFIER = 2
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
        # self.action_space = spaces.Box(
        #     low=np.array([-0.05, -0.05, -0.05, -np.pi/4, 0.0]),
        #     high=np.array([0.05, 0.05, 0.05, np.pi/4, 1.0]),
        #     dtype=np.float32
        # )
        self.action_space = spaces.Box(low=-1, high=1, shape=(5,), dtype=np.float32)
        self.bridge = CvBridge()
        
        self.robot_name = "my_gen3"

        # Launch ROS core and Gazebo
        # port = "11311"
        # self.launfile = "spawn_kortex_robot.launch"
        # subprocess.Popen(["roscore","-p",port])
        # print("roscore launched!")
        # subprocess.Popen(["roslaunch","-p", "11311","kortex_gazebo","spawn_kortex_robot.launch"])
        rospy.init_node('ppo_controller') 
        rospy.loginfo('ppo_controller launded')

        # Initialize ROS subscribers and publishers
        self.states = {'image':[],'end_eff_space':[]}
        self.cartesian_curr_sub = rospy.Subscriber(self.robot_name+'/base_feedback', BaseCyclic_Feedback, self.read_curr_state)
        self.camera_sub = rospy.Subscriber(self.robot_name+'/camera/image_raw', Image, self.read_image)
      
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
        rospy.wait_for_service("/gazebo/unpause_physics")
        self.unpause = rospy.ServiceProxy("/gazebo/unpause_physics",Empty_srv)
        
        rospy.wait_for_service("/gazebo/pause_physics")
        self.pause = rospy.ServiceProxy("/gazebo/pause_physics",Empty_srv)
        
        rospy.wait_for_service("/gazebo/reset_simulation")
        self.reset_proxy = rospy.ServiceProxy("/gazebo/reset_simulation",Empty_srv)
        
        print("all service and topics called")

        
        # Initialize state variables
        self.states = {'image':[],'end_eff_space':[]}
        self.states_lock = threading.Lock()
        
        # Start ROS callback thread
        self.callback_thread = threading.Thread(target = self.run_callback)
        self.callback_thread.daemon = True
        self.callback_thread.start()

        self.episode_reward = 0
        self.last_action = np.zeros(5)
        self.last_match_score = 0
        
        # Reward weights
        self.w1 = 0.4  # Task board visibility reward
        self.w2 = 0.3  # Match score improvement reward
        self.w3 = 0.2  # Action smoothness reward
        self.w4 = 0.1  # Energy efficiency reward

    def read_curr_state(self,msg:BaseCyclic_Feedback):
        commanded_tool_pose = {
                'x': msg.base.commanded_tool_pose_x,
                'y': msg.base.commanded_tool_pose_y,
                'z': msg.base.commanded_tool_pose_z,
                'theta_x': msg.base.commanded_tool_pose_theta_x,
                'theta_y': msg.base.commanded_tool_pose_theta_y,
                'theta_z': msg.base.commanded_tool_pose_theta_z
            }

        self.states['end_eff_space'] = commanded_tool_pose
        rospy.loginfo('Data received')

    def read_image(self,msg:Image):
        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        cv_image_rgb = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)

        # Convert to PIL Image
        pil_image = PILImage.fromarray(cv_image_rgb)
        self.states['image'] = pil_image
        rospy.loginfo('Image received')
        
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
        return time.sleep(0.01) #self.wait_for_action_end_or_abort()

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

    def detect_task_board(self,image):
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


    def compute_match_score(self, image, template_path='src/rl_env/task_board_template.png'):
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
        try:
            # Load the template and convert it to grayscale
            template = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)
            if template is None:
                rospy.logerr(f"Failed to load template from {template_path}")
                return 0, None
            
            # Convert input image to grayscale
            if len(image.shape) == 3:
                gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray_image = image

            # Use ORB detector
            orb = cv2.ORB_create()
            
            # Detect and compute for both images
            kp1, des1 = orb.detectAndCompute(template, None)
            kp2, des2 = orb.detectAndCompute(gray_image, None)
            
            # Check if we have valid descriptors
            if des1 is None or des2 is None or len(des1) == 0 or len(des2) == 0:
                return 0, None

            # Ensure descriptors are in the correct format
            des1 = np.float32(des1)
            des2 = np.float32(des2)

            # Use BFMatcher
            bf = cv2.BFMatcher(cv2.NORM_L2)
            matches = bf.match(des1, des2)

            # Sort matches by distance
            matches = sorted(matches, key=lambda x: x.distance)

            # Calculate match score
            good_matches = [m for m in matches if m.distance < 50]  # Adjust threshold as needed
            match_score = len(good_matches) / len(kp1) if len(kp1) > 0 else 0

            # Only compute bounding box if we have enough good matches
            if len(good_matches) > 10:
                src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

                M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
                if M is not None:
                    h, w = template.shape
                    pts = np.float32([[0, 0], [0, h-1], [w-1, h-1], [w-1, 0]]).reshape(-1, 1, 2)
                    dst = cv2.perspectiveTransform(pts, M)
                    
                    x_min = max(0, int(np.min(dst[:, :, 0])))
                    y_min = max(0, int(np.min(dst[:, :, 1])))
                    x_max = min(gray_image.shape[1], int(np.max(dst[:, :, 0])))
                    y_max = min(gray_image.shape[0], int(np.max(dst[:, :, 1])))
                    
                    bbox = (x_min, y_min, x_max, y_max)
                else:
                    bbox = None
            else:
                bbox = None

            return match_score, bbox

        except Exception as e:
            rospy.logerr(f"Error in compute_match_score: {e}")
            return 0, None


    def compute_task_reward(self, image):
        """
        Computes the reward based on the match score and provides a reward value.
        The reward is higher when more of the task board is visible (better match).
        """
        # Convert PIL Image to numpy array if it's a PIL Image
        if isinstance(image, PILImage.Image):
            image = np.array(image)
        
        # Ensure image is in BGR format for OpenCV
        if len(image.shape) == 3 and image.shape[2] == 3:
            image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        else:
            rospy.logerr("Invalid image format")
            return 0, None
        
        try:
            match_score, bbox = self.compute_match_score(image_bgr)
            
            # Calculate the reward
            reward = match_score  # Match score directly used as reward (scaled 0-1)
            
            # If there is no match, reward is zero
            if match_score < 0.1:
                reward = 0
            
            return reward, bbox
            
        except Exception as e:
            rospy.logerr(f"Error in compute_task_reward: {e}")
            return 0, None

    def _convert_angles_to_radians(self, end_eff_state_dict):
        """Helper method to convert angles from degrees to radians"""
        return np.array([
            end_eff_state_dict['x'],
            end_eff_state_dict['y'],
            end_eff_state_dict['z'],
            np.deg2rad(end_eff_state_dict['theta_x']),  # Convert to radians
            np.deg2rad(end_eff_state_dict['theta_y']),  # Convert to radians
            np.deg2rad(end_eff_state_dict['theta_z']),  # Convert to radians
            0.0  # timestep
        ], dtype=np.float32)

    def _preprocess_image(self, image):
        """Helper method to preprocess image to correct size"""
        if image is None:
            return np.zeros((84, 84, 3), dtype=np.uint8)
        
        # Convert PIL Image to numpy array if needed
        if isinstance(image, PILImage.Image):
            image = np.array(image)
        
        # Ensure image has 3 channels (RGB)
        if len(image.shape) == 2:  # Grayscale
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[2] == 4:  # RGBA
            image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
        
        # Resize image to match observation space
        image_resized = cv2.resize(image, (84, 84), interpolation=cv2.INTER_AREA)
        
        # Ensure the image is in HWC format (height, width, channels)
        return image_resized.astype(np.uint8)

    def step(self, action):
        try:
            # Execute action using ROS
            my_cartesian_speed = CartesianSpeed()
            my_cartesian_speed.translation = 0.1
            my_cartesian_speed.orientation = 15

            my_constrained_pose = ConstrainedPose()
            my_constrained_pose.constraint.oneof_type.speed.append(my_cartesian_speed)

            # Apply action deltas to current position
            current_pose = self.states['end_eff_space']
            my_constrained_pose.target_pose.x = current_pose['x'] + action[0]
            my_constrained_pose.target_pose.y = current_pose['y'] + action[1]
            my_constrained_pose.target_pose.z = current_pose['z'] + action[2]
            my_constrained_pose.target_pose.theta_z = current_pose['theta_z'] + action[3]
            
            # Execute action
            req = ExecuteActionRequest()
            req.input.oneof_action_parameters.reach_pose.append(my_constrained_pose)
            req.input.name = f"pose"
            req.input.handle.action_type = ActionType.REACH_POSE
            req.input.handle.identifier = 1000

            rospy.loginfo(f"Sending pose ...")
            try:
                self.execute_action(req)
                # Wait for action to complete
                rospy.sleep(0.5)  # Fixed sleep to allow action to complete
            except rospy.ServiceException as e:
                rospy.logerr(f"Failed to send pose: {e}")
                return self.reset()[0], 0, True, True, {}

            # Get single observation after action completes
            end_eff_state = self._convert_angles_to_radians(self.states['end_eff_space'])
            image_array = self._preprocess_image(self.states['image'])
            
            observation = {
                'image': image_array,
                'end_eff_space': end_eff_state
            }

            # Calculate rewards
            try:
                match_score, bbox = self.compute_task_reward(image_array)
                visibility_reward = match_score if match_score is not None else 0
                
                # Calculate other rewards
                score_improvement = visibility_reward - self.last_match_score
                action_diff = np.linalg.norm(action - self.last_action)
                energy_penalty = -np.sum(np.square(action))
                
                # Combine rewards
                reward = (
                    self.w1 * visibility_reward +
                    self.w2 * score_improvement +
                    self.w3 * (-action_diff) +
                    self.w4 * energy_penalty
                )

                # Update last values
                self.last_action = action
                self.last_match_score = visibility_reward
                
                # Check episode termination
                done = False
                truncated = False
                if visibility_reward > 0.8:
                    reward += 10.0
                    done = True
                elif visibility_reward < 0.1:
                    reward -= 5.0
                    done = True
                
                info = {
                    'visibility_reward': visibility_reward,
                    'improvement_reward': score_improvement,
                    'smoothness_reward': -action_diff,
                    'energy_penalty': energy_penalty,
                    'match_score': visibility_reward,
                    'bbox': bbox
                }

                self.episode_reward += reward
                return observation, reward, done, truncated, info

            except Exception as e:
                rospy.logerr(f"Error calculating rewards: {e}")
                return self.reset()[0], 0, True, True, {}

        except Exception as e:
            rospy.logerr(f"Error in step: {e}")
            return self.reset()[0], 0, True, True, {}

    def reset(self, seed=None, options=None):
        """Reset the environment"""
        super().reset(seed=seed)
        
        try:
            # Reset simulation
            self.reset_proxy()
            rospy.sleep(1.0)  # Wait for reset to complete
            
            # Reset internal variables
            self.episode_reward = 0
            self.last_action = np.zeros(5)
            self.last_match_score = 0

            # Get single observation
            end_eff_state = self._convert_angles_to_radians(self.states['end_eff_space'])
            image_array = self._preprocess_image(self.states['image'])
            
            observation = {
                'image': image_array,
                'end_eff_space': end_eff_state
            }
            
            return observation, {}
            
        except Exception as e:
            rospy.logerr(f"Error in reset: {e}")
            # Return zero observation on error
            return {
                'image': np.zeros((84, 84, 3), dtype=np.uint8),
                'end_eff_space': np.zeros(7, dtype=np.float32)
            }, {}

    def close(self):
        """
        Clean up ROS1 resources.
        """
        rospy.signal_shutdown("Closing environment")



def log_metrics(filename,msg):
    with open(filename, 'a') as file:
        file.write(msg)

def save_checkpoint(agent, episode, filename):
    checkpoint = {
        'episode': episode,
        'actor_state_dict': agent.actor_network.state_dict(),
        'critic_state_dict': agent.critic_network.state_dict(),
        'actor_optimizer_state_dict': agent.actor_optimizer.state_dict(),
        'critic_optimizer_state_dict': agent.critic_optimizer.state_dict()
    }
    torch.save(checkpoint, filename)

def load_checkpoint(agent, filename):
    if os.path.isfile(filename):
        checkpoint = torch.load(filename)
        agent.actor_network.load_state_dict(checkpoint['actor_state_dict'])
        agent.critic_network.load_state_dict(checkpoint['critic_state_dict'])
        agent.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
        agent.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])
        start_episode = checkpoint['episode']
        best_eval_reward = checkpoint['best_eval_reward']
        print(f"Checkpoint loaded: {filename}")
        return start_episode, best_eval_reward
    else:
        print(f"No checkpoint found at {filename}")
        return 0, float('-inf')

def evaluate(agent, env, num_episodes):
    """
    Evaluate the agent's performance.
    
    Args:
        agent (Agent): The agent to evaluate
        env (gym.Env): The environment
        num_episodes (int): Number of episodes to evaluate
    
    Returns:
        tuple: (mean_reward, std_reward, mean_episode_length)
    """
    total_rewards = []
    episode_lengths = []
    max_timesteps = 1800
    for _ in range(num_episodes):
        observation = env.reset()
        episode_reward = 0
        done = False
        step = 0
        while not done and step < max_timesteps:
            action, _ = agent.select_action(observation)
            observation, reward, done, _ = env.step(action)
            episode_reward += reward
            step += 1
        total_rewards.append(episode_reward)
        episode_lengths.append(step)
    return np.mean(total_rewards), np.std(total_rewards), np.mean(episode_lengths)

def train():
    # Create environment
    env = KinovaEnv()
    rospy.loginfo(check_env(env))
    
    # Training hyperparameters for quick test
    total_timesteps = 10_000  # Reduced from 1M to 10k steps
    n_steps = 512  # Reduced from 2048 to 512
    n_epochs = 5   # Reduced from 10 to 5
    batch_size = 32  # Reduced from 64 to 32
    
    # Set up logging with correct datetime usage
    current_time = datetime.now().strftime('%Y%m%d-%H%M%S')
    log_dir = os.path.join("logs", f"kinova_ppo_{current_time}")
    os.makedirs(log_dir, exist_ok=True)
    
    rospy.loginfo("Starting test training run with:")
    rospy.loginfo(f"Total timesteps: {total_timesteps}")
    rospy.loginfo(f"Steps per update: {n_steps}")
    rospy.loginfo(f"Epochs per update: {n_epochs}")
    rospy.loginfo(f"Batch size: {batch_size}")
    
    # Initialize PPO with custom policy and hyperparameters
    policy_kwargs = dict(
        features_extractor_class=CustomCNN,
        net_arch=dict(pi=[64, 32], vf=[64, 32])  # Changed from list to dict
    )
    
    model = PPO(
        "MultiInputPolicy",
        env,
        n_steps=n_steps,
        batch_size=batch_size,
        n_epochs=n_epochs,
        learning_rate=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        policy_kwargs=policy_kwargs,
        verbose=1,
        tensorboard_log=log_dir
    )
    
    # Create callbacks
    tensorboard_callback = TensorboardCallback(log_dir)
    eval_callback = EvaluateCallback(
        eval_env=env,
        log_dir=log_dir,
        eval_freq=1000,
        n_eval_episodes=3
    )
    
    # Training loop with more frequent evaluation
    eval_interval = 1000  # Evaluate every 1k steps instead of 10k
    best_mean_reward = -np.inf
    
    try:
        rospy.loginfo("Starting training...")
        # Train the agent with both callbacks
        model.learn(
            total_timesteps=total_timesteps,
            callback=[tensorboard_callback, eval_callback],
            tb_log_name="PPO_test"
        )
        
        # Save final model
        model.save(f"{log_dir}/final_model")
        rospy.loginfo("Test training completed!")
        
    except KeyboardInterrupt:
        rospy.loginfo("Training interrupted! Saving current model...")
        model.save(f"{log_dir}/interrupted_model")
    except Exception as e:
        rospy.logerr(f"Error during training: {e}")
        raise e
    
    return model, env

def test(model_path, env, num_episodes=10):
    """Test the trained model"""
    # Load the trained model
    model = PPO.load(model_path)
    
    episode_rewards = []
    for episode in range(num_episodes):
        obs = env.reset()
        done = False
        total_reward = 0
        step = 0
        
        while not done:
            # Get action from model
            action, _ = model.predict(obs, deterministic=True)
            
            # Execute action
            obs, reward, done, _ = env.step(action)
            total_reward += reward
            step += 1
            
        episode_rewards.append(total_reward)
        print(f"Episode {episode + 1}: Steps = {step}, Reward = {total_reward:.2f}")
    
    # Print summary statistics
    mean_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)
    print(f"\nTest Results over {num_episodes} episodes:")
    print(f"Mean Reward: {mean_reward:.2f} ± {std_reward:.2f}")
    
    return mean_reward, std_reward

if __name__ == "__main__":
    try:
        # Train the model
        model, env = train()
        
        # Test the trained model
        print("\nTesting final model...")
        test(f"{model.logger.dir}/final_model.zip", env)
        
        print("\nTesting best model...")
        test(f"{model.logger.dir}/best_model.zip", env)
        
    except Exception as e:
        rospy.logerr(f"Error during execution: {e}")
    finally:
        try:
            # Proper cleanup sequence
            rospy.loginfo("Starting cleanup...")
            
            # First close the environment
            if 'env' in locals():
                env.close()
                rospy.loginfo("Environment closed")
            
            # Kill specific nodes in order
            subprocess.call(["rosnode", "kill", "/gazebo"])
            rospy.sleep(1)  # Give time for Gazebo to shutdown
            
            # Kill remaining nodes
            subprocess.call(["rosnode", "kill", "-a"])
            rospy.sleep(1)  # Give time for nodes to shutdown
            
            # Final ROS shutdown
            rospy.signal_shutdown("Program ended")
            rospy.sleep(1)  # Give time for ROS to shutdown
            
            rospy.loginfo("Cleanup completed")
            
        except Exception as e:
            rospy.logerr(f"Error during cleanup: {e}")
        finally:
            # Force exit to avoid ROS time issues
            os._exit(0)  # Use os._exit instead of sys.exit
