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
from std_srvs.srv import *
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
import datetime 
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
    def __init__(self, observation_space: spaces.Dict):
        # Extract image dimensions from observation space
        self.image_dims = observation_space['image'].shape
        self.state_dims = observation_space['end_eff_space'].shape[0]
        
        super().__init__(observation_space, features_dim=128)
        
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
            nn.Linear(512 + 64, 128),  # 512 from CNN + 64 from state_net
            nn.ReLU()
        )

    def forward(self, observations):
        # Process image through CNN
        image = observations['image'].permute(0, 3, 1, 2) / 255.0
        image_features = self.cnn(image)
        
        # Process robot state
        state_features = self.state_net(observations['end_eff_space'])
        
        # Combine features
        combined = torch.cat([image_features, state_features], dim=1)
        return self.combined(combined)

class TensorboardCallback(BaseCallback):
    def __init__(self, log_dir, verbose=0):
        super(TensorboardCallback, self).__init__(verbose)
        self.log_dir = log_dir

    def _on_step(self):
        # Log additional info every step
        self.logger.record('train/episode_reward', self.training_env.get_attr('episode_reward')[0])
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
        self.action_space = spaces.Box(
            low=np.array([-0.05, -0.05, -0.05, -np.pi/4, 0.0]),
            high=np.array([0.05, 0.05, 0.05, np.pi/4, 1.0]),
            dtype=np.float32
        )
        self.bridge = CvBridge()
        
        self.robot_name = "my_gen3"

        # Launch ROS core and Gazebo
        # port = "11311"
        # self.launfile = "spawn_kortex_robot.launch"
        # subprocess.Popen(["roscore","-p",port])
        # print("roscore launched!")
        # subprocess.Popen(["roslaunch","-p", "11311","kortex_gazebo","spawn_kortex_robot.launch"])
        rospy.init_node('ppo_controller') 
        # print("Gazebo Launched")

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
        # rospy.wait_for_service("/gazebo/unpause_physics")
        # self.unpause = rospy.ServiceProxy("/gazebo/unpause_physics",Empty)
        
        # rospy.wait_for_service("/gazebo/pause_physics")
        # self.pause = rospy.ServiceProxy("/gazebo/pause_physics",Empty)
        
        # rospy.wait_for_service("/gazebo/reset_simulation")
        # self.reset_proxy = rospy.ServiceProxy("/gazebo/reset_simulation",Empty)
        
        print("all service and topics called")

        
        # Initialize state variables
        self.states = {'image':[],'end_eff_space':[]}
        self.states_lock = threading.Lock()
        
        # Start ROS callback thread
        self.callback_thread = threading.Thread(target = self.run_callback)
        self.callback_thread.daemon = True
        self.callback_thread.start()

        self.last_action = np.zeros(5)  # Initialize with zeros for first action
        self.last_match_score = 0  # Track previous match score
        
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


    def compute_match_score(self,image, template_path='task_board_template.png'):
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


    def compute_task_reward(self, image):
        """
        Computes the reward based on the match score and provides a reward value.
        The reward is higher when more of the task board is visible (better match).
        """
        match_score, bbox = self.compute_match_score(image)
        
        # Calculate the reward
        reward = match_score  # Match score directly used as reward (scaled 0-1)
        
        # If there is no match, reward is zero
        if match_score < 0.1:
            reward = 0
        
        # Plot results
        plt.figure(figsize=(8, 6))
        plt.imshow(image)  # Show image in RGB format
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
            action (np.array): Action to be executed [∆x, ∆y, ∆z, ∆θ, agripper]
        
        Returns:
            tuple: (observation, reward, done, info)
        """
        # self.unpause()
        time.sleep(0.1)  # Allow time for action execution
        # self.pause()
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
            self.last_action_notif_type = None
            try:
                self.execute_action(req)
            except rospy.ServiceException:
                rospy.logerr(f"Failed to send pose")
                return False
            rospy.loginfo(f"Waiting for pose to finish...")
            

           

            # Get new observation
            observation = {
                'image': self.states['image'],
                'end_eff_space': self.states['end_eff_space']
            }

            # Calculate rewards
            # 1. Task board visibility reward
            match_score, bbox = self.compute_task_reward(self.states['image'])
            visibility_reward = match_score
            
            # 2. Match score improvement reward
            score_improvement = match_score - self.last_match_score
            improvement_reward = score_improvement
            
            # 3. Action smoothness reward
            action_diff = np.linalg.norm(action - self.last_action)
            smoothness_reward = -action_diff  # Negative because we want to minimize sudden changes
            
            # 4. Energy efficiency reward
            energy_penalty = -np.sum(np.square(action))  # Negative because it's a penalty
            
            # Combine rewards with weights
            reward = (
                self.w1 * visibility_reward +
                self.w2 * improvement_reward +
                self.w3 * smoothness_reward +
                self.w4 * energy_penalty
            )

            # Update last values for next step
            self.last_action = action
            self.last_match_score = match_score
            
            # Check if episode should end
            done = False
            if match_score > 0.8:  # High match score - success
                reward += 10.0  # Bonus reward
                done = True
            elif match_score < 0.1:  # Lost sight of board
                reward -= 5.0  # Penalty
                done = True
            
            # Additional info for logging
            info = {
                'visibility_reward': visibility_reward,
                'improvement_reward': improvement_reward,
                'smoothness_reward': smoothness_reward,
                'energy_penalty': energy_penalty,
                'match_score': match_score,
                'bbox': bbox
            }

            self.episode_reward += reward
            return observation, reward, done, info

        except Exception as e:
            rospy.logerr(f"Error in step: {e}")
            return self.reset(), 0, True, {}

    def reset(self, seed=None, options=None):
        """
        Reset the environment to an initial state.
        
        Args:
            seed (int, optional): Seed for random number generator
            options (dict, optional): Additional options for reset
        
        Returns:
            tuple: (observation, info)
        """
        super().reset(seed=seed)  # Initialize RNG if seed is provided
        # self.reset()
        # Reset simulation
        rospy.sleep(1)

        # Reset internal variables
        self.last_action = np.zeros(5)
        self.last_match_score = 0
        self.episode_reward = 0

        # Move robot to home position
        self.example_home_the_robot()
        
        # Get initial observation
        observation = {
            'image': self.states['image'],
            'end_eff_space': self.states['end_eff_space']
        }

        # Return observation and empty info dict
        return observation, {}
    
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
    check_env(env)
    
    # Training hyperparameters for quick test
    total_timesteps = 10_000  # Reduced from 1M to 10k steps
    n_steps = 512  # Reduced from 2048 to 512
    n_epochs = 5   # Reduced from 10 to 5
    batch_size = 32  # Reduced from 64 to 32
    
    # Set up logging
    current_time = datetime.now().strftime('%Y%m%d-%H%M%S')
    log_dir = f"logs/kinova_ppo_{current_time}"
    os.makedirs(log_dir, exist_ok=True)
    
    print("Starting test training run with:")
    print(f"Total timesteps: {total_timesteps}")
    print(f"Steps per update: {n_steps}")
    print(f"Epochs per update: {n_epochs}")
    print(f"Batch size: {batch_size}")
    
    # Initialize PPO with custom policy and hyperparameters
    policy_kwargs = {
        "features_extractor_class": CustomCNN,
        "features_extractor_kwargs": dict(features_dim=128),
        "net_arch": [dict(pi=[64, 32], vf=[64, 32])]  # Smaller network for faster training
    }
    
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
    
    # Create callback for logging
    callback = TensorboardCallback(log_dir)
    
    # Training loop with more frequent evaluation
    eval_interval = 1000  # Evaluate every 1k steps instead of 10k
    best_mean_reward = -np.inf
    
    def evaluate_callback(_locals, _globals):
        nonlocal best_mean_reward
        
        # Evaluate the agent
        mean_reward, std_reward = evaluate_policy(
            model, 
            env,
            n_eval_episodes=3,  # Reduced from 10 to 3 episodes
            deterministic=True
        )
        
        print(f"Evaluation reward: {mean_reward:.2f} ± {std_reward:.2f}")
        
        # Log to tensorboard
        model.logger.record('eval/mean_reward', mean_reward)
        model.logger.record('eval/reward_std', std_reward)
        
        # Save best model
        if mean_reward > best_mean_reward:
            best_mean_reward = mean_reward
            model.save(f"{log_dir}/best_model")
            print(f"New best model saved with reward {mean_reward:.2f}")
            
        return True
    
    try:
        print("Starting training...")
        # Train the agent
        model.learn(
            total_timesteps=total_timesteps,
            callback=[
                callback,
                evaluate_callback
            ],
            tb_log_name="PPO_test"
        )
        
        # Save final model
        model.save(f"{log_dir}/final_model")
        print("Test training completed!")
        
    except KeyboardInterrupt:
        print("Training interrupted! Saving current model...")
        model.save(f"{log_dir}/interrupted_model")
    except Exception as e:
        print(f"Error during training: {e}")
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
        
    except KeyboardInterrupt:
        print("\nTraining interrupted by user!")
        # Clean up ROS and Gazebo
        try:
            # Kill Gazebo node
            subprocess.call(["rosnode", "kill", "/gazebo"])
            print("Gazebo node killed")
            
            # Kill all ROS nodes
            subprocess.call(["rosnode", "kill", "-a"])
            print("All ROS nodes killed")
            
            # Shutdown ROS
            rospy.signal_shutdown("User interrupted training")
            print("ROS shutdown complete")
            
            # Close the environment
            if 'env' in locals():
                env.close()
                print("Environment closed")
                
        except Exception as e:
            print(f"Error during cleanup: {e}")
        finally:
            print("Exiting program")
            sys.exit(0)
            
    except rospy.ROSInterruptException:
        print("ROS interrupted")
    except Exception as e:
        print(f"Error during execution: {e}")
    finally:
        # Always try to clean up
        try:
            subprocess.call(["rosnode", "kill", "/gazebo"])
            subprocess.call(["rosnode", "kill", "-a"])
            rospy.signal_shutdown("Program ended")
            if 'env' in locals():
                env.close()
        except:
            pass

