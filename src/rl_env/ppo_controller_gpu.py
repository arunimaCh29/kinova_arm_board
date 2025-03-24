#!/home/dev/anaconda3/envs/myrosenv/bin/python

"""
This script implements a Proximal Policy Optimization (PPO) algorithm for training a Jaco2 robotic arm
in a ROS/Gazebo environment. It uses PyTorch for the neural network implementation and ROS for
robot control and simulation.
"""

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

from network import ACGDNetwork


class ActorNetwork(nn.Module):
    """
    Actor network for the PPO algorithm.
    """
    def __init__(self, n_actions, state_dim, fc1_dims=256, fc2_dims=128, fc3_dims = 128,fc4_dims = 64):
        super(ActorNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, fc1_dims).to(device)
        self.fc2 = nn.Linear(fc1_dims, fc2_dims).to(device)
        self.fc3 = nn.Linear(fc2_dims, fc3_dims).to(device)
        self.fc4 = nn.Linear(fc3_dims, fc4_dims).to(device)
        self.mean = nn.Linear(fc3_dims, n_actions).to(device)
        self.log_std = nn.Parameter(torch.zeros(n_actions).to(device))
        log_metrics(filename, f"Actor Network -- Dimension of each Layer -- Layer 1 :{fc1_dims}, Layer 2 : {fc2_dims}, Layer 3 : {fc3_dims}, Layer 4 : {fc4_dims}, Output Layer : {n_actions}\n")

    def forward(self, x):
        """
        Forward pass through the network.
        
        Args:
            x (torch.Tensor): Input state
        
        Returns:
            tuple: (mean, std) of the action distribution
        """
        x = x.to(device)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        mean = self.mean(x)
        std = self.log_std.exp()
        return mean, std

class CriticNetwork(nn.Module):
    """
    Critic network for the PPO algorithm.
    """
    def __init__(self, state_dim, fc1_dims=256, fc2_dims=128):
        super(CriticNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, fc1_dims).to(device)
        self.fc2 = nn.Linear(fc1_dims, fc2_dims).to(device)
        self.value = nn.Linear(fc2_dims, 1).to(device)
        log_metrics(filename, f"Critick Network -- Dimension of each layer -- Layer 1 : {fc1_dims}, Layer 2 : {fc2_dims}, Output Layer : {1}\n")

    def forward(self, x):
        """
        Forward pass through the network.
        
        Args:
            x (torch.Tensor): Input state
        
        Returns:
            torch.Tensor: Estimated state value
        """
        x = x.to(device)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        value = self.value(x)
        return value

class Agent:
    """
    PPO Agent implementation.
    """
    def __init__ (self, state_dim, action_dim, lr = 3e-4, gamma = 0.99,epsilon = 0.2,lmbda = 0.95, epoch = 10, batch_size = 32):
        self.actor_network = ActorNetwork(action_dim,state_dim).to(device)
        self.critic_network = CriticNetwork(state_dim).to(device)
        self.actor_optimizer = optim.Adam(self.actor_network.parameters(),lr = lr)
        self.critic_optimizer = optim.Adam(self.critic_network.parameters(),lr = lr)
        self.gamma = gamma
        self.epsilon = epsilon
        self.lmbda = lmbda
        self.epochs = epoch
        self.batch_size = batch_size
        self.MseLoss = nn.MSELoss()
        self.actor_loss = 0
        self.critic_loss = 0
        self.total_loss = 0
        self.actor_scheduler = optim.lr_scheduler.StepLR(self.actor_optimizer, step_size=100, gamma=0.9)
        self.critic_scheduler = optim.lr_scheduler.StepLR(self.critic_optimizer, step_size=100, gamma=0.9)
        log_metrics(filename, f"Gamma : {gamma}, Epsilon : {epsilon}, Learning Rate : {lr}, Lambda : {lmbda}, Epoch : {epoch}, Batch size : {batch_size}\n")

    def save_models(self, path):
        """Save the actor and critic models."""
        if not os.path.exists(path):
            os.makedirs(path)
        torch.save(self.actor_network.state_dict(), os.path.join(path, f'actor_{current_time}.pth'))
        torch.save(self.critic_network.state_dict(), os.path.join(path, f'critic_{current_time}.pth'))
        print(f"Models saved to {path}")

    def load_models(self, path):
        """Load the actor and critic models."""
        self.actor_network.load_state_dict(torch.load(os.path.join(path, f'actor_{current_time}.pth'), map_location=device))
        self.critic_network.load_state_dict(torch.load(os.path.join(path, f'critic_{current_time}.pth'), map_location=device))
        print(f"Models loaded from {path}")

    def select_action(self,state):
        """
        Select an action based on the current state.
        
        Args:
            state (np.array): Current state
        
        Returns:
            tuple: (action, action_log_prob)
        """
        with torch.no_grad():
            state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
            mean, std = self.actor_network(state)
        if torch.isnan(mean).any() or torch.isnan(std).any():
            print("NaN values detected in action selection")
            return np.zeros(7), torch.tensor(0.0)
        mean = torch.nan_to_num(mean, nan=0.0)
        std = torch.nan_to_num(std, nan=1.0)
        cov_matrix = torch.diag(std**2) 
        dist = MultivariateNormal(mean,covariance_matrix=cov_matrix)
        action = dist.sample()
        action_log_prob = dist.log_prob(action).sum(dim=-1)
        action = 2 * torch.tanh(action)
        return action.cpu().detach().numpy()[0], action_log_prob.cpu().detach()
    
    def compute_advantages(self, rewards, values, next_values, dones):
        """
        Compute advantage estimates.
        
        Args:
            rewards (np.array): Array of rewards
            values (np.array): Array of state values
            next_values (np.array): Array of next state values
            dones (np.array): Array of done flags
        
        Returns:
            np.array: Computed advantages
        """
        advantages = []
        gae = 0
        rewards = np.atleast_1d(rewards)
        values = np.atleast_1d(values)
        next_values = np.atleast_1d(next_values)
        dones = np.atleast_1d(dones)
        for step in reversed(range(len(rewards))):
            if step == len(rewards) - 1:
                next_value = 0
            else:
                next_value = next_values[step] 
            delta = rewards[step] + self.gamma * next_value * (1 - dones[step]) - values[step]
            gae = delta + self.gamma * self.lmbda * (1 - dones[step]) * gae
            advantages.insert(0, gae)  
        return np.array(advantages)

    def learn(self, trajectories):
        """
        Update policy and value function using collected trajectories.
        
        Args:
            trajectories (list): List of trajectory data
        """
        if not trajectories:
            print("Warning: Empty trajectories. Skipping learning step.")
            return
        
        # Unpack the trajectories
        states, actions, log_probs, rewards, next_states, dones = zip(*trajectories)
        
        # Convert numpy arrays to PyTorch tensors and move to the correct device
        states = torch.tensor(np.array(states), dtype=torch.float32).to(device)
        actions = torch.tensor(np.array(actions), dtype=torch.float32).to(device)
        old_log_probs = torch.stack(log_probs).to(device)
        rewards = torch.tensor(np.array(rewards), dtype=torch.float32).to(device)
        next_states = torch.tensor(np.array(next_states), dtype=torch.float32).to(device)
        dones = torch.tensor(np.array(dones), dtype=torch.bool).to(device)

        #remove if not requireds 
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-8)

        with torch.no_grad():
            # Compute state values
            values = self.critic_network(states).squeeze().to(device)
            next_values = self.critic_network(next_states).squeeze().to(device)
            
            # Compute advantages
            advantages = self.compute_advantages(
                rewards.cpu().numpy(),
                values.cpu().numpy(),
                next_values.cpu().numpy(),
                dones.cpu().numpy()
            )
            
            # Normalize advantages
            advantages = torch.tensor(advantages, dtype=torch.float32).to(device)
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            
            # Compute returns
            returns = advantages + values
        
        # Perform multiple epochs of training
        for _ in range(self.epochs):
            # Iterate over mini-batches
            for i in range(0, len(states), self.batch_size):
                batch_indices = slice(i, i + self.batch_size)
                batch_states = states[batch_indices]
                batch_actions = actions[batch_indices]
                batch_log_probs = old_log_probs[batch_indices]
                batch_returns = returns[batch_indices].squeeze()
                batch_advantages = advantages[batch_indices]
                
                # Compute new action probabilities
                mean, std = self.actor_network(batch_states)
                dist = Normal(mean, std)
                new_log_probs = dist.log_prob(batch_actions).sum(dim=-1)
                
                # Compute probability ratio
                log_ratio = new_log_probs - batch_log_probs
                #log_ratio = torch.clamp(log_ratio, -20, 20)  # Prevent numerical instability #------------------------------------------------------------------
                ratios = torch.exp(log_ratio)
                
                # Compute surrogate objectives
                surr1 = ratios * batch_advantages
                surr2 = torch.clamp(ratios, 1.0 - self.epsilon, 1.0 + self.epsilon) * batch_advantages

                entropy = dist.entropy().mean()
                
                # Compute actor (policy) loss
                policy_loss =  - (torch.min(surr1, surr2).mean() + 0.01* entropy)

                # Compute critic (value) loss
                value_pred = self.critic_network(batch_states).squeeze()
                value_loss = F.mse_loss(value_pred, batch_returns,reduction = 'mean')

                # Compute actor (policy) loss
                # Note: Assuming policy_loss is already computed

                total_loss = policy_loss + 0.5* value_loss

                # # Perform backpropagation for critic
                # self.critic_optimizer.zero_grad()
                # value_loss.backward()
                # # Clip gradients for critic
                # torch.nn.utils.clip_grad_norm_(self.critic_network.parameters(), max_norm=0.7)
                # # Update critic network parameters
                # self.critic_optimizer.step()
                # self.critic_scheduler.step()

                # # Perform backpropagation for actor
                # self.actor_optimizer.zero_grad()
                # policy_loss.backward()
                # # Clip gradients for actor
                # torch.nn.utils.clip_grad_norm_(self.actor_network.parameters(), max_norm=0.7)
                # # Update actor network parameters
                # self.actor_optimizer.step()
                # self.actor_scheduler.step()

                self.critic_optimizer.zero_grad()
                self.actor_optimizer.zero_grad()

                total_loss.backward()

                torch.nn.utils.clip_grad_norm_(self.critic_network.parameters(), max_norm=0.5)
                torch.nn.utils.clip_grad_norm_(self.actor_network.parameters(), max_norm=0.5)

                self.critic_optimizer.step()
                self.actor_optimizer.step()
                self.critic_scheduler.step()               
                self.actor_scheduler.step()

        # Store the final loss values for logging
        self.actor_loss = policy_loss.item()
        self.critic_loss = value_loss.item()
        self.total_loss = total_loss.item()

def log_metrics(filename,msg):
    with open(filename, 'a') as file:
        file.write(msg)

def save_checkpoint(agent, episode, filename):
    checkpoint = {
        'episode': episode,
        'actor_state_dict': agent.actor_network.state_dict(),
        'critic_state_dict': agent.critic_network.state_dict(),
        'actor_optimizer_state_dict': agent.actor_optimizer.state_dict(),
        'critic_optimizer_state_dict': agent.critic_optimizer.state_dict(),
        'best_eval_reward': best_eval_reward
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

if __name__ == '__main__':
    # Set random seeds for reproducibility
    random.seed(1)
    np.random.seed(1)
    torch.manual_seed(1)

    # Set up the device (CPU or GPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Training parameters
    # actor_lr = 0.001
    # critic_lr = 0.001
    batch_size = 32
    TIME_DELTA = 0
    n_episodes = 500
    max_timesteps = 1500
    eval_interval = 50
    eval_epsiodes = 15

    # Initialize lists and variables for tracking progress
    episode_rewards = []
    best_eval_reward = float('-inf')

    

    # Set up TensorBoard logging and saving parameters in doc files
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    filename = f"runs/jaco2_ppo_{current_time}/{current_time}_logfile.txt"
    log_dir = f'runs/jaco2_ppo_{current_time}'
    writer = SummaryWriter(log_dir)
    log_metrics(filename,f"Episodes : {n_episodes}, Max Timesteps : {max_timesteps}, Evaluation Interval: {eval_interval}. Evaluation Epsiodes : {eval_epsiodes}\n")

    checkpoint_dir = f'runs/jaco2_ppo_{current_time}/checkpoints'
    os.makedirs(checkpoint_dir, exist_ok=True)


    # Initialize the environment
    env = Jaco2Env()
    print("observation space dimension :", env.observation_space.shape[0])
    print("action space dimension :", env.action_space.shape[0])

    # Initialize the agent
    agent = Agent(state_dim=env.observation_space.shape[0], action_dim=env.action_space.shape[0])

    # Wait for ROS and Gazebo to initialize
    time.sleep(10)
    print("Training begins")



    try:
        for i in range(n_episodes):
            # Reset the environment and get initial observation
            observation = env.reset()
            done = False
            score = 0
            trajectories = []

            # Run the episode for a maximum number of timesteps
            for t in range(max_timesteps):
                # Select an action based on the current observation
                action, log_prob = agent.select_action(observation)

                # Take a step in the environment
                next_observation, reward, done, info = env.step(action)

                # Accumulate the reward
                score += reward

                # Store the transition data
                trajectories.append([observation, action, log_prob, reward, next_observation, done])

                # Update the observation
                observation = next_observation

                # Check if the episode is done
                if done:
                    print("Episodic task completed early")
                    break

            # Print and store the total reward for this episode
            print(f"Total reward for episode {i} is {score}")
            episode_rewards.append(score)

            # Log training metrics
            writer.add_scalar('Training/Episode Reward', score, i)
            writer.add_scalar('Training/Episode Length', t+1, i)

            # Update the agent
            try :
                agent.learn(trajectories)
            except RuntimeError as e :
                print(f"Error during learning : {e}")
                continue

            # Log loss metrics
            writer.add_scalar('Training/Actor Loss', agent.actor_loss, i)
            writer.add_scalar('Training/Critic Loss', agent.critic_loss, i)
            writer.add_scalar('Training/Total Loss', agent.total_loss,i)

            t_msgs = f"Epsiode : {i}, Reward : {score:.2f}, epsiode Length : {t+1}, Actor Loss : {agent.actor_loss:.2f}, Critic Loss : {agent.critic_loss:.2f}\n"
            log_metrics(filename,t_msgs)
            # Periodic evaluation

            if (i + 1) % 10 == 0:  # Save every 10 episodes, adjust as needed
                timeee = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
                checkpoint_file = os.path.join(checkpoint_dir, f'latest_checkpoint_{timeee}.pth')
                save_checkpoint(agent, i + 1, checkpoint_file)

            if (i + 1) % eval_interval == 0:
                mean_reward, std_reward, mean_length = evaluate(agent, env, eval_epsiodes)
                writer.add_scalar('Evaluation/Mean Reward', mean_reward, i)
                writer.add_scalar('Evaluation/Std Reward', std_reward, i)
                writer.add_scalar('Evaluation/Mean Episode Length', mean_length, i)              
                print(f"Evaluation after {i+1} episodes: Mean reward: {mean_reward:.2f} ± {std_reward:.2f}, Mean length: {mean_length:.2f}")   
                e_msgs = f"Evaluation after {i+1} episodes: Mean reward: {mean_reward:.2f} ± {std_reward:.2f}, Mean length: {mean_length:.2f}\n"
                log_metrics(filename,e_msgs)         
                
                # Save the best model
                if mean_reward > best_eval_reward:
                    best_eval_reward = mean_reward
                    agent.save_models(f'runs/jaco2_ppo_{current_time}/best_model_episode_{i+1}_{current_time}')
                    print(f"New best model saved at episode {i+1}_{current_time}")

        print(episode_rewards)
        agent.save_models(f'runs/jaco2_ppo_{current_time}/models')
        print("loading model")
        agent.load_models(f'runs/jaco2_ppo_{current_time}/mpdels')
        print("model loaded")

        # Final evaluation
        final_mean_reward, final_std_reward, final_mean_length = evaluate(agent, env, num_episodes=50)
        print(f"Final evaluation: Mean reward: {final_mean_reward:.2f} ± {final_std_reward:.2f}, Mean length: {final_mean_length:.2f}")
        log_metrics(filename,f"Final evaluation: Mean reward: {final_mean_reward:.2f} ± {final_std_reward:.2f}, Mean length: {final_mean_length:.2f}\n")

        # Close TensorBoard writer
        writer.close()

    except rospy.ROSInterruptException:
        timeee = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        checkpoint_file = os.path.join(checkpoint_dir, f'latest_checkpoint_{timeee}.pth')
        save_checkpoint(agent, i + 1,checkpoint_file)
        pass
    finally:
        rospy.signal_shutdown("Training complete")