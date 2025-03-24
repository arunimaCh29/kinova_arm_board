import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from simulation_gym_env import KinovaEnv
from network import ACGDNetwork
from torch.distributions import MultivariateNormal


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class PPOAgent:
    def __init__(self, state_dim, action_dim, lr=3e-4, gamma=0.99, epsilon=0.2, lmbda=0.95, epochs=10, batch_size=32):
        self.actor = ACGDNetwork().to(device)
        self.critic = ACGDNetwork().to(device)
        self.critic.fc2 = nn.Linear(128, 1).to(device)  # Modify output layer to return a scalar
        
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr)
        
        self.gamma = gamma
        self.epsilon = epsilon
        self.lmbda = lmbda
        self.epochs = epochs
        self.batch_size = batch_size
        self.mse_loss = nn.MSELoss()

    def select_action(self, image, state):
        image = torch.tensor(image, dtype=torch.float32).unsqueeze(0).to(device)
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
        
        mean = self.actor(image, state)
        log_std = torch.zeros_like(mean, device=device)
        std = torch.exp(log_std)
        dist = MultivariateNormal(mean, torch.diag_embed(std))
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action.cpu().detach().numpy()[0], log_prob.cpu().detach()
    
    def compute_advantages(self, rewards, values, next_values, dones):
        advantages = []
        gae = 0
        for step in reversed(range(len(rewards))):
            delta = rewards[step] + self.gamma * next_values[step] * (1 - dones[step]) - values[step]
            gae = delta + self.gamma * self.lmbda * (1 - dones[step]) * gae
            advantages.insert(0, gae)
        return np.array(advantages)
    
    def learn(self, trajectories):
        states, images, actions, log_probs, rewards, next_states, dones = zip(*trajectories)
        images = torch.tensor(np.array(images), dtype=torch.float32).to(device)
        states = torch.tensor(np.array(states), dtype=torch.float32).to(device)
        actions = torch.tensor(np.array(actions), dtype=torch.float32).to(device)
        old_log_probs = torch.tensor(np.array(log_probs), dtype=torch.float32).to(device)
        rewards = torch.tensor(np.array(rewards), dtype=torch.float32).to(device)
        dones = torch.tensor(np.array(dones), dtype=torch.float32).to(device)

        values = self.critic(images, states).squeeze()
        next_values = torch.cat((values[1:], torch.tensor([0.0], device=device)))
        advantages = self.compute_advantages(rewards.cpu().numpy(), values.cpu().numpy(), next_values.cpu().numpy(), dones.cpu().numpy())
        advantages = torch.tensor(advantages, dtype=torch.float32).to(device)
        returns = advantages + values
        
        for _ in range(self.epochs):
            for i in range(0, len(states), self.batch_size):
                batch_idx = slice(i, i + self.batch_size)
                batch_images = images[batch_idx]
                batch_states = states[batch_idx]
                batch_actions = actions[batch_idx]
                batch_log_probs = old_log_probs[batch_idx]
                batch_returns = returns[batch_idx].squeeze()
                batch_advantages = advantages[batch_idx]
                
                mean = self.actor(batch_images, batch_states)
                log_std = torch.zeros_like(mean, device=device)
                std = torch.exp(log_std)
                dist = MultivariateNormal(mean, torch.diag_embed(std))
                new_log_probs = dist.log_prob(batch_actions)
                
                ratios = torch.exp(new_log_probs - batch_log_probs)
                surr1 = ratios * batch_advantages
                surr2 = torch.clamp(ratios, 1.0 - self.epsilon, 1.0 + self.epsilon) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss = self.mse_loss(self.critic(batch_images, batch_states).squeeze(), batch_returns)
                
                self.actor_optimizer.zero_grad()
                policy_loss.backward()
                self.actor_optimizer.step()
                
                self.critic_optimizer.zero_grad()
                value_loss.backward()
                self.critic_optimizer.step()
    
if __name__ == "__main__":
    env = KinovaEnv()
    agent = PPOAgent(state_dim=8, action_dim=4)
    
    for episode in range(500):
        obs = env.reset()
        done = False
        trajectories = []
        while not done:
            image, state = obs['image'], obs['end_eff_space']
            action, log_prob = agent.select_action(image, state)
            next_obs, reward, done, _ = env.step(action)
            next_image, next_state = next_obs['image'], next_obs['end_eff_space']
            trajectories.append([state, image, action, log_prob, reward, next_state, done])
            obs = next_obs
        agent.learn(trajectories)
