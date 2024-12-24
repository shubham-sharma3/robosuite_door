import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
import numpy as np
from networks_ppo import ActorNetwork, CriticNetwork
from buffer_ppo import PPORolloutBuffer

class PPOAgent:
    def __init__(self,
                 input_dims,
                 n_actions,
                 lr_actor=3e-4,
                 lr_critic=1e-3,
                 gamma=0.99,
                 K_epochs=80,
                 eps_clip=0.2,
                 buffer_size=2048,
                 batch_size=64,
                 entropy_coeff=0.01,
                 device=None):
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        self.batch_size = batch_size
        self.entropy_coeff = entropy_coeff

        self.device = device if device else torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        # Initialize actor and critic networks
        self.actor = ActorNetwork(input_dims, n_actions=n_actions, checkpoint_dir='tmp/ppo').to(self.device)
        self.critic = CriticNetwork(input_dims, n_actions=n_actions, checkpoint_dir='tmp/ppo').to(self.device)

        # Initialize old actor for calculating log probabilities
        self.policy_old = ActorNetwork(input_dims, n_actions=n_actions, checkpoint_dir='tmp/ppo').to(self.device)
        self.policy_old.load_state_dict(self.actor.state_dict())

        # Initialize optimizers
        self.optimizer_actor = optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.optimizer_critic = optim.Adam(self.critic.parameters(), lr=lr_critic)

        # Initialize rollout buffer
        self.buffer = PPORolloutBuffer(buffer_size, input_dims, n_actions)

        # Loss function
        self.MseLoss = nn.MSELoss()

    def choose_action(self, state):
        state = torch.FloatTensor(state).to(self.device)
        with torch.no_grad():
            action_mean = self.policy_old(state)
        # For exploration, sample from the policy's distribution
        std = torch.ones_like(action_mean) * 0.5  # Fixed standard deviation; can be parameterized
        dist = Normal(action_mean, std)
        action = dist.sample()
        action_clipped = torch.tanh(action)
        log_prob = dist.log_prob(action).sum(dim=-1)
        # Adjust log_prob for tanh transformation
        log_prob -= (2*(torch.log(2) - action - nn.functional.softplus(-2*action))).sum(dim=-1)
        return action_clipped.cpu().numpy(), log_prob.cpu().numpy()

    def remember(self, state, action, log_prob, reward, done):
        self.buffer.store_transition(state, action, log_prob, reward, done)

    def compute_returns_and_advantages(self, rewards, dones, values, next_values, gamma=0.99, gae_lambda=0.95):
        returns = []
        advantages = []
        gae = 0
        for step in reversed(range(len(rewards))):
            delta = rewards[step] + gamma * next_values[step] * (1 - dones[step]) - values[step]
            gae = delta + gamma * gae_lambda * (1 - dones[step]) * gae
            advantages.insert(0, gae)
            returns.insert(0, gae + values[step])
        advantages = np.array(advantages)
        returns = np.array(returns)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        return returns, advantages

    def update(self):
        # Retrieve data from buffer
        states, actions, log_probs_old, rewards, dones = self.buffer.get()
        self.buffer.clear()

        # Convert to tensors
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.FloatTensor(actions).to(self.device)
        log_probs_old = torch.FloatTensor(log_probs_old).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)

        # Compute state values
        with torch.no_grad():
            values = self.critic(states, actions).squeeze().cpu().numpy()
            # Estimate next state values (assuming next state is the last state)
            # Since it's on-policy, we don't have next actions from the current policy
            next_state = states[-1].unsqueeze(0)
            next_action = actions[-1].unsqueeze(0)
            next_value = self.critic(next_state, next_action).squeeze().cpu().numpy()
            next_values = np.append(values[1:], next_value)

        # Compute returns and advantages
        returns, advantages = self.compute_returns_and_advantages(rewards.cpu().numpy(), dones.cpu().numpy(), values, next_values, self.gamma, 0.95)

        # Convert to tensors
        returns = torch.FloatTensor(returns).to(self.device)
        advantages = torch.FloatTensor(advantages).to(self.device)

        # Optimize policy for K epochs
        for _ in range(self.K_epochs):
            # Re-compute log probabilities and state values
            action_mean = self.actor(states)
            std = torch.ones_like(action_mean) * 0.5
            dist = Normal(action_mean, std)
            log_probs = dist.log_prob(actions).sum(dim=-1)
            entropy = dist.entropy().sum(dim=-1)

            # Compute ratios for clipping
            ratios = torch.exp(log_probs - log_probs_old)

            # Compute surrogate losses
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            actor_loss = -torch.min(surr1, surr2).mean() - self.entropy_coeff * entropy.mean()

            # Compute critic loss
            state_values = self.critic(states, actions).squeeze()
            critic_loss = self.MseLoss(state_values, returns)

            # Take gradient step for actor
            self.optimizer_actor.zero_grad()
            actor_loss.backward()
            self.optimizer_actor.step()

            # Take gradient step for critic
            self.optimizer_critic.zero_grad()
            critic_loss.backward()
            self.optimizer_critic.step()

        # Update old policy
        self.policy_old.load_state_dict(self.actor.state_dict())

    def save_models(self):
        torch.save(self.actor.state_dict(), 'tmp/ppo/actor.pth')
        torch.save(self.critic.state_dict(), 'tmp/ppo/critic.pth')
        torch.save(self.policy_old.state_dict(), 'tmp/ppo/policy_old.pth')
        print("PPO models saved successfully.")

    def load_models(self):
        try:
            self.actor.load_state_dict(torch.load('tmp/ppo/actor.pth'))
            self.critic.load_state_dict(torch.load('tmp/ppo/critic.pth'))
            self.policy_old.load_state_dict(torch.load('tmp/ppo/policy_old.pth'))
            print("PPO models loaded successfully.")
        except:
            print("Failed to load PPO models. Starting from scratch.")