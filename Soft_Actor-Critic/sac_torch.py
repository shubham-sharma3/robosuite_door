import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
import numpy as np
from sac_networks import ActorNetwork, CriticNetwork
from sac_buffer import ReplayBuffer
import torch.nn.functional as F


class SACAgent:
    def __init__(self,
                 input_dims,
                 n_actions,
                 alpha=0.2,
                 gamma=0.99,
                 tau=0.005,
                 lr_actor=3e-4,
                 lr_critic=3e-4,
                 max_size=1000000,
                 batch_size=256,
                 device=torch.device('cpu')):
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.device = device

        self.alpha = alpha  # Entropy coefficient

        # Initialize Actor network
        self.actor = ActorNetwork(input_dims, n_actions=n_actions).to(self.device)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr_actor)

        # Initialize Critic networks
        self.critic = CriticNetwork(input_dims, n_actions=n_actions).to(self.device)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr_critic)

        # Initialize target Critic networks
        self.critic_target = CriticNetwork(input_dims, n_actions=n_actions).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        # Initialize Replay Buffer
        self.memory = ReplayBuffer(max_size, input_dims, n_actions, device=self.device)

    def choose_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        mean, std = self.actor(state)
        dist = Normal(mean, std)
        x_t = dist.rsample()  # For reparameterization trick (mean + std * N(0,1))
        y_t = torch.tanh(x_t)
        action = y_t.cpu().detach().numpy()[0]

        # Compute log_prob
        log_prob = dist.log_prob(x_t)
        # Enforcing Action Bound
        log_prob -= torch.log(1 - y_t.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=-1)
        return action, log_prob.cpu().detach().numpy()

    def store_transition(self, state, action, reward, state_, done):
        self.memory.store_transition(state, action, reward, state_, done)

    def sample_memory(self):
        states, actions, rewards, states_, dones = self.memory.sample_buffer(self.batch_size)
        return states, actions, rewards, states_, dones

    def update(self):
        if len(self.memory) < self.batch_size:
            return

        states, actions, rewards, states_, dones = self.sample_memory()

        # ---------------------------- update critic ---------------------------- #
        with torch.no_grad():
            # Compute target Q-values
            mean, std = self.actor(states_)
            dist = Normal(mean, std)
            x_t = dist.rsample()
            y_t = torch.tanh(x_t)
            next_actions = y_t
            log_prob = dist.log_prob(x_t)
            log_prob -= torch.log(1 - y_t.pow(2) + 1e-6)
            log_prob = log_prob.sum(dim=-1, keepdim=True)

            target_Q1, target_Q2 = self.critic_target(states_, next_actions)
            target_Q = torch.min(target_Q1, target_Q2) - self.alpha * log_prob
            target_Q = rewards + (1 - dones) * self.gamma * target_Q

        current_Q1, current_Q2 = self.critic(states, actions)
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # ---------------------------- update actor ---------------------------- #
        mean, std = self.actor(states)
        dist = Normal(mean, std)
        x_t = dist.rsample()
        y_t = torch.tanh(x_t)
        actions_new = y_t
        log_prob = dist.log_prob(x_t)
        log_prob -= torch.log(1 - y_t.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)

        Q1_new, Q2_new = self.critic(states, actions_new)
        Q_new = torch.min(Q1_new, Q2_new)

        actor_loss = (self.alpha * log_prob - Q_new).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # ----------------------- update target networks ----------------------- #
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def save_models(self):
        torch.save(self.actor.state_dict(), 'tmp/sac/actor.pth')
        torch.save(self.critic.state_dict(), 'tmp/sac/critic.pth')
        torch.save(self.critic_target.state_dict(), 'tmp/sac/critic_target.pth')
        print("SAC models saved successfully.")

    def load_models(self):
        try:
            self.actor.load_state_dict(torch.load('tmp/sac/actor.pth', map_location=self.device))
            self.critic.load_state_dict(torch.load('tmp/sac/critic.pth', map_location=self.device))
            self.critic_target.load_state_dict(torch.load('tmp/sac/critic_target.pth', map_location=self.device))
            print("SAC models loaded successfully.")
        except:
            print("Failed to load SAC models. Starting from scratch.")