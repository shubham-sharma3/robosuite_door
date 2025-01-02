import torch
import torch.nn as nn
import torch.nn.functional as F


class ActorNetwork(nn.Module):
    def __init__(self, input_dims, fc1_dims=256, fc2_dims=128, n_actions=2, name='actor'):
        super(ActorNetwork, self).__init__()
        self.fc1 = nn.Linear(*input_dims, fc1_dims)
        self.fc2 = nn.Linear(fc1_dims, fc2_dims)
        self.mean = nn.Linear(fc2_dims, n_actions)
        self.log_std = nn.Linear(fc2_dims, n_actions)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        # Initialize weights to small random values
        for layer in [self.fc1, self.fc2, self.mean, self.log_std]:
            nn.init.kaiming_normal_(layer.weight, nonlinearity='relu')
            nn.init.constant_(layer.bias, 0)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        mean = self.mean(x)
        log_std = self.log_std(x)
        log_std = torch.clamp(log_std, min=-20, max=2)  # To prevent too large or small std
        std = torch.exp(log_std)
        return mean, std


class CriticNetwork(nn.Module):
    def __init__(self, input_dims, fc1_dims=256, fc2_dims=128, n_actions=2, name='critic'):
        super(CriticNetwork, self).__init__()
        # Q1 architecture
        self.fc1 = nn.Linear(*input_dims, fc1_dims)
        self.fc2 = nn.Linear(fc1_dims + n_actions, fc2_dims)
        self.q = nn.Linear(fc2_dims, 1)

        # Q2 architecture
        self.fc1_2 = nn.Linear(*input_dims, fc1_dims)
        self.fc2_2 = nn.Linear(fc1_dims + n_actions, fc2_dims)
        self.q_2 = nn.Linear(fc2_dims, 1)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        # Initialize weights to small random values
        for layer in [self.fc1, self.fc2, self.q, self.fc1_2, self.fc2_2, self.q_2]:
            nn.init.kaiming_normal_(layer.weight, nonlinearity='relu')
            nn.init.constant_(layer.bias, 0)

    def forward(self, state, action):
        # Q1 forward
        x = F.relu(self.fc1(state))
        x = torch.cat([x, action], dim=1)
        x = F.relu(self.fc2(x))
        q1 = self.q(x)

        # Q2 forward
        x2 = F.relu(self.fc1_2(state))
        x2 = torch.cat([x2, action], dim=1)
        x2 = F.relu(self.fc2_2(x2))
        q2 = self.q_2(x2)

        return q1, q2

    def Q1(self, state, action):
        x = F.relu(self.fc1(state))
        x = torch.cat([x, action], dim=1)
        x = F.relu(self.fc2(x))
        q1 = self.q(x)
        return q1