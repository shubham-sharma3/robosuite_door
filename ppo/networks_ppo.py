import os
import torch
import torch.nn as nn
import torch.nn.functional as F


class ActorNetwork(nn.Module):
    def __init__(self, input_dims, fc1_dims=256, fc2_dims=128,
                 n_actions=2, name='actor', checkpoint_dir='tmp/ppo'):
        super(ActorNetwork, self).__init__()
        self.input_dims = input_dims
        self.fc1 = nn.Linear(*self.input_dims, fc1_dims)
        self.fc2 = nn.Linear(fc1_dims, fc2_dims)
        self.output = nn.Linear(fc2_dims, n_actions)

        # Initialize weights
        self._init_weights()

        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

    def _init_weights(self):
        # Initialize weights to small random values
        for layer in [self.fc1, self.fc2, self.output]:
            nn.init.kaiming_normal_(layer.weight, nonlinearity='relu')
            nn.init.constant_(layer.bias, 0)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        action_mean = torch.tanh(self.output(x))
        return action_mean


class CriticNetwork(nn.Module):
    def __init__(self, input_dims, fc1_dims=256, fc2_dims=128,
                 n_actions=2, name='critic', checkpoint_dir='tmp/ppo'):
        super(CriticNetwork, self).__init__()
        self.input_dims = input_dims
        self.fc1 = nn.Linear(*self.input_dims, fc1_dims)
        self.fc2 = nn.Linear(fc1_dims + n_actions, fc2_dims)
        self.output = nn.Linear(fc2_dims, 1)

        # Initialize weights
        self._init_weights()

        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

    def _init_weights(self):
        # Initialize weights to small random values
        for layer in [self.fc1, self.fc2, self.output]:
            nn.init.kaiming_normal_(layer.weight, nonlinearity='relu')
            nn.init.constant_(layer.bias, 0)

    def forward(self, state, action):
        x = F.relu(self.fc1(state))
        x = torch.cat([x, action], dim=1)
        x = F.relu(self.fc2(x))
        state_value = self.output(x)
        return state_value