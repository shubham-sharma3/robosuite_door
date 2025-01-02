import numpy as np
import torch


class ReplayBuffer:
    def __init__(self, max_size, input_dims, n_actions, device=torch.device('cpu')):
        self.max_size = max_size
        self.ptr = 0
        self.size = 0
        self.device = device

        self.state_memory = np.zeros((max_size, *input_dims), dtype=np.float32)
        self.new_state_memory = np.zeros((max_size, *input_dims), dtype=np.float32)
        self.action_memory = np.zeros((max_size, n_actions), dtype=np.float32)
        self.reward_memory = np.zeros(max_size, dtype=np.float32)
        self.done_memory = np.zeros(max_size, dtype=np.float32)

    def store_transition(self, state, action, reward, state_, done):
        index = self.ptr % self.max_size
        self.state_memory[index] = state
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.new_state_memory[index] = state_
        self.done_memory[index] = done
        self.ptr += 1
        if self.size < self.max_size:
            self.size += 1

    def sample_buffer(self, batch_size):
        max_mem = self.size
        batch = np.random.choice(max_mem, batch_size, replace=False)

        states = torch.tensor(self.state_memory[batch]).to(self.device)
        actions = torch.tensor(self.action_memory[batch]).to(self.device)
        rewards = torch.tensor(self.reward_memory[batch]).unsqueeze(1).to(self.device)
        states_ = torch.tensor(self.new_state_memory[batch]).to(self.device)
        dones = torch.tensor(self.done_memory[batch]).unsqueeze(1).to(self.device)

        return states, actions, rewards, states_, dones

    def __len__(self):
        return self.size