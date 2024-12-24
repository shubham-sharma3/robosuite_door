import numpy as np

class PPORolloutBuffer:
    def __init__(self, max_size, input_dims, n_actions):
        self.states = np.zeros((max_size, *input_dims), dtype=np.float32)
        self.actions = np.zeros((max_size, n_actions), dtype=np.float32)
        self.log_probs = np.zeros(max_size, dtype=np.float32)
        self.rewards = np.zeros(max_size, dtype=np.float32)
        self.is_terminals = np.zeros(max_size, dtype=np.float32)
        self.ptr = 0
        self.max_size = max_size

    def store_transition(self, state, action, log_prob, reward, done):
        if self.ptr >= self.max_size:
            raise Exception("PPO Buffer is full")
        self.states[self.ptr] = state
        self.actions[self.ptr] = action
        self.log_probs[self.ptr] = log_prob
        self.rewards[self.ptr] = reward
        self.is_terminals[self.ptr] = done
        self.ptr += 1

    def clear(self):
        self.ptr = 0

    def get(self):
        return (self.states[:self.ptr],
                self.actions[:self.ptr],
                self.log_probs[:self.ptr],
                self.rewards[:self.ptr],
                self.is_terminals[:self.ptr])