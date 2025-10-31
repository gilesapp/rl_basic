from collections import namedtuple, deque
import numpy as np
import random
import torch

field_names = ["state", "action", "reward", "next_state", "done"]


class ReplayBuffer:
    """ Fixed-size buffer to store experience tuples"""

    def __init__(self, buffer_size, batch_size, seed, device):
        """Initialize a ReplayBuffer object. """
        self.buffer_size = buffer_size  # size of replay buffer
        self.batch_size = batch_size  # how many mem tuples to sample at a time
        random.seed(seed)
        self.device = device

        # Define Named Tuple - field_names=["state", "action", "reward", "next_state", "done"]
        self.experience = namedtuple("Experience", field_names=field_names)

        # Data structure to hold the memories
        self.memory = deque(maxlen=buffer_size)

    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)

    def sample(self):
        """ Randomly sample a batch of experiences """

        # Sample an experience with length k from list of memories
        random.seed()
        experiences = random.sample(self.memory, k=self.batch_size)

        # For each item in the tuple, stack vertically and convert to GPU torch tensor
        states = np.array([e.state for e in experiences if e is not None])
        states = torch.from_numpy(states).float().to(self.device)  # (float)

        actions = np.array([e.action for e in experiences if e is not None])
        actions = torch.from_numpy(actions).float().to(self.device)  # (float)

        rewards = np.array([e.reward for e in experiences if e is not None])
        rewards = torch.from_numpy(rewards).float().to(self.device)  # (float)

        next_states = np.array([e.next_state for e in experiences if e is not None])
        next_states = torch.from_numpy(next_states).float().to(self.device)  # float

        dones = np.array([e.done for e in experiences if e is not None]).astype(np.uint8)  # Make bool an int
        dones = torch.from_numpy(dones).float().to(self.device)

        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)
    
    
class RolloutBuffer:
    def __init__(self, device):
        self.states, self.actions, self.logps, self.rews, self.dones, self.values = [], [], [], [], [], []
        self.device = device
    
    def add(self, s, a, logp, r, d, v):
        self.states.append(s); self.actions.append(a); self.logps.append(logp)
        self.rews.append(r); self.dones.append(d); self.values.append(v)
    
    def compute_returns_advantages(self, last_val, gamma=0.99, lam=0.95):
        rewards = np.array(self.rews + [last_val])
        dones   = np.array(self.dones + [0])
        values  = np.array(self.values + [last_val])

        deltas = rewards[:-1] + gamma * values[1:] * (1 - dones[:-1]) - values[:-1]
        gae = 0; advs = []
        for step in reversed(range(len(deltas))):
            gae = deltas[step] + gamma * lam * (1 - dones[step]) * gae
            advs.insert(0, gae)
        returns = np.array(advs) + np.array(self.values)
        
        return torch.tensor(advs, dtype=torch.float32).to(self.device), torch.tensor(returns, dtype=torch.float32).to(self.device)
    
    def get(self):
        return (torch.tensor(np.array(self.states), dtype=torch.float32).to(self.device),
                torch.tensor(np.array(self.actions), dtype=torch.float32).to(self.device),
                torch.tensor(np.array(self.logps),  dtype=torch.float32).to(self.device),
                *self.compute_returns_advantages(0))