import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from torch.distributions import Normal


def hidden_init(layer):
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return (-lim, lim)


class QNet(nn.Module):
    """Q-net Model."""

    def __init__(self, state_size, action_size, seed, fc1_units=128, fc2_units=128):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            seed (int): Random seed
            fc1_units (int): Number of nodes in the first hidden layer
            fc2_units (int): Number of nodes in the second hidden layer
        """
        super(QNet, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units, action_size)
        # self.fc3 = nn.Linear(fc2_units, action_size)
        self.reset_parameters()

    def reset_parameters(self):
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        # self.fc3.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state):
        """Build a q-value network that maps state -> Q-values (arg: actions)."""
        x = F.relu(self.fc1(state))
        return self.fc2(x)
    
    
class Actor(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed, fc1_units=128, fc2_units=128):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fc1_units (int): Number of nodes in first hidden layer
            fc2_units (int): Number of nodes in second hidden layer
        """
        super(Actor, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.fc3 = nn.Linear(fc2_units, action_size)
        self.reset_parameters()

    def reset_parameters(self):
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state):
        """Build an actor (policy) network that maps states -> actions."""
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return F.tanh(self.fc3(x))


class Critic(nn.Module):
    """Critic (Value) Model."""

    def __init__(self, state_size, action_size, seed, fc1_units=128, fc2_units=128):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fc1_units (int): Number of nodes in the first hidden layer
            fc2_units (int): Number of nodes in the second hidden layer
        """
        super(Critic, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units+action_size, fc2_units)
        self.fc3 = nn.Linear(fc2_units, 1)
        self.reset_parameters()

    def reset_parameters(self):
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state, action):
        """Build a critic (value) network that maps (state, action) pairs -> Q-values."""
        x = F.relu(self.fc1(state))
        x = torch.cat((x, action), dim=1)
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class MLP(nn.Module):
    def __init__(self, d_in, d_out, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, d_out)
        )
    def forward(self, x):
        return self.net(x)
    

class SoftActor(nn.Module):
    def __init__(self, s_dim, a_dim, a_bound=1.0):
        super().__init__()
        self.backbone = MLP(s_dim, 256)
        self.mu = nn.Linear(256, a_dim)
        self.log_std = nn.Linear(256, a_dim)
        self.a_bound = a_bound
    
    def dist(self, s):
        h = self.backbone(s)
        mu = self.mu(h)
        log_std = torch.clamp(self.log_std(h), -20, 2)
        return Normal(mu, log_std.exp())
    
    def forward(self, s, deterministic=False):
        dist = self.dist(s)
        if deterministic:
            a = dist.mean
        else:
            a = dist.rsample() 
        a = torch.tanh(a) * self.a_bound
        log_pi = dist.log_prob(a/self.a_bound).sum(-1, keepdim=True)
        log_pi -= (2*(np.log(2) - a - F.softplus(-2*a))).sum(-1, keepdim=True)
        return a, log_pi


class SoftCritic(nn.Module):
    def __init__(self, s_dim, a_dim):
        super().__init__()
        self.q1 = MLP(s_dim+a_dim, 1)
        self.q2 = MLP(s_dim+a_dim, 1)
        
    def both(self, s, a):
        sa = torch.cat([s, a], -1)
        return self.q1(sa), self.q2(sa)
    
    def forward(self, s, a):
        return torch.min(*self.both(s, a))