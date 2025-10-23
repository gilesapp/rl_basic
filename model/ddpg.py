import numpy as np
from .replay_buffer import ReplayBuffer
from .actor_critic import Actor, Critic
import torch
import torch.nn.functional as F
import torch.optim as optim
import random
import copy

BUFFER_SIZE = int(1e5)  # replay buffer size
BATCH_SIZE = 100  # minibatch size
GAMMA = 0.99  # discount factor
SOFT_UPDATE_ITER = 10  # soft update iterator
TAU = 1e-3  # for soft update of target parameters
LR_ACTOR = 1e-3  # learning rate of the actor
LR_CRITIC = 1e-3  # learning rate of the critic
WEIGHT_DECAY = 0  # L2 weight decay
EPSILON = 0.1  # greedy epsilon

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class OUNoise:
    """Ornstein-Uhlenbeck process."""

    def __init__(self, size, seed, mu=0., theta=0.15, sigma=0.2):
        """Initialize parameters and noise process."""
        self.size = size
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.state = self.mu
        random.seed(seed)
        self.reset()

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = copy.copy(self.mu)

    def sample(self):
        """Update internal state and return it as a noise sample."""
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.standard_normal(self.size)
        self.state = x + dx
        return self.state
    
    
def hard_update(source, target):
    for tp, sp in zip(target.parameters(), source.parameters()):
        tp.data.copy_(sp.data)    
    
    
def soft_update(local_model, target_model, tau):
    """Soft update model parameters.
    θ_target = τ*θ_local + (1 - τ)*θ_target
    Params
    ======
        local_model: PyTorch model (weights will be copied from)
        target_model: PyTorch model (weights will be copied to)
        tau (float): interpolation parameter
    """
    for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
        target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)


class DDPG_Agent:
    """Interacts with and learns from the environment."""

    def __init__(self, state_size, action_size, n_agents, random_seed):
        """Initialize an Agent object.

        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            random_seed (int): random seed
        """
        self.state_size = state_size
        self.action_size = action_size
        self.isEval = False
        random.seed(random_seed)
        self.epislon_decay = 1
        self.learn_step_counter = 0

        # Actor Network (w/ Target Network)
        self.actor_local = Actor(state_size, action_size, random_seed, 256).to(device)
        self.actor_target = Actor(state_size, action_size, random_seed, 256).to(device)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=LR_ACTOR)

        # Critic Network (w/ Target Network)
        self.critic_local = Critic(state_size, action_size, random_seed, 256).to(device)
        self.critic_target = Critic(state_size, action_size, random_seed, 256).to(device)
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=LR_CRITIC, weight_decay=WEIGHT_DECAY)
        
        hard_update(self.actor_local, self.actor_target)
        hard_update(self.critic_local, self.critic_target)

        # Noise process
        self.noise = OUNoise((n_agents, action_size), random_seed)

        # Replay memory
        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, random_seed, device)

    def load_model(self, pth):
        self.actor_local.load_state_dict(torch.load(pth))
        # self.actor_target.load_state_dict(torch.load(pth))
    
    def save_model(self, pth):
        # self.actor_local._save_to_state_dict(pth)
        torch.save(self.actor_local.state_dict(), pth)

    def eval(self):
        self.isEval = True

    def act(self, state):
        """Returns actions for given state as per current policy."""
        state = np.expand_dims(state, 0)
        state = torch.tensor(state, dtype=torch.float).to(device)

        self.actor_local.eval()
        with torch.no_grad():
            action = self.actor_local(state).cpu().data
        self.actor_local.train()
       
        # if not eval mode and greater than greedy epsilon
        # if not self.isEval and self.epislon_decay > 0.1 and np.random.uniform() > EPSILON:
        if not self.isEval and np.random.random() < max(self.epislon_decay, EPSILON):
            noise = self.noise.sample()
            action += noise
            
        action = action[0]
        # action = F.softmax(action[0], dim=0)
        
        # return action
        return torch.clip(action, -1, 1).numpy()

    def step(self, states, actions, rewards, next_states, dones):
        """Save experience in replay memory, and use random sample from buffer to learn."""
        # Save experience / reward

        # for state, action, reward, next_state, done in zip(states, actions, rewards, next_states, dones):
        #     self.memory.add(state, action, reward, next_state, done)

        self.memory.add(states, actions, rewards, next_states, dones)

        # Learn, if enough samples are available in memory
        mem_len = len(self.memory)

        # if mem_len % BATCH_SIZE == 0 or (mem_len > 2000 and mem_len % (BATCH_SIZE//2) == 0):
        if mem_len > 1000 and mem_len % SOFT_UPDATE_ITER == 0:
            experiences = self.memory.sample()
            self.learn(experiences, GAMMA)
            self.epislon_decay *= 0.98


    def learn(self, experiences, gamma):
        """Update policy and value parameters using given batch of experience tuples.
        Q_targets = r + γ * critic_target(next_state, actor_target(next_state))
        where:
            actor_target(state) -> action
            critic_target(state, action) -> Q-value
        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples
            gamma (float): discount factor
        """
        states, actions, rewards, next_states, dones = experiences
        
        # ---------------------------- update critic ---------------------------- #
        # Get predicted next-state actions and Q values from target models
        with torch.no_grad():
            actions_next = self.actor_target(next_states)
            Q_targets_next = self.critic_target(next_states, actions_next)
            # Compute Q targets for current states (y_i)
            Q_targets = rewards.unsqueeze(1) + (gamma * Q_targets_next * (1 - dones.unsqueeze(1)))
        
        # Compute critic loss
        Q_expected = self.critic_local(states, actions)
        critic_loss = F.mse_loss(Q_expected, Q_targets)
        
        # Minimize the loss
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        # ---------------------------- update actor ---------------------------- #
        # Compute actor loss
        self.critic_local.eval()
        actions_pred = self.actor_local(states)
        actor_loss = -1 * self.critic_local(states, actions_pred)
        actor_loss = actor_loss.mean()
        self.critic_local.train()
        # print("actor loss", actor_loss.item(),
        #       "actor grad", self.actor_local.parameters().__next__().grad)
        # Minimize the loss
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # ----------------------- update target networks ----------------------- #
        if self.learn_step_counter % SOFT_UPDATE_ITER == 0:
            soft_update(self.critic_local, self.critic_target, TAU)
            soft_update(self.actor_local, self.actor_target, TAU)
        self.learn_step_counter += 1
