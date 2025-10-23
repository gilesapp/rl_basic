import numpy as np
from .replay_buffer import ReplayBuffer
from .qnet import QNet
import torch
import torch.nn.functional as F
import torch.optim as optim
import random
import copy

BUFFER_SIZE = int(1e4)  # replay buffer size
BATCH_SIZE = 128  # minibatch size
GAMMA = 0.99  # discount factor
SOFT_UPDATE_ITER = 10  # soft update iterator
LR_ACTOR = 2e-3  # learning rate of the actor
LR_CRITIC = 1e-3  # learning rate of the critic
WEIGHT_DECAY = 0  # L2 weight decay
EPSILON = 0.01  # greedy epsilon

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    
def hard_update(source, target):
    for tp, sp in zip(target.parameters(), source.parameters()):
        tp.data.copy_(sp.data)    
    

class DQN_Agent:
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

        # Q Network
        self.qnet_local = QNet(state_size, action_size, random_seed, 128).to(device)
        self.qnet_target = QNet(state_size, action_size, random_seed, 128).to(device)
        self.optimizer = optim.Adam(self.qnet_local.parameters(), lr=LR_ACTOR)
  
        hard_update(self.qnet_local, self.qnet_target)

        # Replay memory
        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, random_seed, device)

    def load_model(self, pth):
        self.qnet_local.load_state_dict(torch.load(pth))
    
    def save_model(self, pth):
        torch.save(self.qnet_local.state_dict(), pth)

    def eval(self):
        self.isEval = True

    def act(self, state):
        """Returns actions for given state as per current policy."""
        state = np.expand_dims(state, 0)
        state = torch.tensor(state, dtype=torch.float).to(device)

        if not self.isEval and np.random.random() < max(self.epislon_decay, EPSILON):
            action = np.random.randint(self.action_size)
        else:
            self.qnet_local.eval()
            with torch.no_grad():
                action = self.qnet_local(state).cpu().data.argmax().item()
            self.qnet_local.train()
 
        return action

    def step(self, states, actions, rewards, next_states, dones):
        """Save experience in replay memory, and use random sample from buffer to learn."""
        # Save experience / reward

        # for state, action, reward, next_state, done in zip(states, actions, rewards, next_states, dones):
        #     self.memory.add(state, action, reward, next_state, done)

        self.memory.add(states, actions, rewards, next_states, dones)

        # Learn, if enough samples are available in memory
        mem_len = len(self.memory)

        if mem_len > 500 and mem_len % SOFT_UPDATE_ITER == 0:
            experiences = self.memory.sample()
            self.learn(experiences, GAMMA)
            self.epislon_decay *= 0.98


    def learn(self, experiences, gamma):
        """Update policy and value parameters using given batch of experience tuples.
        Q_targets = r + γ * qnet_target(next_state)
        where:
            qnet_target(state) -> Q-value (arg: action)
        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples
            gamma (float): discount factor
        """
        states, actions, rewards, next_states, dones = experiences
        
        # ---------------------------- update qnet ---------------------------- #
        # Get predicted next-state actions and Q values from target models
        with torch.no_grad():
            Q_targets_next = self.qnet_target(next_states).max(1)[0].view(-1, 1)
            Q_targets = rewards.unsqueeze(1) + (gamma * Q_targets_next * (1 - dones.unsqueeze(1)))
            
        # Compute q loss
        Q_expected = self.qnet_local(states).gather(1, actions.long().unsqueeze(1))
        q_loss = F.mse_loss(Q_expected, Q_targets)
     
        # Minimize the loss
        self.optimizer.zero_grad()
        q_loss.backward()
        self.optimizer.step()

        # ----------------------- update target networks ----------------------- #
        if self.learn_step_counter % SOFT_UPDATE_ITER == 0:
            hard_update(self.qnet_local, self.qnet_target)
        self.learn_step_counter += 1
