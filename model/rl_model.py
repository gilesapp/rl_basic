import numpy as np
from .replay_buffer import ReplayBuffer
from .bNN import *
import torch
import torch.nn.functional as F
import torch.optim as optim
import random
import copy
from .bModel import bModel
from .util import hard_update, soft_update


"""
Ornstein-Uhlenbeck process
"""
class OUNoise:
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


"""
Implementation of DDPG
"""
class DDPG(bModel):
    def __init__(self, *args, 
                 TAU = 1e-3,
                 LR_ACTOR=1e-3, 
                 LR_CRITIC=1e-3,
                 WEIGHT_DECAY=0, 
                 **kw):
        """
        Params
        ======
            TAU (float):               soft update of target parameters
            LR_ACTOR (float):          learning rate of actor
            LR_CRITIC (float):         learning rate of critic
            WEIGHT_DECAY (int):        L2 weight decay
        """
        super().__init__(*args, **kw)

        # meta param
        self.TAU = TAU
        self.LR_ACTOR = LR_ACTOR
        self.LR_CRITIC = LR_CRITIC
        self.WEIGHT_DECAY = WEIGHT_DECAY
        
        # Actor Network (w/ Target Network)
        self.actor_local = Actor(self.state_size, self.action_size, self.random_seed, 256).to(self.device)
        self.actor_target = Actor(self.state_size, self.action_size, self.random_seed, 256).to(self.device)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=self.LR_ACTOR)

        # Critic Network (w/ Target Network)
        self.critic_local = Critic(self.state_size, self.action_size, self.random_seed, 256).to(self.device)
        self.critic_target = Critic(self.state_size, self.action_size, self.random_seed, 256).to(self.device)
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=self.LR_CRITIC, weight_decay=self.WEIGHT_DECAY)
        
        hard_update(self.actor_target, self.actor_local)
        hard_update(self.critic_target, self.critic_local)

        # Noise process
        self.noise = OUNoise((self.n_agents, self.action_size), self.random_seed)

    def get_action_space(self):
        return [[-1.0, 1.0] for i in range(self.action_size)]
    
    def load_model(self, pth):
        self.actor_local.load_state_dict(torch.load(pth))
    
    def save_model(self, pth):
        torch.save(self.actor_local.state_dict(), pth)

    def act(self, state):
        """Returns actions for given state as per current policy."""
        state = np.expand_dims(state, 0)
        state = torch.tensor(state, dtype=torch.float).to(self.device)

        self.actor_local.eval()
        with torch.no_grad():
            action = self.actor_local(state).cpu().data
        self.actor_local.train()
       
        # if not eval mode and greater than greedy epsilon
        # if not self.isEval and self.epislon_decay > 0.1 and np.random.uniform() > EPSILON:
        if not self.isEval and np.random.random() < max(self.epislon_decay, self.EPSILON):
            noise = self.noise.sample()
            action += noise
            
        action = action[0]

        return torch.clip(action, -1, 1).numpy()

    def step(self, states, actions, rewards, next_states, dones):
        """Save experience in replay memory, and use random sample from buffer to learn."""
        # Save experience / reward
        for state, action, reward, next_state, done in zip(states, actions, rewards, next_states, dones):
            self.memory.add(state, action, reward, next_state, done)

        # Learn, if enough samples are available in memory
        mem_len = len(self.memory)

        # if mem_len >= self.BUFFER_THRESHOLD and mem_len % self.SOFT_UPDATE_ITER == 0:
        if mem_len >= self.BUFFER_THRESHOLD:
            experiences = self.memory.sample()
            self.learn(experiences, self.GAMMA)
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
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        # ---------------------------- update actor ---------------------------- #
        # Compute actor loss
        actions_pred = self.actor_local(states)
        actor_loss = -self.critic_local(states, actions_pred).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # ----------------------- update target networks ----------------------- #
        if self.learn_step_counter % self.SOFT_UPDATE_ITER == 0:
            soft_update(self.critic_target, self.critic_local, self.TAU)
            soft_update(self.actor_target, self.actor_local, self.TAU)
        self.learn_step_counter += 1


"""
Implementation of DQN
"""
class DQN(bModel):
    def __init__(self, *args, **kw):
        """
        Params
        ======
            bModel
        """
        super().__init__(*args, **kw)

        # Q Network
        self.qnet_local = QNet(self.state_size, self.action_size, self.random_seed, 128).to(self.device)
        self.qnet_target = QNet(self.state_size, self.action_size, self.random_seed, 128).to(self.device)
        self.optimizer = optim.Adam(self.qnet_local.parameters(), lr=self.LR)
  
        hard_update(self.qnet_target, self.qnet_local)
    
    def load_model(self, pth):
        self.qnet_local.load_state_dict(torch.load(pth))
    
    def save_model(self, pth):
        torch.save(self.qnet_local.state_dict(), pth)

    def act(self, state):
        """Returns actions for given state as per current policy."""
        state = np.expand_dims(state, 0)
        state = torch.tensor(state, dtype=torch.float).to(self.device)

        if not self.isEval and np.random.random() < max(self.epislon_decay, self.EPSILON):
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

        for state, action, reward, next_state, done in zip(states, actions, rewards, next_states, dones):
            self.memory.add(state, action, reward, next_state, done)

        # Learn, if enough samples are available in memory
        mem_len = len(self.memory)

        # if mem_len >= self.BUFFER_THRESHOLD and mem_len % self.SOFT_UPDATE_ITER == 0:
        if mem_len >= self.BUFFER_THRESHOLD:
            experiences = self.memory.sample()
            self.learn(experiences, self.GAMMA)
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
        if self.learn_step_counter % self.SOFT_UPDATE_ITER == 0:
            hard_update(self.qnet_target, self.qnet_local)
        self.learn_step_counter += 1
        
        
"""
Implementation of SAC
"""
class SAC(bModel):
    def __init__(self, *args, 
                 TAU=0.005, 
                 ALPHA=0.2,
                 AUTO_ALPHA=True, 
                 TARGET_ENT=-2.0,
                 DETERM=False,
                 **kw):
        """
        Params
        ======
            TAU (float):               soft update of target parameters
            ALPHA (float):             
            AUTO_ALPHA (float):        alpha learning
            TARGET_ENT (int):          
        """
        super().__init__(*args, **kw)

        # meta param
        self.TAU = TAU
        self.ALPHA = ALPHA
        self.AUTO_ALPHA = AUTO_ALPHA
        self.TARGET_ENT = TARGET_ENT
        self.DETERM = DETERM
        
        # Actor/Critic Network (w/ Critic Target Network)
        self.actor = SoftActor(self.state_size, self.action_size).to(self.device)
        self.critic = SoftCritic(self.state_size, self.action_size).to(self.device)
        self.critic_tgt = SoftCritic(self.state_size, self.action_size).to(self.device)
        
        # self.critic_tgt.load_state_dict(self.critic.state_dict())
        hard_update(self.critic_tgt, self.critic)

        self.log_alpha = torch.tensor(np.log(self.ALPHA), requires_grad=True, device=self.device)
        self.alpha_opt = optim.Adam([self.log_alpha], lr=self.LR) if self.AUTO_ALPHA else None
        self.target_ent = self.TARGET_ENT if self.TARGET_ENT else -self.action_size

        self.actor_opt  = optim.Adam(self.actor.parameters(),  lr=self.LR)
        self.critic_opt = optim.Adam(self.critic.parameters(), lr=self.LR)

    def get_action_space(self):
        return [] if self.DETERM else [[-1.0, 1.0]]
    
    @property
    def alpha(self):
        return self.log_alpha.exp()

    def act(self, state):
        state = np.expand_dims(state, 0)
        state = torch.tensor(state, dtype=torch.float).to(self.device)
        
        self.actor.eval()
        with torch.no_grad():
            action, _ = self.actor(state, self.DETERM)
        self.actor.train()
        action = action.cpu().data
        action = action[0]
        return torch.clip(action, -1, 1).numpy()

    def step(self, states, actions, rewards, next_states, dones):
        """Save experience in replay memory, and use random sample from buffer to learn."""
        # Save experience / reward

        for state, action, reward, next_state, done in zip(states, actions, rewards, next_states, dones):
            self.memory.add(state, action, reward, next_state, done)

        # Learn, if enough samples are available in memory
        mem_len = len(self.memory)

        # if mem_len >= self.BUFFER_THRESHOLD and mem_len % self.SOFT_UPDATE_ITER == 0:
        if mem_len >= self.BUFFER_THRESHOLD:
            experiences = self.memory.sample()
            self.learn(experiences, self.GAMMA)
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
        
        # ----- Critic loss -----
        with torch.no_grad():
            action_next, log_pi_next = self.actor(next_states)
            Q_targets_next = self.critic_tgt(next_states, action_next) - self.alpha * log_pi_next
            Q_targets = rewards.unsqueeze(1) + (gamma * Q_targets_next * (1 - dones.unsqueeze(1)))
            
        Q_expected1, Q_expected2 = self.critic.both(states, actions)
        critic_loss = F.mse_loss(Q_expected1, Q_targets) + F.mse_loss(Q_expected2, Q_targets)

        self.critic_opt.zero_grad()
        critic_loss.backward()
        self.critic_opt.step()

        # ----- Actor loss -----
        actions_pred, log_pi = self.actor(states)
        Q_expected = self.critic(states, actions_pred)
        actor_loss = (self.alpha * log_pi - Q_expected).mean()

        self.actor_opt.zero_grad()
        actor_loss.backward()
        self.actor_opt.step()

        # ----- Alpha loss -----
        if self.alpha_opt:
            alpha_loss = -(self.alpha * (log_pi + self.target_ent).detach()).mean()
            self.alpha_opt.zero_grad()
            alpha_loss.backward()
            self.alpha_opt.step()

        # ----------------------- update target networks ----------------------- #
        if self.learn_step_counter % self.SOFT_UPDATE_ITER == 0:
            soft_update(self.critic_tgt, self.critic)
        self.learn_step_counter += 1