import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim


class bModel:
    """Basic functions"""

    def __init__(self, state_size, action_size, random_seed, agent_wrapper=None,
                 BUFFER_SIZE=1e4,
                 BUFFER_THRESHOLD=500,
                 BATCH_SIZE = 64,
                 GAMMA = 0.99,
                 SOFT_UPDATE_ITER = 10,
                 LR = 1e-3,
                 EPSILON = 0.05):
        """Initialize Base model.

        Params
        ======
            state_size (int):          dimension of each state
            action_size (int):         dimension of each action
            n_agents (int):            number of agents
            random_seed (int):         random seed
            BUFFER_SIZE (int):         replay buffer size
            BATCH_SIZE (int):          minibatch size
            GAMMA (float):             discount factor
            SOFT_UPDATE_ITER (int):    soft update iterator
            LR = (float):              learning rate
            EPSILON = (float):         greedy epsilon
        """
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model_name = 'BaseModel'
        self.agent_wrapper = agent_wrapper
        
        # meta param
        self.BUFFER_SIZE = int(BUFFER_SIZE)
        self.BUFFER_THRESHOLD = BUFFER_THRESHOLD
        self.BATCH_SIZE = BATCH_SIZE
        self.GAMMA = GAMMA 
        self.SOFT_UPDATE_ITER = SOFT_UPDATE_ITER
        self.LR = LR
        self.EPSILON = EPSILON

        # model param
        self.state_size = state_size
        self.action_size = action_size
        self.random_seed = random_seed
        self.isEval = False
        self.epislon_decay = 1
        self.learn_step_counter = 0
        
        # update action bound
        if self.agent_wrapper:
            self.agent_wrapper.update_action_bound(self.get_action_space(), self.action_size)

    def wrap_action(self, action):
        """Wrap action."""
        if self.agent_wrapper:
            return self.agent_wrapper.action_wrapper(action)
        else:
            action
        
    def get_action_space(self):
        """Returns model action space."""
        return []
    
    def load_model(self, pth):
        """***.load_state_dict(torch.load(pth))."""
        pass
    
    def save_model(self, pth):
        """torch.save(***.state_dict(), pth)."""
        pass

    def eval(self):
        self.isEval = True

    def act(self, state):
        """Returns actions for given state as per current policy."""
        return []

    def step(self, states, actions, rewards, next_states, dones):
        """Save experience in replay memory, and use random sample from buffer to learn."""
        pass

    def learn(self, experiences, gamma):
        """Update policy and value parameters using given batch of experience tuples."""
        pass
