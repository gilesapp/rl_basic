import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F
from .rl_model import *
from .util import linear_map
import numpy as np 


class bAgentWrapper(): 
    def __init__(self):
        self.model_abound = []
        self.env_abound = []
        self.isMap = False
        self.env_action_space = None
    
    def update_env_action_space(self, env_action_space):
        self.env_action_space = env_action_space
    
    def update_model_abound(self, model_abound):
        self.model_abound = model_abound
        
    # init action bound according to env
    def update_action_bound(self, model_abound, action_size): 
        assert self.env_action_space, "please call 'update_env_action_space()' in agent.make_model() before update action bound"
        self.update_model_abound(model_abound)
        if isinstance(self.env_action_space, gym.spaces.box.Box):
            env_a_low = self.env_action_space.low
            env_a_high = self.env_action_space.high
            
            for i in range(action_size):
                if isinstance(env_a_low, np.ndarray):
                    env_a_low_i = env_a_low[i]
                    env_a_high_i = env_a_high[i]
                else:
                    env_a_low_i = env_a_low
                    env_a_high_i = env_a_high
        
                if not (env_a_low_i == model_abound[i][0] and env_a_high_i == model_abound[i][1]):
                    self.env_abound.append([env_a_low_i, env_a_high_i])
                    self.isMap = True
        
        if self.isMap:
            print(f"starting to map model action bound [{model_abound}] -> env action bound [{self.env_abound}]")
    
    def state_wrapper(self, state):
        if isinstance(state, dict):
            return state['observation']
        else:
            return state

    def action_wrapper(self, actions): # batch
        if self.isMap:
            if isinstance(actions, torch.Tensor):
                actions_new = actions.clone()
            else:
                actions_new = np.ones_like(actions)
            for a_idx in range(actions.shape[1]):
                actions_new[:, a_idx] = (linear_map(actions[:, a_idx], self.model_abound[a_idx][0], self.model_abound[a_idx][1], self.env_abound[a_idx][0], self.env_abound[a_idx][1]))
            return actions_new
        else:
            return actions
        
    
class bAgent():
    def __init__(self, env, n_agent=1, seed_num=2025, agent_wrapper=bAgentWrapper(), debug_info=False):
        self.env = env  
        self.n_agent = n_agent
        self.seed_num = seed_num
        self.agent_wrapper = agent_wrapper
        self.debug_info = debug_info
        
        self.model_save_pth = 'ckpt/'
        self.fig_save_pth = 'fig/'
                  
    def log_info(self, debug=False):
        print(f"init {self.n_agent} {self.model.model_name} agent(s) success")
        print(f"env: {self.env.spec.id}, state size {self.state_size}, action size {self.action_size}")
        if debug:
            print(f"--debug-- env state space {self.env.observation_space}")
            print(f"--debug-- env action space {self.env.action_space}")

    def update_sa(self):
        state, _ = self.env.reset()
        state = self.agent_wrapper.state_wrapper(state)
        self.state_size = len(state)
        if isinstance(self.env.action_space, gym.spaces.discrete.Discrete):
            self.action_size = self.env.action_space.n
        elif isinstance(self.env.action_space, gym.spaces.box.Box):
            self.action_size = self.env.action_space.shape[0]
    
    def make_model(self, model, **kw):
        self.update_sa() # init state size and action size
        self.agent_wrapper.update_env_action_space(self.env.action_space) # register env action bound to action wrapper
        self.model = model(self.state_size, self.action_size, self.seed_num, self.agent_wrapper, **kw)
        self.log_info(self.debug_info)
        
    def train(self):
        """train model on env"""
        pass
    
    def test(self, model_pth):
        """test model on env"""
        pass