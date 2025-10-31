import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm
from .rl_model import *
from .util import linear_map
import numpy as np
import os  


class bAgentWrapper(): 
    def state_wrapper(self, state):
        if isinstance(state, dict):
            return state['observation']
        else:
            return state

    def action_wrapper(self, action, model_abound, env_abound, isMap=False):
        if isMap:
            new_a = []
            for a_idx, a in enumerate(action):
                new_a.append(linear_map(a, model_abound[a_idx][0], model_abound[a_idx][1], env_abound[a_idx][0], env_abound[a_idx][1]))
            return new_a
        else:
            return action 
        
    
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
        print(f"{self.n_agent} {self.model.model_name} agent(s) init success")
        print(f"env: {self.env.spec.id}, state size {self.state_size}, action size {self.action_size}")
        if debug:
            print(f"--debug-- env state space {self.env.observation_space}")
            print(f"--debug-- env action space {self.env.action_space}")
          
    def make_model(self, model, **kw):
        self.update_sa() # init state size and action size
        self.model = model(self.state_size, self.action_size, self.seed_num, **kw)
        self.isMap, self.model_abound, self.env_abound = self.cal_abound()  # init action bound according to env
        self.log_info(self.debug_info)

    def update_sa(self):
        state, _ = self.env.reset()
        state = self.agent_wrapper.state_wrapper(state)
        self.state_size = len(state)
        if isinstance(self.env.action_space, gym.spaces.discrete.Discrete):
            self.action_size = self.env.action_space.n
        elif isinstance(self.env.action_space, gym.spaces.box.Box):
            self.action_size = self.env.action_space.shape[0]
            
    def cal_abound(self):
        isMap = False
        model_abound = self.model.get_action_space()
        env_abound = []
        if isinstance(self.env.action_space, gym.spaces.box.Box):
            env_a_low = self.env.action_space.low
            env_a_high = self.env.action_space.high
            
            for i in range(self.action_size):
                if isinstance(env_a_low, np.ndarray):
                    env_a_low_i = env_a_low[i]
                    env_a_high_i = env_a_high[i]
                else:
                    env_a_low_i = env_a_low
                    env_a_high_i = env_a_high
        
                if not (env_a_low_i == model_abound[i][0] and env_a_high_i == model_abound[i][1]):
                    env_abound.append([env_a_low_i, env_a_high_i])
                    isMap = True
             
        return isMap, model_abound, env_abound
    
    def train(self):
        """train model on env"""
        pass
    
    def test(self, model_pth):
        """test model on env"""
        pass