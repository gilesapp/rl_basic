import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm
from .rl_model import *
from .util import linear_map, plot_train
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
    def __init__(self, model_name, env, n_agent=1, seed_num=2025, agent_wrapper=None):
        self.env = env
        self.agent_wrapper = bAgentWrapper() if not agent_wrapper else agent_wrapper
        self.n_agent = n_agent
        self.seed_num = seed_num
        
        self.update_sa() # init state size and action size
        
        self.model_dict = {'dqn': DQN, 'ddpg': DDPG}
        self.support_model = list(self.model_dict.keys())
        assert model_name in self.support_model, f"invalid model name [{model_name}], supported model: {self.support_model}"
        self.model = self.make_model(model_name, self.state_size, self.action_size, self.n_agent, self.seed_num,
                                     BUFFER_SIZE=1e5,
                                     BUFFER_THRESHOLD=500,
                                     BATCH_SIZE = 128,
                                     GAMMA = 0.99,
                                     SOFT_UPDATE_ITER = 10,
                                     LR = 1e-3,
                                     EPSILON = 0.01)
        
        self.isMap, self.model_abound, self.env_abound = self.cal_abound()
        
        self.model_name = model_name
        self.model_save_pth = 'ckpt/'
        self.fig_save_pth = 'fig/'
        
    def make_model(self, model_name, *args, **kw):
        return self.model_dict[model_name](*args, **kw)

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
    
    # train
    def train(self, n_episodes, print_every, save_pth=None, plot=False):
        train_reward = self.train_agent(n_episodes, print_every, save_pth)
        if plot:
            os.makedirs(self.fig_save_pth, exist_ok=True)
            plot_train(train_reward, f'{self.fig_save_pth}{self.env.spec.id}_{self.model_name}_eps{n_episodes}')
    
    # test
    def test(self, model_pth):
        self.model.load_model(model_pth)
        self.test_agent()
    
    def train_agent(self, n_episodes=1000, print_every=50, save_pth=None):
        train_reward = []
        
        for i_episode in tqdm.tqdm(range(n_episodes)):
            local_reward = 0
            done = False
            
            state, info = self.env.reset()
            state = self.agent_wrapper.state_wrapper(state)
            
            while not done:
                action = self.model.act(state)
                action = self.agent_wrapper.action_wrapper(action, self.model_abound, self.env_abound, self.isMap)
           
                next_state, reward, terminated, truncated, info = self.env.step(action)
            
                next_state = self.agent_wrapper.state_wrapper(next_state)
                done = terminated or truncated
            
                self.model.step([state], [action], [reward], [next_state], [done])
                
                local_reward += reward
                
                state = next_state
                
            if i_episode % print_every == 0:
                print(f"eps {i_episode}, reward {local_reward}") 

            train_reward.append(local_reward)
        
        if save_pth == None:
            save_pth = self.model_save_pth
        os.makedirs(save_pth, exist_ok=True)
        self.model.save_model(f"{save_pth}{self.env.spec.id}_{self.model_name}_eps{n_episodes}.pt")
     
        return train_reward
        
    def test_agent(self):
        local_reward = 0
        steps = 0
        done = False
        
        state, info = self.env.reset()
        state = self.agent_wrapper.state_wrapper(state)
        
        self.model.eval()

        while not done:
            action = self.model.act(state)
            action = self.agent_wrapper.action_wrapper(action, self.model_abound, self.env_abound, self.isMap)
         
            next_state, reward, terminated, truncated, info = self.env.step(action)
            next_state = self.agent_wrapper.state_wrapper(next_state)
            done = terminated or truncated
        
            local_reward += reward
            steps += 1
            
            state = next_state
            
        print(f"steps {steps}, reward {local_reward}")
        
               