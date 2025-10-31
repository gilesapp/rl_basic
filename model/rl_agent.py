import numpy as np
import os
import tqdm
from .bAgent import bAgent, bAgentWrapper
from .rl_model import *
from .util import plot_train


# wrappers
class AgentWrapper(bAgentWrapper):
    '''Standard agent wrapper for interacting with gym envs (e.g. classic controls)'''
    def __init__(self):
        super().__init__()
    
class ParkingAgentWrapper(bAgentWrapper):
    '''modify state with "obs" + "goal state" for parking-v0 env'''
    def __init__(self):
        super().__init__()
        
    def state_wrapper(self, state):
        if isinstance(state, dict):
            return np.concatenate((state['observation'], state['desired_goal']))
        else:
            return state
        
        
# agents
class OffPolicyAgent(bAgent):
    def __init__(self, env, n_agent=1, seed_num=2025, agent_wrapper=bAgentWrapper(), debug_info=False):
        super().__init__(env, n_agent, seed_num, agent_wrapper, debug_info)
    
    def train(self, n_episodes=1000, print_every=50, save_pth=None, plot=False):
        train_reward = []
        
        for i_episode in tqdm.trange(n_episodes):
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

        if plot:
            os.makedirs(self.fig_save_pth, exist_ok=True)
            plot_train(train_reward, f'{self.fig_save_pth}{self.env.spec.id}_{self.model_name}_eps{n_episodes}')
    
    def test(self, model_pth):
        self.model.load_model(model_pth)
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


class OnPolicyAgent(bAgent):
    def __init__(self, env, n_agent=1, seed_num=2025, agent_wrapper=bAgentWrapper(), debug_info=False):
        super().__init__(env, n_agent, seed_num, agent_wrapper, debug_info)
    
    def train(self, n_episodes=1000, print_every=50, save_pth=None, plot=False):
        train_reward = []
        
        for i_episode in tqdm.trange(n_episodes):
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

        if plot:
            os.makedirs(self.fig_save_pth, exist_ok=True)
            plot_train(train_reward, f'{self.fig_save_pth}{self.env.spec.id}_{self.model_name}_eps{n_episodes}')
    
    def test(self, model_pth):
        self.model.load_model(model_pth)
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
        
        
class DQNAgent(OffPolicyAgent):
    def __init__(self, env, n_agent=1, seed_num=2025, agent_wrapper=bAgentWrapper(), debug_info=False, **kw):
        super().__init__(env, n_agent, seed_num, agent_wrapper, debug_info)
        
        self.make_model(DQN, **kw)

class DDPGAgent(OffPolicyAgent):
    def __init__(self, env, n_agent=1, seed_num=2025, agent_wrapper=bAgentWrapper(), debug_info=False, **kw):
        super().__init__(env, n_agent, seed_num, agent_wrapper, debug_info)
        
        self.make_model(DDPG, **kw)

class SACAgent(OffPolicyAgent):
    def __init__(self, env, n_agent=1, seed_num=2025, agent_wrapper=bAgentWrapper(), debug_info=False, **kw):
        super().__init__(env, n_agent, seed_num, agent_wrapper, debug_info)
        
        self.make_model(SAC, **kw)

class PPOAgent(OnPolicyAgent):
    def __init__(self, env, n_agent=1, seed_num=2025, agent_wrapper=bAgentWrapper(), debug_info=False, **kw):
        super().__init__(env, n_agent, seed_num, agent_wrapper, debug_info)
        
        self.make_model(PPO, **kw)
