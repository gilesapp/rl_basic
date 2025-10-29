from .bAgent import bAgent, bAgentWrapper
import numpy as np

# agents
class BaseAgent(bAgent):
    '''Standard agent for interacting with gym envs (e.g. classic controls)'''
    def __init__(self, model_name, env, n_agent=1, seed_num=2025, agent_wrapper=None):
        super().__init__(model_name, env, n_agent=1, seed_num=2025, agent_wrapper=None)

# wrappers
class BaseAgentWrapper(bAgentWrapper):
    '''Standard agent wrapper for interacting with gym envs (e.g. classic controls)'''
    def __init__(self):
        super().__init__()
    
class ParkingAgentWrapper(bAgentWrapper):
    '''modify state with "obs" + "goal state" for parking-v0 env'''
    def __init__(self):
        super().__init__()
        
    def state_wrapper(self, state):
        if isinstance(state, dict):
            return np.concatenate(state['observation'], state['desired_goal'])
        else:
            return state