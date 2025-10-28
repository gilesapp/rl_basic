from .bAgent import bAgent

class RLAgent(bAgent):
    '''Standard agent for interacting with gym envs (e.g. classic controls)'''
    def __init__(self, model_name, env, n_agent=1, seed_num=2025, agent_wrapper=None):
        super().__init__(model_name, env, n_agent=1, seed_num=2025, agent_wrapper=None)