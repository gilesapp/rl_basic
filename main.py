################################################################################################
# Module: main func
# 
# Description: 
#   example - how to get start
#   1. create env
#   2. create agent
#   3. train/test
# 
# Version: 1.0
#
# Change History:
# Version   | Author      | Date         | Changes
#---------------------------------------------------------
# v1.0      | gilesapp    | 2025-10-28   | envs ["classic control", "highway"], models ["dqn", "ddpg"]
#################################################################################################
import gymnasium as gym
import highway_env
from model.rl_agent import *


class Run():
    def __init__(self, model_name, env_name, agent_wrapper=None):
        self.model_name = model_name
        self.env_name = env_name
        self.agent_wrapper = agent_wrapper
    
    # it's recommended by gym offcial doc to create one instance for each mode as render_mode is Read-Only
    # env_train = render_mode=None
    # env_test  = render_mode="human"
    # env_video = render_mode="rgb_array"    
    # So for each env mode, we need to create a new agent
    def train(self, n_episodes=100, print_every=50):
        env = gym.make(self.env_name)
        train_agent = BaseAgent(self.model_name, env, agent_wrapper=self.agent_wrapper)
        train_agent.train(n_episodes=n_episodes, print_every=print_every, plot=True)  # default model save path: ckpt/[env name]_[model name]_[eps num]

    def test(self, load_model_pth):
        env = gym.make(self.env_name, render_mode="human")
        test_agent = BaseAgent(self.model_name, env, agent_wrapper=self.agent_wrapper)
        test_agent.test(model_pth=load_model_pth)
        
    
def main():
    # init
    run = Run(model_name="ddpg",  # "dqn", "ddpg", "sac"
              env_name="parking-v0", # "CartPole-v1", "Pendulum-v1", "parking-v0"
              agent_wrapper=ParkingAgentWrapper())
    
    # run
    run.train(n_episodes=1000)
    # run.test("ckpt/parking-v0_ddpg_eps100.pt")


if __name__ == "__main__":
    main()