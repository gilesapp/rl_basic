################################################################################################
# Description: 
#   example - how to get start
#   1. create env
#   2. create agent, [Optional] create wrapper for state and action
#   3. train/test
# 
# Version: 1.1
#
# Change History:
# Version   | Author      | Date         | Changes
#---------------------------------------------------------
# v1.0      | gilesapp    | 2025-10-28   | envs ["classic control", "highway"], models ["dqn", "ddpg"]
# v1.1      | gilesapp    | 2025-10-31   | models ["sac", "ppo"]
#################################################################################################
import gymnasium as gym
import highway_env
from model.rl_agent import *
 
 
def main():
    env = gym.make("CartPole-v1", render_mode="human")
    
    agent = DQNAgent(env)
    
    # run
    # agent.train(n_episodes=150, plot=True) # default model save path: ckpt/[env name]_[model name]_[eps num]
    agent.test("ckpt/CartPole-v1_dqn_eps150.pt")
    
    env.close()


if __name__ == "__main__":
    main()