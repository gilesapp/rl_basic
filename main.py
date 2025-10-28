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
from model.rl_agent import RLAgent


def main():
    # create env (recommend)
    # env_train = render_mode=None
    # env_test  = render_mode="human"
    # env_video = render_mode="rgb_array"
    run_mode = "test"
    env_name = "Pendulum-v1" # 'Pendulum-v1', 'CartPole-v1'
    model_name = "ddpg"
    load_model_pth = "ckpt/Pendulum-v1_ddpg_eps1200.pt"
    
    # it's recommended by offcial doc to create one instance for each mode as render mode is Read-Only
    if run_mode == "train":
        env = gym.make(env_name, render_mode=None)
    if run_mode == "test":
        env = gym.make(env_name, render_mode="human")
    
    # create agent
    rl_agent = RLAgent(model_name, env)
    
    # train/test agent
    if run_mode == "train":
        rl_agent.train(n_episodes=1200, print_every=50, save_pth=None, plot=True)  # default model save path: ckpt/[env name]_[model name]_[eps num]
    if run_mode == "test":
        rl_agent.test(model_pth=load_model_pth)
    
    env.close()


if __name__ == "__main__":
    main()