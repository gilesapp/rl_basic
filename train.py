import gymnasium as gym
import highway_env
from model.rl_agent import *

# it's recommended by gym offcial doc to create one instance for each mode as render_mode is Read-Only
# env_train = render_mode=None
# env_test  = render_mode="human"
# env_video = render_mode="rgb_array"   
def run_parking(mode="train", n_eps=1000, ckpt_pth=""):
    env = gym.make("parking-v0", render_mode="human" if mode == "test" else None)
    
    agent_wrapper=ParkingAgentWrapper()
    agent = DDPGAgent(env, agent_wrapper=agent_wrapper, debug_info=True, 
                     BUFFER_SIZE=1e5,
                     BUFFER_THRESHOLD=500,
                     BATCH_SIZE=128,
                     GAMMA=0.99,
                     SOFT_UPDATE_ITER=10,
                     LR=1e-3,
                     EPSILON=0.01)
    
    # run
    if mode == "train":
        agent.train(n_episodes=n_eps, print_every=50, plot=True)
    if mode == 'test':
        agent.test(ckpt_pth)
    
    env.close()
    
def run_pendulum(mode="train", n_eps=1000, ckpt_pth=""):
    env = gym.make("Pendulum-v1", render_mode="human" if mode == "test" else None)
    
    # agent = DDPGAgent(env, debug_info=True, 
    #                  BUFFER_SIZE=1e5,
    #                  BUFFER_THRESHOLD=500,
    #                  BATCH_SIZE=128,
    #                  GAMMA=0.99,
    #                  SOFT_UPDATE_ITER=10,
    #                  LR=1e-3,
    #                  EPSILON=0.01)

    agent = SACAgent(env, debug_info=True, 
                     BUFFER_SIZE=1e5,
                     BUFFER_THRESHOLD=500,
                     BATCH_SIZE=128,
                     GAMMA=0.99,
                     SOFT_UPDATE_ITER=10,
                     LR=3e-4,
                     EPSILON=0.01,
                     TAU=0.005, 
                     ALPHA=0.2,
                     AUTO_ALPHA=True, 
                     TARGET_ENT=-1.0,
                     DETERM=False)
    
    # run
    if mode == "train":
        agent.train(n_episodes=n_eps, print_every=50, plot=True)
    if mode == 'test':
        agent.test(ckpt_pth)
    
    env.close()
    
def run_cartpole(mode="train", n_eps=1000, ckpt_pth=""):
    env = gym.make("CartPole-v1", render_mode="human" if mode == "test" else None)
    
    # agent = DQNAgent(env, debug_info=True, 
    #                  BUFFER_SIZE=1e5,
    #                  BUFFER_THRESHOLD=500,
    #                  BATCH_SIZE=128,
    #                  GAMMA=0.99,
    #                  SOFT_UPDATE_ITER=10,
    #                  LR=1e-3,
    #                  EPSILON=0.01)
    
    agent = SACAgent(env, debug_info=True, 
                     BUFFER_SIZE=1e5,
                     BUFFER_THRESHOLD=500,
                     BATCH_SIZE=128,
                     GAMMA=0.99,
                     SOFT_UPDATE_ITER=10,
                     LR=3e-4,
                     EPSILON=0.01,
                     TAU=0.005, 
                     ALPHA=0.2,
                     AUTO_ALPHA=True, 
                     TARGET_ENT=-1.0,
                     DETERM=True)
    
    # run
    if mode == "train":
        agent.train(n_episodes=n_eps, print_every=50, plot=True)
    if mode == 'test':
        agent.test(ckpt_pth)
    
    env.close()
    
def main():
    # run_cartpole("train", 150, "ckpt/CartPole-v1_dqn_eps150.pt")
    run_pendulum("train", 1200, "ckpt/Pendulum-v1_sac_eps1200.pt")
    # run_parking("test", 2000, "ckpt/parking-v0_ddpg_eps2000.pt")

if __name__ == "__main__":
    main()