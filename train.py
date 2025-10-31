import gymnasium as gym
import highway_env
from model.rl_agent import *

# it's recommended by gym offcial doc to create one instance for each mode as render_mode is Read-Only
# env_train = render_mode=None
# env_test  = render_mode="human"
# env_video = render_mode="rgb_array"   

def train_parking():
    env = gym.make("parking-v0")
    
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
    agent.train(n_episodes=1200, print_every=50, plot=True)
    # agent.test("ckpt/parking-v0_ddpg_eps1000.pt")
    
    env.close()
    
def train_pendulum():
    env = gym.make("Pendulum-v1")
    
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
                     TARGET_ENT=-2.0,
                     DETERM=False)
    
    # run
    agent.train(n_episodes=1200, print_every=50, plot=True)
    # agent.test("ckpt/Pendulum-v1_ddpg_eps1000.pt")
    
    env.close()
    
def train_cartpole():
    env = gym.make("CartPole-v1")
    
    agent_wrapper=ParkingAgentWrapper()
    agent = DQNAgent(env, debug_info=True, 
                     BUFFER_SIZE=1e5,
                     BUFFER_THRESHOLD=500,
                     BATCH_SIZE=128,
                     GAMMA=0.99,
                     SOFT_UPDATE_ITER=10,
                     LR=1e-3,
                     EPSILON=0.01)
    
    # run
    agent.train(n_episodes=150, print_every=50, plot=True)
    # agent.test("ckpt/CartPole-v1_dqn_eps150.pt")
    
    env.close()
    
def main():
    train_pendulum()

if __name__ == "__main__":
    main()