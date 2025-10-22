from model.ddpg import DDPG_Agent
import gymnasium as gym
from collections import deque
import tqdm


def DDPG_train(model, env, save_pth, n_agents=1, n_episodes=100, n_step=100, print_every=10):
    best_reward = -99999
    
    for i_episode in tqdm.tqdm(range(n_episodes)):
        state, info = env.reset()
        local_reward = 0
        
        for t in range(n_step):
            action = model.act(state)
         
            # next_state, reward, terminated, truncated, info = env.step(action.argmax().item())
            # print(f"state {state}, action {action}")
            next_state, reward, terminated, truncated, info = env.step(action)
           
            model.step(state, action, reward, next_state, terminated or truncated)
            
            local_reward += reward
            state = next_state
            
            if terminated or truncated:
                state, info = env.reset()
                # local_reward = 0
                # print(next_state, reward, terminated, truncated, info)
            
        if i_episode % print_every == 0:
            print(f"eps {i_episode}, reward {local_reward}")

        if local_reward > best_reward:
            best_reward = local_reward
            model.save_model(save_pth)    


def DDPG_test(env, model):
    local_reward = 0
    steps = 0
    state, info = env.reset()
    
    while True:
        action = model.act(state)
        
        next_state, reward, terminated, truncated, info = env.step(action)
        
        local_reward += reward
        steps += 1
        
        # if terminated:
        #     state, info = env.reset()
        #     print(f"steps {steps}, reward {local_reward}")
            
        if terminated or truncated:
            print(f"steps {steps}, reward {local_reward}")
            break
      
        
def main():
    state_size = 3
    action_size = 1
    n_agent = 1
    seed_num = 2025
    
    n_episodes = 250
    n_step=200
    print_every = 10
  
    model = DDPG_Agent(state_size, action_size, n_agent, seed_num)
    
    # train
    # env = gym.make("Pendulum-v1")
    # DDPG_train(model, env, "ckpt/pendulum_ddpg.pt", n_episodes=n_episodes, n_step=n_step, print_every=print_every)
    
    # test
    env = gym.make("Pendulum-v1", render_mode="human")
    model.load_model("ckpt/pendulum_ddpg.pt")
    model.eval()
    DDPG_test(env, model)
    
    env.close()

if __name__ == "__main__":
    main()