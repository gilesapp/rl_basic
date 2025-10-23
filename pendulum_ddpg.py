from model.ddpg import DDPG_Agent
import gymnasium as gym
import tqdm
import model.util as ut


def rl_train(model, env, save_pth, n_agents=1, n_episodes=100, n_step=100, print_every=10):
    train_reward = []
    for i_episode in tqdm.tqdm(range(n_episodes)):
        state, info = env.reset()
        local_reward = 0
        
        for t in range(n_step):
            action = model.act(state) * 2

            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
           
            model.step(state, action, reward, next_state, done)
            
            local_reward += reward
            state = next_state
            
            if done:
                break
            
        if i_episode % print_every == 0:
            print(f"eps {i_episode}, reward {local_reward}") 

        train_reward.append(local_reward)
        
    # model.save_model(save_pth)
    return train_reward
        
        
def rl_test(env, model):
    local_reward = 0
    steps = 0
    state, info = env.reset()
    model.eval()
    
    while True:
        action = model.act(state) * 2
        
        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        
        local_reward += reward
        steps += 1
        
        state = next_state
            
        if done:
            print(f"steps {steps}, reward {local_reward}")
            break
      
        
def main():
    state_size = 3
    action_size = 1
    n_agent = 1
    seed_num = 2025
    
    n_episodes = 1200
    n_step=200
    print_every = 50
  
    model = DDPG_Agent(state_size, action_size, n_agent, seed_num)
    
    # train
    env = gym.make("Pendulum-v1")
    train_reward = rl_train(model, env, "ckpt/pendulum_ddpg.pt", n_episodes=n_episodes, n_step=n_step, print_every=print_every)
    ut.plot_train(train_reward, 'Pendulum_DDPG')
    
    # test
    # env = gym.make("Pendulum-v1", render_mode="human")
    # model.load_model("ckpt/pendulum_ddpg.pt")
    # rl_test(env, model)
    
    env.close()

if __name__ == "__main__":
    main()