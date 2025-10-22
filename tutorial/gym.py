# gym
import gymnasium as gym
from stable_baselines3 import DQN


def DQN_example():
    env = gym.make("CartPole-v1", render_mode="human")

    model = DQN("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=10000, log_interval=4)
    model.save("dqn_cartpole")

    del model # remove to demonstrate saving and loading

    model = DQN.load("dqn_cartpole")

    obs, info = env.reset()
    while True:
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            obs, info = env.reset()
        
        print(f"obs: {obs}, reward: {reward}, info: {info}, terminated: {terminated}, truncated: {truncated}")
        
        
def main():
    print('hello world')


if __name__ == "__main__":
    main()