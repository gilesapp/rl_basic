# Run `pip install "gymnasium[classic-control]"` for this example.
import gymnasium as gym
import highway_env

# Create our training environment - a cart with a pole that needs balancing
env = gym.make("CartPole-v1", render_mode="human")

# Reset environment to start a new episode
observation, info = env.reset()
# observation: what the agent can "see" - cart position, velocity, pole angle, etc.
# info: extra debugging information (usually not needed for basic learning)

print(f"env type {type(env)}")
# print(f"Starting observation: {env.observation_space.low}, {type(env.observation_space)}")
print(f"action space {env.action_space.n}, {type(env.action_space)}, env name {env.spec.id}")
# Example output: [ 0.01234567 -0.00987654  0.02345678  0.01456789]
# [cart_position, cart_velocity, pole_angle, pole_angular_velocity]

# [Pendulum] action space Box(-2.0, 2.0, (1,), float32), <class 'gymnasium.spaces.box.Box'>
# [CartPole] action space Discrete(2), <class 'gymnasium.spaces.discrete.Discrete'>
# [parking] action space Box(-1.0, 1.0, (2,), float32), <class 'gymnasium.spaces.box.Box'>

episode_over = False
total_reward = 0

while not episode_over:
    # Choose an action: 0 = push cart left, 1 = push cart right
    action = env.action_space.sample()  # Random action for now - real agents will be smarter!
    print(f"action {action}")
    # Take the action and see what happens
    observation, reward, terminated, truncated, info = env.step(action)
    print(observation, reward, terminated, truncated, info)
    # reward: +1 for each step the pole stays upright
    # terminated: True if pole falls too far (agent failed)
    # truncated: True if we hit the time limit (500 steps)

    total_reward += reward
    episode_over = terminated or truncated

print(f"Episode finished! Total reward: {total_reward}")
env.close()