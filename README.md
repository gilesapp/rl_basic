# RL Agent tool

## Introduction
This is an RL agent creation tool which provides a standard interface for implementing custom RL models and training/testing on gym envs 

e.g. qdn/ddpg/sac/ppo model -> agent -> agent.train/test -> gym env

## How to install
Download zip and run `pip install -r requirements.txt`

## Example
run `python example.py`

## Train and Test custom RL model
run `python train.py`

## How to create custom RL model and Train/Test on Env
1. Create RL model (example DQN, DDPG, SAC, PPO in `rl_model.py`)
    - Step I. create `custom_Net(nn.Module)` in `bNN.py`
    - Step II. create `custom_Model(bModel)` in `rl_model.py`
2. Create Agent Wrapper (example in `rl_agent.py`)
    - Step III. create `custom_AgentWrapper(bAgentWrapper)` in `rl_agent.py`
    - **Note**: `bAgentWrapper` **automatically maps env action bound to model action bound by default** 
3. Create Agent with RL model (example in `rl_agent.py`)
    - Step IV. create `custom_Agent(OffPolicyAgent)` in `rl_agent.py`
    - Step V. `custom_Agent` -> `self.make_model(custom_Model, **kw)` 
    - **Note**: supports `OffPolicyAgent(bAgent)` `OnPolicyAgent(bAgent)`
4. Train/Test on Env
    - Step VI. `env = gym.make()`
    - Step VII. `agent = custom_Agent(env)`
    - Step VIII. `agent.train()` `agent.test()`