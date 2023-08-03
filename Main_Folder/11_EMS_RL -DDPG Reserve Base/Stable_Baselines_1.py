# -*- coding: utf-8 -*-
"""
Created on Tue May  2 11:19:01 2023

@author: Karthikeyan
"""
#%% Loading packages
from Env_custom import EMSenv
import gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3 import DQN
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.vec_env import DummyVecEnv
#%%

# Create the environment
env = EMSenv
#%%
# Check the environment
check_env(env)

# Wrap the environment with the DummyVecEnv wrapper
env = DummyVecEnv([lambda: env])

# Create the DQN agent
model = DQN('MlpPolicy', env, learning_rate=1e-3, buffer_size=50000, exploration_fraction=0.1,
            exploration_final_eps=0.02, verbose=1)

# Train the agent for 10000 steps
model.learn(total_timesteps=10000)