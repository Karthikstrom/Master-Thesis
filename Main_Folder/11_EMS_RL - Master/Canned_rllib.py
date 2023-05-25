# -*- coding: utf-8 -*-
"""
Created on Fri Apr 28 12:51:01 2023

@author: Karthikeyan
"""


#%% Importing packages
import os
import sys
import path
import datetime
import numpy as np
import pandas as pd

#import gymnasium as gym
from ray import tune


from Env_custom import EMSenv

current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

from Essential_functions import load_wholedata

df=load_wholedata()
df=df[df.index.year==2019]


#%%
import gym
import numpy as np
from stable_baselines import DDPG
from stable_baselines.ddpg.policies import MlpPolicy
import tf_slim as slim
# create gym environment
env = EMSenv(df)

# create DDPG model
model = DDPG(MlpPolicy, env, verbose=1)

# train the model
model.learn(total_timesteps=10000)

# save the model
model.save("ddpg_ems")

# load the saved model
loaded_model = DDPG.load("ddpg_ems")

# evaluate the loaded model
obs = env.reset()
for i in range(24):
    action, _states = loaded_model.predict(obs)
    obs, reward, done, info = env.step(action)
    if done:
        print("Episode finished after {} timesteps".format(i+1))
        break

#%% Trial

test_env=EMSenv(df)

tune.run("PPO",
         config={"env":EMSenv,
                 "evaluation_interval":100,
                 "evaluation_num_episodes":1000
                },
         #checkpoint_freq=100,
         #local_dir="exp1"
    )



# from ray.rllib.algorithms.ddpg.ddpg import DDPGConfig

# #test_env=EMSenv(df)
# config = DDPGConfig().training(lr=0.01).resources(num_gpus=1)
# print(config.to_dict())  
# # Build a Trainer object from the config and run one training iteration.
# algo = config.build(env=EMSenv) 
# algo.train()  