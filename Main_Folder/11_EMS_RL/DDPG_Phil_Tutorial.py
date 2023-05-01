# -*- coding: utf-8 -*-
"""
Created on Fri Apr 28 23:00:59 2023

@author: Karthikeyan
"""

#%% Loading packages
from torch import nn
import torch
import gym
from collections import deque
import itertools
import numpy as np
import random
#%% Importing data and Environment
import os
import sys
import path
import datetime
import numpy as np
import pandas as pd

from Env_custom import EMSenv

current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

from Essential_functions import load_wholedata

df=load_wholedata()
df=df[df.index.year==2019]
df = df.round(decimals=2)
#%% Hyperparamters
#Discount factor
Gamma=1
#Number of samples from replay buffer to update 
Batch_size=32
#Number of transitions to store before over-writing all transitions?
Buffer_size=50000
#How many transitions in replay buffer before start computing gradients
Min_replay_size=1000
#Epsilon greedy strategy
epsilon_start=1
epsilon_end=0.02
epsilon_decay=10000
#Target parameters equal to online parameters
Target_update_freq=1000
#%% Calling the environment

class Network(nn.Module):
    def __init__(self,env):
        super().__init__()
        
        in_features=int(np.prod(env.observation_space.shape))
        
        self.net =nn.Sequential(
            nn.Linear(in_features,64),
            nn.Tanh(),
            nn.Linear(64, env.action_space.n)
                                )
        
    def forward(self,x):
        return self.net(x)
    
    def act(self,obs):
        #converting it into tensor
        obs_t=torch.as_tensor(obs,dtype=torch.float32)
        #Finding all possible(all actions) the q-values - unsqueeze (all i/p in pytorch expects a batch dimension so individual)
        q_values= self(obs_t.unsqueeze(0))
        #Finding the action with the highest q- value
        max_q_index=torch.argmax(q_values,dim=1)[0]
        #Change it into an integer
        action= max_q_index.detach().item()
        return action


#%%
test_env=EMSenv(df)

replay_buffer=deque(maxlen=Buffer_size)
reward_buffer=deque([0.0],maxlen=100)
episode_reward=0.0

online_net=Network(test_env)
target_net=Network(test_env)

#Create optimizer
optimizer=torch.optim.Adam(online_net.parameters(),lr=5e-4)

target_net.load_state_dict(online_net.state_dict())



#Initialize Replay Buffer
obs=test_env.reset()

for _ in range(Min_replay_size):
    action=test_env.action_space.sample()
    new_obs,rew,done,_=test_env.step(action)
    transition= (obs,action,rew,done,new_obs)
    replay_buffer.append(transition)
    obs=new_obs
    
    if done:
        obs=test_env.reset()

#%% Main Training Loop
obs = test_env.reset()

for step in range(30000):
    epsilon=np.interp(step,[0,epsilon_decay],[epsilon_start,epsilon_end])
    
    rnd_sample=random.random()
    
    if rnd_sample<= epsilon:
        action=test_env.action_space.sample()
    else:
        action=online_net.act(obs)
        #print(action)
    
    new_obs,rew,done,_=test_env.step(action)
    transition= (obs,action,rew,done,new_obs)
    replay_buffer.append(transition)
    obs=new_obs
    
    episode_reward +=rew
    
    if done:
        obs=test_env.reset()
        reward_buffer.append(episode_reward)
        episode_reward=0.0
    
    #Start Gradient step
    #Sampling batch size from replay buffer
    transitions= random.sample(replay_buffer,Batch_size)
    
    #Transition tuple to list
    obses=np.asarray([t[0] for t in transitions])
    actions=np.asarray([t[1] for t in transitions])
    rews=np.asarray([t[2] for t in transitions])
    dones=np.asarray([t[3] for t in transitions])
    new_obses=np.asarray([t[4] for t in transitions])
    
    #List to a pytorch tensor
    obses_t=torch.as_tensor(obses,dtype=torch.float32)
    actions_t=torch.as_tensor(actions,dtype=torch.int64).unsqueeze(-1)
    rews_t=torch.as_tensor(rews,dtype=torch.float32).unsqueeze(-1)
    dones_t=torch.as_tensor(dones,dtype=torch.float32).unsqueeze(-1)
    new_obses_t=torch.as_tensor(new_obses,dtype=torch.float32)
    
    #Compute Target
    target_q_values=target_net(new_obses_t)
    max_target_q_values=target_q_values.max(dim=1,keepdim=True)[0]
    
    targets=rews_t+ Gamma * (1-dones_t) * max_target_q_values
    
    # Compute Loss
    
    q_values= online_net(obses_t)
    
    action_q_values=torch.gather(input=q_values,dim=1,index=actions_t)
    
    loss=nn.functional.smooth_l1_loss(action_q_values,targets)
    #print(loss)
    # Gradient 
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    # Update Target Network
    if step % Target_update_freq ==0:
        target_net.load_state_dict(online_net.state_dict())
    
    #Logging
    
    if step % 1000==0:
        print()
        print('Step',step)
        print('Avg Rew', np.mean(reward_buffer))
    
    
#%% 
    
# create an instance of the environment using testing data
#test_env = EMSenv(test_df)

# initialize variables for tracking performance
num_episodes = 1
episode_rewards = []
avg_reward = 0.0
action_v=[]

# simulate the environment for a fixed number of episodes using the online network
for i in range(num_episodes):
    obs = test_env.reset()
    episode_reward = 0.0
    done = False
    while done == False:
        action = online_net.act(obs)
        obs, reward, done, info = test_env.step(action)
        episode_reward += reward
        action_v.append(action)
        print(obs)
    episode_rewards.append(episode_reward)

# calculate average reward over all episodes
#avg_reward = sum(episode_rewards) / num_episodes

# print the average reward
#print("Average Reward:", avg_reward)