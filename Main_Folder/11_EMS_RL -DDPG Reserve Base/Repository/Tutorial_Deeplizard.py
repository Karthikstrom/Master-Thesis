# -*- coding: utf-8 -*-
"""
Created on Wed Mar  1 08:19:50 2023

@author: Karthikeyan
"""

#%% Loading packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym
import random
import time
from IPython.display import clear_output
#%% Creating the environment
env = gym.make('FrozenLake-v1', render_mode='ansi')

""""
In OpenAI Gym, the render_mode parameter is used to specify how the 
environment should be rendered when it is run.

The render_mode parameter has several possible values:

'human': renders the environment on the screen using a graphical user
 interface (GUI), which is intended for human interaction.
'rgb_array': returns the rendered environment as a NumPy array of RGB values, 
which can be useful for debugging or machine learning applications.
'ansi': renders the environment as text in the terminal using ASCII 
characters.

"""
#%% Creating Q-Table / Definite state action space
action_space_size = env.action_space.n
state_space_size = env.observation_space.n

# Initializing Q table with zeros
q_table = np.zeros((state_space_size, action_space_size))
#%% Initializing W-Learning
num_episodes = 10000
max_steps_per_episode = 100

learning_rate = 0.1
discount_rate = 0.99

exploration_rate = 1
max_exploration_rate = 1
min_exploration_rate = 0.01
exploration_decay_rate = 0.001
#%% Algorithm training 
rewards_all_episodes = []

for episode in range(num_episodes):
    # initialize new episode params
    # reset the state of the env back to starting state 
    state = env.reset()[0]
    # done shows if the episode has reached the terminal state or not
    done = False
    # since we start with each episode with zero reward
    rewards_current_episode = 0
    
    
    # loop for each time step
    for step in range(max_steps_per_episode):
        # Exploration-exploitation trade-off
        # The function call random.uniform(0, 1) generates a random number 
        # between 0 (inclusive) and 1 (exclusive), meaning that the number 
        # can be 0 but cannot be 1
        
        exploration_rate_threshold = random.uniform(0, 1)
        # As time progresses exploration rate reduces so chances of exploitation skyrockets
        if exploration_rate_threshold >=exploration_rate: #(Exploitation - Best Q-value from the state)
            action = np.argmax(q_table[state,:]) 
        #argmax is a mathematical function that returns the index of the 
        #maximum element in an array or sequence. It is a commonly used 
        #function in machine learning and data analysis for finding the most 
        #likely or optimal choice among a set of options.
        
        else:
            action = env.action_space.sample() #(Exploration)
            
        new_state, reward, done, truncated, info = env.step(action)
        
        # Update Q-table for Q(s,a)
        # Q learning is recursive, i.e s depends on Q value of s' which in
        # turn depends on s''
        q_table[state, action] = q_table[state, action] * (1 - learning_rate) + learning_rate * (reward + discount_rate * np.max(q_table[new_state, :]))
    
        state = new_state
        rewards_current_episode += reward
   
        if done==True:
            break
    
    # Exponential decay of the exploration rate
    exploration_rate = min_exploration_rate + \
    (max_exploration_rate - min_exploration_rate) * np.exp(-exploration_decay_rate*episode)
    
    rewards_all_episodes.append(rewards_current_episode)
    
#%% After All episodes complete

rewards_per_thousand_episodes = np.split(np.array(rewards_all_episodes),num_episodes/1000)
count = 1000

print("********Average reward per thousand episodes********\n")
for r in rewards_per_thousand_episodes:
    print(count, ": ", str(sum(r/1000)))
    count += 1000
    
#%%Printing the updated Q-table
print("\n\n********Q-table********\n")
print(q_table)
    