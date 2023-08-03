# -*- coding: utf-8 -*-
"""
Created on Thu May  4 16:56:14 2023

@author: Karthikeyan
"""

#%% Importing packages
from keras.models import Sequential
from keras.layers import Dense,Dropout,Flatten,Conv2D
from tensorflow.keras.losses import Huber
from keras.optimizers import Adam
from collections import deque
import tensorflow as tf
import time
import numpy as np
import pandas as pd
from tqdm import tqdm
import os
import random
from Env_custom import EMSenv
import matplotlib.pyplot as plt
from keras.models import load_model

import sys
import path
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

from Essential_functions import load_wholedata,real_load
#%% Calling the custom environment
env=EMSenv()
#%% Hyperparameter
Replay_memory_size=30000
Episodes=700
Min_replay_memory=4000
minibatch_size=32
Target_update_every=100
Discount_rate=0.1
model_filename = 'real_load.h5'


# Exploration settings
epsilon = 1  # not a constant, going to be decayed
EPSILON_DECAY = 0.995 #0.997-700 episodic learning
MIN_EPSILON = 0.1
epsilon_dec=[] 
#%% Creating the agent
class DQNAgent:
    def __init__(self):
        #main model # gets trained every step
        self.model=self.create_model()
        
        #Target model # .predict against every step
        self.target_model=self.create_model()
        self.target_model.set_weights(self.model.get_weights())
        
        #Replay memory intialization
        self.replay_memory=deque(maxlen=Replay_memory_size)
        
        self.target_update_counter=0
        
        #Main Network loss counter
        self.nn_loss=[]
        
    def create_model(self):
        model=Sequential()
        model.add(Dense(64, input_shape=env.observation_space.shape,activation='relu'))
        model.add(Dropout(0.2))
        model.add(Dense(128,activation='relu'))
        model.add(Dropout(0.2))
        model.add(Dense(128,activation='relu'))
        model.add(Dense(env.action_space.n,activation="linear"))
        
        model.compile(loss='mse',optimizer=Adam(learning_rate=0.0001))
        
        return model
    
    # Return the history object and check
    
    def update_replay_memory(self,transition):
        self.replay_memory.append(transition)
        
    def get_qs(self,state):
        #model.predict always returns a list so [0]
        prediction=self.target_model.predict(np.array(state).reshape(-1,*state.shape),verbose=0)[0]
        return prediction
    
    def train(self,terminal_state,step):
        
        #To set the minimum buffer length before actually training
        #This is coupled with the main training loop
        if len(self.replay_memory)<Min_replay_memory:
            return 
        
        # Get a minibatch of random samples from memory replay table
        minibatch = random.sample(self.replay_memory, minibatch_size)

        # Get current states from minibatch, then query NN model for Q 
        
        
        # Extracts only the state
        current_states = np.array([transition[0] for transition in minibatch])
        current_qs_list = self.model.predict(current_states,verbose=0)

        # Get future states from minibatch, then query NN model for Q values
        # When using target network, query it, otherwise main network should be queried
        new_current_states = np.array([transition[3] for transition in minibatch])
        future_qs_list = self.target_model.predict(new_current_states,verbose=0)
        
        
        # Empty arrays for fitting in NN
        X=[]
        y=[]
        
        #Now we enumerate our batches
        
        for index,(current_state, action, reward, new_current_state, done) in enumerate(minibatch):
            
            # If not a terminal state, get new q from future states, otherwise set it to 0
            # almost like with Q Learning, but we use just part of equation here
            
            if not done:
                max_future_q = np.max(future_qs_list[index])
                new_q = reward + Discount_rate * max_future_q
            else:
                new_q = reward
                
            # Update Q value for given state
            current_qs = current_qs_list[index]
            current_qs[action] = new_q

            # And append to our training data
            X.append(current_state)
            y.append(current_qs)
            
        # Fit on all samples as one batch, log only on terminal state
        hist=self.model.fit(np.array(X), np.array(y), batch_size=minibatch_size, verbose=0, shuffle=False)
        self.nn_loss.append(hist.history['loss'])
        
        # Update target network counter every episode
        if terminal_state:
            self.target_update_counter += 1

        # If counter reaches set value, update target network with weights of main network
        if self.target_update_counter > Target_update_every:
            self.target_model.set_weights(self.model.get_weights())
            self.target_update_counter = 0

#%% Main training loop
agent=DQNAgent()
ep_rewards=[]
ep=0

for episode in tqdm(range(1,Episodes+1),ascii=True,unit="episode"):
    episode_reward=0
    step=1 #?
    current_state=env.reset()
    done=False
    
    
    while not done:
        # returns index
        if np.random.random() > epsilon:
            action=np.argmax(agent.get_qs(np.array(current_state)))
        else:
            action = np.random.randint(0,env.action_space.n)
        
        
        new_state,reward,done,_=env.step(action)
        
        episode_reward+=reward 
        
        agent.update_replay_memory((current_state,action,reward,new_state,done))
        agent.train(done,step)
        
        current_state=new_state
        step+=1
        
    ep_rewards.append(episode_reward)
    #print(episode_reward)
        # Decay epsilon
    if epsilon > MIN_EPSILON:
        epsilon *= EPSILON_DECAY
        epsilon = max(MIN_EPSILON, epsilon)
        epsilon_dec.append(epsilon)
        
        
    ep+=1
        
    if ep%100==0:
        plt.plot(ep_rewards)
        plt.show()
        plt.plot(agent.nn_loss)
        plt.show()
        agent.target_model.save(model_filename)
        
        
#%% Epsilon checker

# for i in range(10000):
#     if epsilon > MIN_EPSILON:
#         epsilon *= EPSILON_DECAY
#         epsilon = max(MIN_EPSILON, epsilon)
#         epsilon_dec.append(epsilon)

# plt.plot(epsilon_dec)
#%% Save Model
agent.target_model.save(model_filename)
#agent.save_weights(weights_filename)
#%% Load Model
loaded_model = load_model('real_load.h5')


def get_qs_test(loaded_model,state):
    #model.predict always returns a list so [0]
    return loaded_model.predict(np.array(state).reshape(-1,*state.shape),verbose=0)[0]
#%% Main testing loop
test_episode=1
all_actions=[]
for i in range(test_episode):
        current_state=env.reset()
        done=False
        while not done:
            action=np.argmax(get_qs_test(loaded_model,np.array(current_state)))
            new_state,reward,done,_=env.step(action)
            current_state=new_state
            all_actions.append(action)
            
plt.plot(env.map_action(all_actions))
#%% Testing weekly loop
test_episode=2
all_actions=[]

#Initial Batttery capacity
temp_bat_cap=3
batt_cap=[]

for i in range(test_episode):
        current_state=env.train_reset()
        current_state[3]=temp_bat_cap
        done=False
        while not done:
            
            batt_cap.append(current_state[3])
            action=np.argmax(get_qs_test(loaded_model,np.array(current_state)))
            new_state,reward,done,_=env.step(action)
            current_state=new_state
            all_actions.append(action)
            print(env.map_action(action))
        
        #passing on the battery capacity to the next day    
        temp_bat_cap=current_state[3]

plt.plot(env.map_action(all_actions))

#%% Test data

# df_test=real_load()
# df_test=df_test['2019-06-01':'2019-06-07']
# df_test


# fig


