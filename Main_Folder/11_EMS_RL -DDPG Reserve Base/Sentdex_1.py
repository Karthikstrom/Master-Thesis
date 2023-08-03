# -*- coding: utf-8 -*-
"""
Created on Tue May  2 18:44:01 2023

@author: Karthikeyan
"""
#%% Importing packages
from keras.models import Sequential
from keras.layers import Dense,Dropout,Flatten,Conv2D
from keras.callbacks import TensorBoard
from keras.optimizers import Adam
from collections import deque
import tensorflow as tf
import time
import numpy as np
from tqdm import tqdm
import os
import random
from Env_custom import EMSenv
import matplotlib.pyplot as plt
#%% Calling the custom environment
env=EMSenv()
#%% Hyperparameter
Replay_memory_size=50000
Episodes=200
Min_replay_memory=100
minibatch_size=64
Target_update_every=10
Discount_rate=0.2
MODEL_NAME="128x64"

# Exploration settings
epsilon = 1  # not a constant, going to be decayed
EPSILON_DECAY = 0.99975
MIN_EPSILON = 0.001

#Stats setting 
AGGREGATE_STATS_EVERY=50
SHOW_PREVIEW=False
#%% Custom TensorBoard class

class ModifiedTensorBoard(TensorBoard):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.step = 1
        self.writer = tf.summary.create_file_writer(self.log_dir)
        self._log_write_dir = self.log_dir
    
    def set_model(self, model):
        self.model = model
    
        self._train_dir = os.path.join(self._log_write_dir, 'train')
        self._train_step = self.model._train_counter
    
        self._val_dir = os.path.join(self._log_write_dir, 'validation')
        self._val_step = self.model._test_counter
    
        self._should_write_train_graph = False
    
    def on_epoch_end(self, epoch, logs=None):
        self.update_stats(**logs)
    
    def on_batch_end(self, batch, logs=None):
        pass
    
    def on_train_end(self, _):
        pass
    
    def update_stats(self, **stats):
        with self.writer.as_default():
            for key, value in stats.items():
                tf.summary.scalar(key, value, step = self.step)
                self.writer.flush()
#%% Creating the agent
class DQNAgent:
    def __init__(self):
        #main model # gets trained every step
        self.model=self.create_model()
        
        #Target model # .predict against every step
        self.target_model=self.create_model()
        self.target_model.set_weights(self.model.get_weights())
        
        #Replay 
        self.replay_memory=deque(maxlen=Replay_memory_size)
        
        #Tensorboard- track internally when we are ready to update target model
        self.tensorboard=ModifiedTensorBoard(log_dir=f"logs/{MODEL_NAME}-{int(time.time())}")
    
        self.target_update_counter=0
        
        
    def create_model(self):
        model=Sequential()
        model.add(Dense(128, input_shape=env.observation_space.shape))
        model.add(Dropout(0.2))
        model.add(Dense(64,activation='relu'))
        model.add(Dense(env.action_space.n,activation="linear"))
        
        model.compile(loss='mse',optimizer=Adam(learning_rate=0.001))
        
        return model
    
    def update_replay_memory(self,transition):
        self.replay_memory.append(transition)
        
    def get_qs(self,state):
        #model.predict always returns a list so [0]
        return self.model.predict(np.array(state).reshape(-1,*state.shape))[0]
    
    def train(self,terminal_state,step):
        if len(self.replay_memory)<Min_replay_memory:
            return 
        
        # Get a minibatch of random samples from memory replay table
        minibatch = random.sample(self.replay_memory, minibatch_size)

        # Get current states from minibatch, then query NN model for Q values
        current_states = np.array([transition[0] for transition in minibatch])
        current_qs_list = self.model.predict(current_states)

        # Get future states from minibatch, then query NN model for Q values
        # When using target network, query it, otherwise main network should be queried
        new_current_states = np.array([transition[3] for transition in minibatch])
        future_qs_list = self.target_model.predict(new_current_states)
        
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
        self.model.fit(np.array(X), np.array(y), batch_size=minibatch_size, verbose=3, shuffle=False, callbacks=[self.tensorboard] if terminal_state else None)
        
        # Update target network counter every episode
        if terminal_state:
            self.target_update_counter += 1

        # If counter reaches set value, update target network with weights of main network
        if self.target_update_counter > Target_update_every:
            self.target_model.set_weights(self.model.get_weights())
            self.target_update_counter = 0

#%% Main loop
agent=DQNAgent()
ep_rewards=[]


for episode in tqdm(range(1,Episodes+1),ascii=True,unit="episode"):
    agent.tensorboard.step=episode
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
    #print(ep_rewards)
    if not episode % AGGREGATE_STATS_EVERY or episode == 1:
        average_reward = sum(ep_rewards[-AGGREGATE_STATS_EVERY:])/len(ep_rewards[-AGGREGATE_STATS_EVERY:])
        min_reward = min(ep_rewards[-AGGREGATE_STATS_EVERY:])
        max_reward = max(ep_rewards[-AGGREGATE_STATS_EVERY:])
        agent.tensorboard.update_stats(reward_avg=average_reward, reward_min=min_reward, reward_max=max_reward, epsilon=epsilon)
    

        # Decay epsilon
    if epsilon > MIN_EPSILON:
        epsilon *= EPSILON_DECAY
        epsilon = max(MIN_EPSILON, epsilon)
        
#%% Testing loop

#Goals
#Check action - only from the target net
#Check reward
#For a week?- so training would be for 

