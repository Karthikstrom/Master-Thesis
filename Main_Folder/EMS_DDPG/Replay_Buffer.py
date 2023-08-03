# -*- coding: utf-8 -*-
"""
Created on Sat Jun  3 14:10:22 2023

@author: Karthikeyan
"""
#%% Import packages
import numpy as np
import os 
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.layers import Dense

from keras.optimizers import Adam
#%% Buffer class

class ReplayBuffer:
    
    # Since continuous action spaces, number of actions is a misnomer, its the components
    def __init__(self,max_size,input_shape,n_actions):
        self.mem_size=max_size
        self.mem_cntre=0
        self.state_memory=np.zeros((self.mem_size,*input_shape))
        self.new_state_memory=np.zeros((self.mem_size,*input_shape))
        self.action_memory= np.zeros((self.mem_size,n_actions))
        self.reward_memory= np.zeros(self.mem_size)
        self.terminal_memory=np.zeros(self.mem_size,dtype=np.bool)
        
    def store_transition(self,state,action,reward,new_state,done):
        index= self.mem_cntre % self.mem_size
        
        self.state_memory[index]=state
        self.new_state_memory[index]=new_state
        self.action_memory[index]=action
        self.reward_memory[index]=reward
        self.terminal_memory[index]=done
        
        self.mem_cntre+=1
     
    #Not useful to learn from zeros in the whole replay buffer - check with deque memory as well
    
    #Retriving values in batches of 34
    def sample_buffer(self,batch_size):
        
        max_mem= min(self.mem_cntre,self.mem_size)
        
        #replace avoids sampling the same values
        batch= np.random.choice(max_mem,batch_size,replace=False)
        
        states=self.state_memory[batch]
        states_=self.new_state_memory[batch]
        actions=self.action_memory[batch]
        rewards=self.reward_memory[batch]
        dones=self.terminal_memory[batch]

        return states,actions,rewards,states_,dones

