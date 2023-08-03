# -*- coding: utf-8 -*-
"""
Created on Sat Jun  3 14:41:18 2023

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

#%% Actor Network

class ActorNetwork(keras.Model):
    def __init__(self,fc1_dims=248,fc2_dims=248,n_actions=1,name='actor',chkpt_dir='tmp/ddpg'):
        super(ActorNetwork,self).__init__()
        self.fc1_dims=fc1_dims
        self.fc2_dims=fc2_dims
        self.n_actions=n_actions
        
        self.model_name=name
        self.checkpoint_dir=chkpt_dir
        self.checkpoint_file=os.path.join(self.checkpoint_dir,self.model_name+'_ddpg.h5')
        
        self.fc1=Dense(self.fc1_dims,activation='relu')
        self.fc2=Dense(self.fc2_dims,activation='relu')
        self.mu=Dense(self.n_actions,activation='linear')
        
    
    def call(self,state):
        prob=self.fc1=(state)
        prob=self.fc2(prob)
        mu=self.mu(prob)
        
        return mu