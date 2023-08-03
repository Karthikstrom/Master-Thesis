# -*- coding: utf-8 -*-
"""
Created on Sat Jun  3 14:40:34 2023

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

#%% Critic Network
class CriticNetwork(keras.Model):
    def __init__(self,fc1_dims=248,fc2_dims=248,name='critic',chkpt_dir='tmp/ddpg'):
        super(CriticNetwork,self).__init__()
        self.fc1_dims=fc1_dims
        self.fc2_dims=fc2_dims
        
        self.model_name=name
        self.checkpoint_dir=chkpt_dir
        self.checkpoint_file=os.path.join(self.checkpoint_dir,self.model_name+'_ddpg.h5')
        
        self.fc1=Dense(self.fc1_dims,activation='relu')
        self.fc2=Dense(self.fc2_dims,activation='relu')
        self.q=Dense(1, activation=None)
    
    def call(self,state,action):
        
        #concatanating state and action
        action_value=self.fc1(tf.concat([state,action],axis=1))
        action_value=self.fc2(action_value)
        
        q=self.q(action_value)
        
        return q