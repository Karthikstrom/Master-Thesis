# -*- coding: utf-8 -*-
"""
Created on Sun Apr 23 23:11:08 2023

@author: Karthikeyan
"""
#%% Checking data

import os
import sys
import path
import datetime
import numpy as np
import pandas as pd

current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

from Essential_functions import load_wholedata

df=load_wholedata()
df=df[df.index.year==2019]
#%% Loading packages
import gym
from gym import spaces
from numpy.random import default_rng


import random

#%%
class EMSenv(gym.Env):
    
    #Setting the rendering mode as human readable
    metadata = {'render.modes': ['human']}
    
    def __init__(self,df):
        
        """
        Must define self.observation_space and self.action_space
        """
        # Define action space: bounds,space type, shape
        # Bounds : ESS charging and discharging rate limitations (d,c)
        # Space Types : Box (float type)
        # Shape : array of rank zero (single element) but rllib cannot handle 
        # scalar actions, so turn it into a numpy array with shape (1,)
        
        self.max_discharging=-2.8
        self.max_charging=2.8
        self.action_space=spaces.Box(low=np.array([self.max_discharging]),high=np.array([self.max_charging]))
        
        
        # Define observation space: bounds,space type, shape
        # Bounds : 
        # Space Types :
        # Shape :
            
        
        #Number of dimensions in the observation space
        self.obs_dim=4
        self.df=df
        
        #Setting the lower limit for the observations
        self.min_load=0
        self.min_pv=0
        self.min_price=0
        
        #Setting the higher limit for the observations
        #Arbitary values
        self.max_load=(self.df['Load'].max())*1.2
        self.max_pv=(self.df['PV'].max())*1.2
        self.max_price=(self.df['RTP'].max())*1.2
        
        #Runtime observation
        self.min_battery_cap=1
        self.max_battery_cap=6
        
        
        #Setting the low and high of observation space
        self.obs_low=np.array([self.min_load,self.min_price,self.min_pv,self.min_battery_cap])
        self.obs_high=np.array([self.max_load,self.max_price,self.max_pv,self.max_battery_cap])
        
        self.observation_space=spaces.Box(low=self.obs_low,high=self.obs_high)
            
        """
        ----------------------------------------------------------------------
        """
        #Initial Index
        self.intial_index=0
        
        #Random number generator that will be used throughout the environment
        self.rng=default_rng()
        
        # All instance variables are defined in _init__() method
        self.current_obs=None
        
        #Done counter
        self.done=False
        
        #Time step length
        self.num_of_time_steps=24
        
        #Setting time counter to start
        self.hour_num=0
        
        
    def reset(self):
        
        """
        Returns: the observation of the intial state
        Reset the environment to the initial state so that a new episode (independent of previous ones) may start
        
        """
        
        #Generating a random number to get a random day in the training year
        self.random_day=random.randint(1, 365)
        
        #Index from 23.00 from the previous day to the end of next day
        #So that decisions are taken from 00:00 and not 01.00
        
        self.start_idx=self.df[self.df.index.dayofyear==self.random_day].index.min()+pd.Timedelta(hours=-1)
        self.end_idx=self.df[self.df.index.dayofyear==self.random_day].index.max()
        
        #Extracting the data for the random training day (24 values) 
        self.load=self.df['Load'][self.start_idx:self.end_idx]
        self.pv=self.df['PV'][self.start_idx:self.end_idx]
        self.price=self.df['RTP'][self.start_idx:self.end_idx]
        
        #Initializing random battery capacity
        battery_cap=self.rng.uniform()*self.max_battery_cap
                
        #Reset should pass the initial observation
        self.current_obs=np.array([self.load[0],self.price[0],self.pv[0],battery_cap])
        
        
        #Reset hour counter
        self.hour_num=0
        
        return self.current_obs

    def step(self,action):
        """
        Given the current obs and action it should 
        Returns[ The next observation, the reward, done and optionally additional info]
        """

        # Action looks like np.array([20.0]). We need to convert that to float 20.0 for easier calculation
        battery_action=action[0]
        
        #Compute Next observation
        #Take time step as input
        # Not sure how it takes decision in the first time step
        next_load=self.load[self.hour_num+1]
        next_pv=self.pv[self.hour_num+1]
        next_price=self.price[self.hour_num+1]
        
        #For the battery capacity, how do i get the previous battery cap?
        #Not sure if this is right but should find a way to use the
        #Current observation
        next_battery_cap=self.current_obs[3]+battery_action
        
        #Next observation array
        next_obs=[next_load,next_price,next_pv,next_battery_cap]
        
        #Compute reward
        
        #transported to the grid (gt)
        grid_t=next_load-next_pv+battery_action
        
        #Simple reward
        reward=-(next_price*grid_t)
        
        # Compute done
        self.hour_num=self.hour_num+1
        
        
        done=False
        
        if self.hour_num == self.num_of_time_steps:
            done=True
        
        
        #Update the current observation
        self.current_obs=next_obs
        
        
        return self.current_obs,reward,done,{}


    def render(self, mode="human"):
        """
        Returns:None
        
        Displays the graphical window
        Not required for EMS
        """
        
        pass
    
    def close(self):
        """
        Returns: None
        This method is optional, used to clean up all resources (threads, graphical windows)
        """
        
        pass
    
    def seed(self, seed=None):
        """
        Returns: List of seeds
        This method is optional. Used to set seeds for the environment's random number generator for
        deterministic behaviour
        """
        
        return

 #%% Testing the environment

test_env=EMSenv(df)

for i in range(5):
    obs=test_env.reset()
    done=False
    while done==False:
        action=np.array([0])
        obs,r,done,_=test_env.step(action)
        print(test_env.random_day)
         