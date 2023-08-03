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


#from.sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import minmax_scale

current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

from Essential_functions import real_load

df=real_load()
df=df[df.index.year==2019]

#Adding predicted values
df['3h_price']=df['RTP'].shift(-3)
df['6h_price']=df['RTP'].shift(-6)
df['12h_price']=df['RTP'].shift(-12)
df['24h_price']=df['RTP'].shift(-24)

df['3h_load']=df['Load'].shift(-3)
df['6h_load']=df['Load'].shift(-6)
df['12h_load']=df['Load'].shift(-12)
df['24h_load']=df['Load'].shift(-24)

df['Hour']=df.index.hour

df.dropna(inplace=True)
#%% Normalizing historic observations
df[:] = minmax_scale(df)
df = df.round(3)
#%% Loading packages
import gym
from gym import spaces
from numpy.random import default_rng
import random
#%% Environment creation


class EMSenv(gym.Env):
    
    #Setting the rendering mode as human readable
    metadata = {'render.modes': ['human']}
    
    def __init__(self):
        
        """
        Must define self.observation_space and self.action_space
        """
        #Defining Observation space           
        
        #Number of dimensions in the observation space
        self.obs_dim=13
        self.df=df
        
        #Historic observation limits
        self.min_load=0
        self.min_pv=0
        self.min_price=0
        self.min_hour=0
        
        self.min_3h_price=0
        self.min_6h_price=0
        self.min_12h_price=0
        self.min_24h_price=0
        
        self.min_3h_load=0
        self.min_6h_load=0
        self.min_12h_load=0
        self.min_24h_load=0
        
        self.max_load=(self.df['Load'].max())*1.2
        self.max_pv=(self.df['PV'].max())*1.2
        self.max_price=(self.df['RTP'].max())*1.2
        self.max_hour=23
        
        self.max_3h_price=(self.df['RTP'].max())*1.2
        self.max_6h_price=(self.df['RTP'].max())*1.2
        self.max_12h_price=(self.df['RTP'].max())*1.2
        self.max_24h_price=(self.df['RTP'].max())*1.2
        
        
        self.max_3h_load=(self.df['Load'].max())*1.2
        self.max_6h_load=(self.df['Load'].max())*1.2
        self.max_12h_load=(self.df['Load'].max())*1.2
        self.max_24h_load=(self.df['Load'].max())*1.2
        
        #Runtime observation limits
        self.soc_min=0.2
        self.soc_max=0.8
        
        # self.pb_in_min=0
        # self.pb_in_max=2.8
        
        # self.pb_out_min=0
        # self.pb_out_max=2.8
        
        #Setting the low and high of observation space
        self.obs_low=np.array([self.min_load,self.min_price,self.min_pv,self.min_hour,
                               self.min_3h_price,self.min_6h_price,self.min_12h_price,self.min_24h_price,
                               self.min_3h_load,self.min_6h_load,self.min_12h_load,self.min_24h_load,
                               self.soc_min])
        
        
        self.obs_high=np.array([self.max_load,self.max_price,self.max_pv,self.max_hour,
                                self.max_3h_price,self.max_6h_price,self.max_12h_price,self.max_24h_price,
                                self.max_3h_load,self.max_6h_load,self.max_12h_load,self.max_24h_load,
                                self.soc_max])
        
        self.observation_space=spaces.Box(low=self.obs_low,high=self.obs_high) 
        
        
        # Defining action space
        self.act_dim=6
        
        self.a1_min=0
        self.a1_max=(self.df['PV'].max())*1.2
        
        self.a2_min=-2.8
        self.a2_max=0
        
        self.a3_min=0
        self.a3_max=(self.df['Load'].max())*1.2
        
        self.a4_min=0
        self.a4_max=(self.df['PV'].max())*1.2
        
        self.a5_min=0
        self.a5_max=(self.df['PV'].max())*1.2
        
        self.a6_min=-2.8
        self.a6_max=2.8
        
        # self.a7_min=0
        # self.a7_max=2.8
        
        self.act_low=np.array([self.a1_min,self.a2_min,self.a3_min,self.a4_min,self.a5_min,self.a6_min])
        self.act_high=np.array([self.a1_max,self.a2_max,self.a3_max,self.a4_max,self.a5_max,self.a6_max])
        
        
        self.action_space=spaces.Box(low=self.act_low,high=self.act_high)
        
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
        
        
        #Set the start date number
        self.test_day_counter=31
        
        #Battery SOC specs
        self.eff_imp=0.9
        self.eff_exp=0.9
        self.battery_capacity=6.6
        self.pb_min=2.8
        self.pb_max=2.8
        
        
            
        
    def reset(self):
        
        """
        Returns: the observation of the intial state
        Reset the environment to the initial state so that a new episode (independent of previous ones) may start
        
        """
        
        #Generating a random number to get a random day in the training year
        self.random_day=random.randint(2,30)
        
        #Index from 23.00 from the previous day to the end of next day
        #So that decisions are taken from 00:00 and not 01.00
        
        self.start_idx=self.df[self.df.index.dayofyear==self.random_day].index.min()+pd.Timedelta(hours=-1)
        self.end_idx=self.df[self.df.index.dayofyear==self.random_day].index.max()
        
        #Extracting the data for the random training day (24 values) 
        self.load=self.df['Load'][self.start_idx:self.end_idx]
        self.pv=self.df['PV'][self.start_idx:self.end_idx]
        self.price=self.df['RTP'][self.start_idx:self.end_idx]
        self.hour_feature=self.df['Hour'][self.start_idx:self.end_idx]
        
        self.price_3h=self.df['3h_price'][self.start_idx:self.end_idx]
        self.price_6h=self.df['6h_price'][self.start_idx:self.end_idx]
        self.price_12h=self.df['12h_price'][self.start_idx:self.end_idx]
        self.price_24h=self.df['24h_price'][self.start_idx:self.end_idx]
        
        
        self.load_3h=self.df['3h_load'][self.start_idx:self.end_idx]
        self.load_6h=self.df['6h_load'][self.start_idx:self.end_idx]
        self.load_12h=self.df['12h_load'][self.start_idx:self.end_idx]
        self.load_24h=self.df['24h_load'][self.start_idx:self.end_idx]
        
        #Initializing run time observations
        soc = round(self.rng.uniform(self.soc_min, self.soc_max), 2)
        # pb_in= round(self.rng.uniform(self.pb_in_min, self.pb_in_max), 2)
        # pb_out= round(self.rng.uniform(self.pb_out_min, self.pb_out_max), 2)
        #Reset should pass the initial observation
        self.current_obs=np.array([self.load[0],self.price[0],self.pv[0],self.hour_feature[0],
                                  self.price_3h[0],self.price_6h[0],self.price_12h[0],self.price_24h[0],
                                  self.load_3h[0],self.price_6h[0],self.price_12h[0],self.price_24h[0],
                                  soc])
        
        
        
        #Reset hour counter
        self.hour_num=0
        
        
        return self.current_obs


    def step(self,a1,a2,a3,a4,a5,a6):
        """
        Given the current obs and action it should 
        Returns[ The next observation, the reward, done and optionally additional info]
        """

        # Action looks like np.array([20.0]). We need to convert that to float 20.0 for easier calculation
        #battery_action=action[0]
        s2l=a1
        b2l=a2
        g2l=a3
        s2b=a4
        s2g=a5
        battery_action=a6
        #Next time step
        self.hour_num=self.hour_num+1
        
        
        #Constraints
        s_total=a1+a2+a4
        l_total=a1+a2+a3
        
        grid_imp=a3+a6
        grid_exp=a5
    
        #Grid import
        
        if (self.current_obs[12] <= self.soc_min) or (self.current_obs[12] >= self.soc_max):
            # Calculate the penalty based on the deviation from the SOC limits
            reward_1 = -2
        else:
            reward_1= 0
                
        reward_3=-10*abs(self.current_obs[2]-s_total)
        reward_4=-10*abs(self.current_obs[0]-l_total)
            
        
        reward_5=-3*(grid_imp-grid_exp)*self.current_obs[1]
        
        reward=reward_1+reward_3+reward_4+reward_5
    
        
        #Compute Next observation
        #Take time step as input
        # Not sure how it takes decision in the first time step
        next_load=self.load[self.hour_num]
        next_pv=self.pv[self.hour_num]
        next_price=self.price[self.hour_num]
        next_hour=self.hour_feature[self.hour_num]
        
        next_3h_price=self.price_3h[self.hour_num]
        next_6h_price=self.price_6h[self.hour_num]
        next_12h_price=self.price_12h[self.hour_num]
        next_24h_price=self.price_24h[self.hour_num]
        
        next_3h_load=self.load_3h[self.hour_num]
        next_6h_load=self.load_6h[self.hour_num]
        next_12h_load=self.load_12h[self.hour_num]
        next_24h_load=self.load_24h[self.hour_num]
        
        #For the battery capacity, how do i get the previous battery cap?
        
        bat_action_cum=a4+a2+a6

        next_soc=self.SOC(self.current_obs[12],bat_action_cum)
        next_soc=np.round(next_soc, 3)
        #next_soc = np.clip(next_soc,self.soc_min, self.soc_max) 
        # next_pb_in=self.pb_in_func(next_soc)
        # next_pb_in=np.round(next_pb_in, 2)
        # next_pb_out=self.pb_out_func(next_soc)
        
        #Next observation array
        next_obs=[next_load,next_price,next_pv,next_hour,next_3h_price,
                  next_6h_price,next_12h_price,next_24h_price,
                  next_3h_load,next_6h_load,next_12h_load,next_24h_load,
                  next_soc]
        
        
         
        done=False
        
        if self.hour_num == self.num_of_time_steps:
            done=True
        
        
        #Update the current observation
        self.current_obs=next_obs
        
        
        return self.current_obs,reward,done,{}


    def pb_in_func(self,soc):
        pb_in_temp=min(self.pb_min,(self.battery_capacity)*(self.soc_max-soc))
        #to take the charge efficiency into consideration
        pb_in_temp=pb_in_temp/self.eff_imp
        return pb_in_temp

    def pb_out_func(self,soc):
        pb_out_temp=min(self.pb_max,(self.battery_capacity)*(soc-self.soc_min))
        #to take the discharge efficiency into consideration
        pb_out_temp=self.eff_exp*pb_out_temp
                
        return pb_out_temp
    
    def SOC(self,soc_last,bat_action):
        soc_temp=soc_last + ((bat_action)/(self.battery_capacity))
        return soc_temp
    
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

test_env=EMSenv()

for i in range(1):
    obs=test_env.reset()
    done=False
    while done==False:
        # Sampling each action
        action = test_env.action_space.sample()
        # Accessing individual actions
        a1 = action[0]
        a2 = action[1]
        a3 = action[2]
        a4 = action[3]
        a5 = action[4]
        a6 = action[5]
        obs,r,done,_=test_env.step(a1,a2,a3,a4,a5,a5)
        print(obs)
# #%%

# num_actions = test_env.action_space.n
# for action in range(num_actions):
#     print(test_env.map_action(action))