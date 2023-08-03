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

from Essential_functions import load_wholedata,real_load

df=real_load()
df=df['2019-05-01':'2019-06-07']
df['Hour']=df.index.hour

#Using absolute forecast as the predicted values
df['3h_price']=df['RTP'].shift(-3)
df['6h_price']=df['RTP'].shift(-6)
df['12h_price']=df['RTP'].shift(-12)
df['24h_price']=df['RTP'].shift(-24)

df['3h_load']=df['Load'].shift(-3)
df['6h_load']=df['Load'].shift(-6)
df['12h_load']=df['Load'].shift(-12)
df['24h_load']=df['Load'].shift(-24)

#%% Normalizing historic observations
#df[:] = minmax_scale(df)
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
        # Define action space: bounds,space type, shape
        # Bounds : ESS charging and discharging rate limitations (d,c)
        # Space Types : Box (float type)
        # Shape : array of rank zero (single element) but rllib cannot handle 
        # scalar actions, so turn it into a numpy array with shape (1,)
        
        self.max_discharging=-2.8
        self.max_charging=2.8
        #self.action_space=spaces.Box(low=np.array([self.max_discharging]),high=np.array([self.max_charging]))
        
        #Discretized Action space 
        self.discretized_actions = np.arange(self.max_discharging,self.max_charging,0.1)
        self.discretized_actions = np.round(self.discretized_actions,decimals=2)
        self.n_actions=len(self.discretized_actions)
        self.action_space = gym.spaces.Discrete(self.n_actions)
        # Define observation space: bounds,space type, shape
        # Bounds :
        # Space Types :
        # Shape :
            
        
        #Number of dimensions in the observation space
        self.obs_dim=14
        self.df=df
        
        #Setting the lower limit for the observations
        self.min_load=0
        self.min_pv=0
        self.min_price=0
        self.min_hour=0
        self.soc_min=0
        
        self.min_3h_price=0
        self.min_6h_price=0
        self.min_12h_price=0
        self.min_24h_price=0
        
        self.min_3h_load=0
        self.min_6h_load=0
        self.min_12h_load=0
        self.min_24h_load=0
        
        #Setting the higher limit for the observations
        #Arbitary values
        self.max_load=(self.df['Load'].max())*1.2
        self.max_pv=(self.df['PV'].max())*1.2
        self.max_price=(self.df['RTP'].max())*1.2
        self.max_hour=23
        self.soc_max=1
        
        self.max_3h_price=(self.df['RTP'].max())*1.2
        self.max_6h_price=(self.df['RTP'].max())*1.2
        self.max_12h_price=(self.df['RTP'].max())*1.2
        self.max_24h_price=(self.df['RTP'].max())*1.2
        
        
        self.max_3h_load=(self.df['Load'].max())*1.2
        self.max_6h_load=(self.df['Load'].max())*1.2
        self.max_12h_load=(self.df['Load'].max())*1.2
        self.max_24h_load=(self.df['Load'].max())*1.2
        
        
        #Runtime observation
        self.battery_capacity=6.6
        self.min_battery_cap=0
        self.max_battery_cap=6.6
        self.eff_imp=1
        self.eff_exp=1
        self.pb_min=2.8
        self.pb_max=2.8
        
        #Setting the low and high of observation space
        self.obs_low=np.array([self.min_load,self.min_price,self.min_pv,self.min_battery_cap,self.min_hour,self.soc_min,
                               self.min_3h_price,self.min_6h_price,self.min_12h_price,self.min_24h_price,
                               self.min_3h_load,self.min_6h_load,self.min_12h_load,self.min_24h_load])
        self.obs_high=np.array([self.max_load,self.max_price,self.max_pv,self.max_battery_cap,self.max_hour,self.soc_max,
                                self.max_3h_price,self.max_6h_price,self.max_12h_price,self.max_24h_price,
                                self.max_3h_load,self.max_6h_load,self.max_12h_load,self.max_24h_load])
        
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
        self.num_of_time_steps=23
        
        #Setting time counter to start
        self.hour_num=0
        
        
        #Set the start date number
        self.test_day_counter=130
        
    def map_action(self,action):
        
            return self.discretized_actions[action]
        
    def reset(self):
        
        """
        Returns: the observation of the intial state
        Reset the environment to the initial state so that a new episode (independent of previous ones) may start
        
        """
        
        #Generating a random number to get a random day in the training year
        self.random_day=random.randint(122,129)
        
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
        
        #Initializing random battery capacity
        battery_cap = round(self.rng.uniform(self.min_battery_cap, self.max_battery_cap), 2)
        soc = round(self.rng.uniform(self.soc_min, self.soc_max), 2)
        #Reset should pass the initial observation
        self.current_obs=np.array([self.load[0],self.price[0],self.pv[0],battery_cap,self.hour_feature[0],soc,
                                   self.price_3h[0],self.price_6h[0],self.price_12h[0],self.price_24h[0],
                                   self.load_3h[0],self.price_6h[0],self.price_12h[0],self.price_24h[0]])
        
        
        #Reset hour counter
        self.hour_num=0
        
        return self.current_obs



    def train_reset(self):
        
        """
        Returns: the observation of the intial state
        Reset the environment to the initial state so that a new episode (independent of previous ones) may start
        
        """
        
        #Index from 23.00 from the previous day to the end of next day
        #So that decisions are taken from 00:00 and not 01.00
        
        self.start_idx=self.df[self.df.index.dayofyear==self.test_day_counter].index.min()+pd.Timedelta(hours=-1)
        self.end_idx=self.df[self.df.index.dayofyear==self.test_day_counter].index.max()
        
        #Extracting the data for the random training day (24 values) 
        self.load=self.df['Load'][self.start_idx:self.end_idx]
        self.pv=self.df['PV'][self.start_idx:self.end_idx]
        self.price=self.df['RTP'][self.start_idx:self.end_idx]
        self.hour_feature=self.df['Hour'][self.start_idx:self.end_idx]
        
        #Initializing random battery capacity
        battery_cap = round(self.rng.uniform(self.min_battery_cap, self.max_battery_cap), 2)
        #Reset should pass the initial observation
        self.current_obs=np.array([self.load[0],self.price[0],self.pv[0],battery_cap,self.hour_feature[0]])
        
        
        #Incrementing test_day_counter to the next day until no of episodes needed
        self.test_day_counter+=1
        
        
        #Reset hour counter
        self.hour_num=0
        
        return self.current_obs

    def step(self,action):
        """
        Given the current obs and action it should 
        Returns[ The next observation, the reward, done and optionally additional info]
        """

        # Action looks like np.array([20.0]). We need to convert that to float 20.0 for easier calculation
        #battery_action=action[0]
        battery_action=self.map_action(action)
        
        if battery_action>0:
            grid_imp=battery_action
            load_exp=0
        elif battery_action<0:
            grid_imp=0
            load_exp=abs(battery_action)
        else:
            grid_imp=0
            load_exp=0
        
        # Electricity purchased
        grid_purchase=self.current_obs[0]-load_exp
        grid_purchase_total=grid_purchase + grid_imp
        
        
        #To keep SOC within limits
        if (self.current_obs[0]<=0.2):
            reward_1=-abs(0.2-self.current_obs[0])
        else:
            reward_1=0
            
        if (self.current_obs[0]>=0.8):
            reward_2=-abs(self.current_obs[0]-0.8)
        else:
            reward_2=0
        
        #Increase financial benefit/ reduce the net electricity cost
        reward_3=-1*grid_purchase_total*self.current_obs[1]
        
        if (grid_purchase<0):
            reward_4=grid_purchase
        else:
            reward_4=0
        
        
        reward=10*reward_1+20*reward_2+5*reward_3+20*reward_4
        
        
        r=[reward_1,reward_2,reward_3,reward_4]
        
        
        
        done=False
        
        if self.hour_num == self.num_of_time_steps:
            done=True
        
        
        
        
        
        
        #Next time step
        self.hour_num=self.hour_num+1
        
        #Compute Next observation
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
        
        
        
        #computing SOC using the funstion
        next_soc=self.SOC(self.current_obs[5],battery_action)
        
        
        #Rounding to make simpler calculations
        next_soc=np.round(next_soc,3)
        next_pb_in=self.pb_in_func(next_soc)
        next_pb_out=self.pb_out_func(next_soc)
        
        next_pb_in=np.round(next_pb_in, 4)
        next_pb_out=np.round(next_pb_out, 4)
        
        next_battery_cap=self.current_obs[3]+battery_action
        next_battery_cap=np.round(next_battery_cap,decimals=2)
        #Next observation array
        next_obs=[next_load,next_price,next_pv,next_battery_cap,next_hour,next_soc,
                  next_3h_price,next_6h_price,next_12h_price,next_24h_price,
                  next_3h_load,next_6h_load,next_12h_load,next_24h_load]
        
        #Update the current observation
        self.current_obs=next_obs
        
        
        return self.current_obs,reward,done,r


    def soc(battery_capacity):
        return 
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

 #%% Testing the environment

test_env=EMSenv()

for i in range(10):
    obs=test_env.reset()
    done=False
    c=0
    while done==False:
        action=test_env.action_space.sample()
        #print(test_env.map_action(action))
        obs,r,done,_=test_env.step(action)
        print(obs[5])
        #print(r)
        #print(c)
        #c=c+1
        
        
#%%

# num_actions = test_env.action_space.n
# for action in range(num_actions):
#     print(test_env.map_action(action))