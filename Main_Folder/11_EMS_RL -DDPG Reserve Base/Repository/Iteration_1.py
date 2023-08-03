# -*- coding: utf-8 -*-
"""
Created on Tue Apr 11 21:10:33 2023

@author: Karthikeyan
"""
#%%Loading packages
import numpy as np
import random
import gym
from gym import spaces
#%% Creating environment

class EnergyStorageSystem(gym.Env):
    
    def __init__(self):
        
        # Action space is between the charge and discharge limits (changes every step after updating SOC)
        self.action_space = spaces.Box(low=-Pdis_max, high=Pcha_max)
        
        # State (In theory only SOC changes price does not)
        self.observation_space = spaces.Box(low=np.array([0, 0]), high=np.array([np.inf, 1]), dtype=np.float32)
        
        # Start state (should be SOC)
        self.current_step = 0
        
        # Episodic length (How to optimize for)
        self.max_steps = len(prices)
        
        
        # What happens in each time step
    def step (self, action):
        True
        
        # Get this from the paper (arbitrage and maintaining SOC)
    def reward_function(profit, soc):
        if soc < soc_threshold:
            penalty = -10         
        else:
            penalty = 0
        return profit + penalty
    
    def reset(self):
        self.current_step = 0
        self.soc = soc_initial
        return self._next_observation()
    