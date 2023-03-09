# -*- coding: utf-8 -*-
"""
Created on Wed Mar  8 21:36:35 2023

@author: Karthikeyan
"""
#%% Loading package

import pandas as pd
import numpy as np 
import seaborn as sns
import matplotlib.pyplot as plt

import os
import sys
import path

current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

from Essential_functions import load_data
from EDA_functions import Fourier_Analysis

#%% Read data
df=load_data()
#%% Functions
def inverse_difference(last_ob, value):
 return value + last_ob

def inverse_difference_series(data,diff):
    diff.dropna(inplace=True)
    inverted = [inverse_difference(data[i], diff[i]) for i in range(len(diff))]
    return inverted
    
#%% Seasonal differencing
df['diff_24']=df['Load'].diff(24)
temp_inverted=inverse_difference_series(df['Load'], df['diff_24'])
df.dropna(inplace=True)
df['inverted']=temp_inverted

