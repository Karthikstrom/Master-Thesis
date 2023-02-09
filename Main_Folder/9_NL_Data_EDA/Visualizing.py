# -*- coding: utf-8 -*-
"""
Created on Wed Feb  8 14:29:51 2023

@author: Karthikeyan
"""
#%% Loading packages

import os
import sys
import path
import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf

current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

from Essential_functions import load_data,load_data2

#%% Read data
df=load_data()
df1=load_data2()
#%% Whole data
fig,ax=plt.subplots()
#ax.plot(df[:500])
ax.plot(df[:500],color='r')
plt.show()
#%% Resampling
dfrs=df1.resample('M').mean()
fig,bx=plt.subplots()
bx.plot(dfrs)
plt.show()
#%% Distribuion
sns.histplot(df)