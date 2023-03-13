# -*- coding: utf-8 -*-
"""
Created on Sun Mar 12 01:24:20 2023

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
import pickle
import itertools
import matplotlib.dates as mdates

current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

from Essential_functions import load_data
#%% Read Data
df=load_data()
df=df[:8760]
#%% Train Test split
split=int(0.9*len(df))
train=df[:split]
test=df[split:]
#%% Calling decomposition model
mst=pickle.load(open(r'C:\Users\Karthikeyan\Desktop\Thesis\Model Database\MSTL1.sav','rb'))
#%% Visualizing the decomposition
plot,ax=plt.subplots()
ax.plot(mst.trend)

plot,bx=plt.subplots()
bx.plot(mst.seasonal['seasonal_24'].iloc[:168])
bx.xaxis.set_major_formatter(mdates.DateFormatter('%a'))

plot,cx=plt.subplots()
cx.plot(mst.seasonal['seasonal_168'].iloc[:168])
cx.xaxis.set_major_formatter(mdates.DateFormatter('%a'))

plot,dx=plt.subplots()
dx.plot(mst.seasonal['seasonal_8760'])

plot,ex=plt.subplots()
ex.plot(mst.resid)

plt.show()
#%%