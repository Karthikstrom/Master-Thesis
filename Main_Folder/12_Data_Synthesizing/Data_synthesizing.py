# -*- coding: utf-8 -*-
"""
Created on Wed Mar  8 19:26:40 2023

@author: Karthikeyan
"""
#%% loading packages
import os
import sys
import path
import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose

current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

from Essential_functions import load_wholedata
#%% Read data
df= load_wholedata()

#%% Functions

#mean and std of noise
m=0
sigma=0.1

def apply_growth_rate(data,growth_rate,n):
    
    #n-number of years into the future
    
    start_index=data.index.min()
    
    for i in range(n):
        temp=[]
        temp=data[-8760:]*(1+(growth_rate/100))
        #adding noise
        noise =np.random.normal(m, sigma, len(temp))
        temp= temp + noise
        data=pd.concat([data,temp],ignore_index=True)
        
    dates=pd.date_range(start_index,periods=len(data),freq='H')
    data.index=dates
    
    return data

def repeat_data(data,n):
    
    start_index=data.index.min()
    
    #Average of the past years
    past_avg=data.groupby(data.index.strftime('%m-%d %H')).mean()
    
    #removing the leap year part
    past_avg=past_avg[:-24]
    
    for i in range(n):
        
        temp=past_avg
        #adding noise
        noise =np.random.normal(m, sigma, len(temp))
        temp=temp + noise
        data=pd.concat([data,temp],ignore_index=True)
    
    dates=pd.date_range(start_index,periods=len(data),freq='H')
    data.index=dates
    
    return data

#%% Results

# Fabricated DataFrame

df_fab=pd.DataFrame()
df_fab['Load']=apply_growth_rate(df['Load'],2,13)
df_fab['RTP']=apply_growth_rate(df['RTP'],2,13)


#%% Visualizing

fig,ax=plt.subplots()
ax.plot(df_fab['Load'].resample('M').mean())


fig,bx=plt.subplots()
bx.plot(df_fab['RTP'].resample('M').mean())

plt.show()