# -*- coding: utf-8 -*-
"""
Created on Mon Apr 17 16:51:20 2023

@author: Karthikeyan
"""

import os
import sys
import path
import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
import itertools
from KPIs import EMS_KPI
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

from Essential_functions import load_wholedata
import matplotlib.dates as mdates

#%% Read Data
df=load_wholedata()
df=df[df.index.year==2019]
df['PV']=7*df['PV']
#%% Baseline- RTP
df['COE_Base_RTP']=df['Load']*df['RTP']
df['COE_Base_TOU']=df['Load']*df['TOU']
print("Baseline-RTP",df['COE_Base_RTP'].sum(),"Euros")
print("Baseline-TOU",df['COE_Base_TOU'].sum(),"Euros")
#%% Only PV

#Not taking dump into account

df[['ps','pp']]=0.0000

for index, row in df.iterrows():
    #isolating values from each row
    pv=row['PV']
    load=row['Load']
    
    if pv>=load:
        pp=0
        ps=pv-load
        
    else:
        ps=0
        pp=load-pv
        
    df.at[index,'pp']=pp
    df.at[index,'ps']=ps

df['COE_PV_RTP']=(df['pp']-df['ps'])*df['RTP']
df['COE_PV_TOU']=(df['pp']-df['ps'])*df['TOU']

print("PV_Only-RTP",df['COE_PV_RTP'].sum(),"Euros")
print("PV_Only-TOU",df['COE_PV_TOU'].sum(),"Euros")

#%% KPI Check

EMS_KPI(df['pp'],df['ps'],df['TOU'])




