# -*- coding: utf-8 -*-
"""
Created on Mon Jan  2 19:06:21 2023

@author: Karthikeyan

Persistent method (Assumption that future value would be same as the present)
"""
#%% Loading packages

import pandas as pd
import numpy as np
import path
import sys
import os

import matplotlib.pyplot as plt
import seaborn as sns
sns.set_context('notebook')
sns.set_style("whitegrid")

current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

from Essential_functions import real_load,data_split,metrics,load_wholedata

#%% Reading data
df=real_load()
df=df['2019-04-01':'2019-07-30']
#%% Shift one time step behind (use previous time step as a input for for the predicted value)
df['pv_pred']=df['PV'].shift(24)
df['load_pred']=df['Load'].shift(24)
df['price_pred']=df['RTP'].shift(24)
# df['shift_2']=df['Global_active_power'].shift(24)
# df['shift_3']=df['Global_active_power'].shift(168)
# df['shift_4']=df['Global_active_power'].shift(720)
# df['shift_5']=df['Global_active_power'].shift(8760)
df.dropna(inplace=True)
#%% train,test split
train,test=data_split(df,0.9)
#%% Metrics
#print("Train data metrics",metrics(train['Load'],train['shift_1']))
print("Test data metrics",metrics(test['PV'],test['shift_1']))
#%% Visualization
# fig,ax=plt.subplots()
# ax.plot(test.index,test['Global_active_power'],label="Actual")
# ax.plot(test.index,test['shift_1'],color='r',label="Predicted")
# plt.xlim(pd.Timestamp('2010-10-01'),pd.Timestamp('2010-10-03'))
# plt.legend()
# plt.show()
#%%
path=r'C:\Users\Karthikeyan\Desktop\Github\Master-Thesis\Main_Folder\4_Single_Step_DNN\Prediction_csv\baseline.csv'
df=df[['pv_pred','load_pred','price_pred']]
df=df['2019-05-01':'2019-06-30']
#df_final['idx']=df_final.index
df.to_csv(path,index=True)