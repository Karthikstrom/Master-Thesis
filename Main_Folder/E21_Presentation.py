# -*- coding: utf-8 -*-
"""
Created on Tue Jul 18 16:36:33 2023

@author: Karthikeyan
"""
#%% Calling packages
import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
from Essential_functions import load_data2,load_data,real_load,DA_prices,load_wholedata
import matplotlib.dates as mdates
import seaborn as sns

df=DA_prices()
#%%
sns.boxplot( x=df.index.year, y=df['RTP'],fliersize=1)
plt.title('Boxplot of Day-ahead Market prices NL by Year')
plt.xlabel('Year')
plt.ylabel('Spot Price [EUR/MWh]')
#plt.savefig(r'C:\Users\Karthikeyan\Desktop\Job Search\E21_Presentation_Plots\box.jpeg',dpi=1000)
plt.show()
#%%

df1=df['2023-04-18']
fig,ax=plt.subplots()
ax.plot(df1.index.hour,df1)
plt.title('NL Day Ahead Market Prices (8-07-2023)')
plt.xlabel('Hour of the day')
plt.ylabel('Spot Price [EUR/MWh]')
plt.show()

#%% Load plot

custom_params = {"axes.spines.right": False, "axes.spines.top": False}
csfont = {'fontname':'Times New Roman'}
sns.set_theme(style="ticks", rc=custom_params)


df=load_wholedata()
fig,cx=plt.subplots()
cx.plot(np.arange(0,24),df.loc['2022-5-02','Load'])
plt.title('Peak shaving')
plt.xlabel('Hour of the day')
plt.ylabel('Demand')
cx.set_yticks([])
#plt.savefig(r'C:\Users\Karthikeyan\Desktop\Job Search\E21_Presentation_Plots\load_shed.jpeg',dpi=1000)
plt.show()
#%%
custom_params = {"axes.spines.right": False, "axes.spines.top": False}
csfont = {'fontname':'Times New Roman'}
sns.set_theme(style="ticks", rc=custom_params)
fig,cx=plt.subplots()
cx.plot(np.arange(0,24),df[2400:2424])
plt.title('Energy Arbitrage potential in wholesale market')
plt.xlabel('Hour of the day')
plt.ylabel('Spot Price')
cx.set_yticks([])
#plt.savefig(r'C:\Users\Karthikeyan\Desktop\Job Search\E21_Presentation_Plots\arbitrage.jpeg',dpi=1000)
plt.show()