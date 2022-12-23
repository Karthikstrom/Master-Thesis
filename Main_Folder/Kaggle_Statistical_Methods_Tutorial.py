# -*- coding: utf-8 -*-
"""
Created on Thu Dec 22 14:44:56 2022

@author: Karthikeyan

https://www.kaggle.com/code/saurav9786/time-series-tutorial

"""

#%% Importing packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from statsmodels.tsa.seasonal import seasonal_decompose
import  statsmodels.api as sm

from Essential_functions import load_data
sns.set_context('notebook')
sns.set_style("whitegrid")
#%% Importing data
df=pd.DataFrame()
df['grid_import']=load_data('2015-11-01','2018-07-30')
#%% Boxplot of each hour
sns.boxplot(x=df.index.hour,y=df['grid_import'])
plt.show()
#%% Seasonal decompose
decomp_df=sm.tsa.seasonal_decompose(df['grid_import'],model="multiplicative")
decomp_df.resid.plot()
seasonality=decomp_df.seasonal
residual=decomp_df.resid
#%% Plotting daily seasonality
fig,ax=plt.subplots()
ax.plot(seasonality[0:100])
plt.show()
#%% Seasonal decompose of residual 
residual.dropna(inplace=True)
decomp_df2=sm.tsa.seasonal_decompose(residual,model="multiplicative")
#Noisty trend and sort of a granual same seasonality(for both additive and multiplicative)
#multiplicative model captures more of the seasonality
#%% Plotting each year on top of each other YTW
#%% Mean/Rolling mean as the forecast plot
df['avg_forecast']=df['grid_import'].rolling(24).mean()
fig,bx=plt.subplots()
bx.plot(df['grid_import'])
bx.plot(df['avg_forecast'])
plt.show()
