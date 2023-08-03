# -*- coding: utf-8 -*-
"""
Created on Tue Jan  3 16:51:44 2023

@author: Karthikeyan
"""

#%% Loading packages

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

from statsmodels.tsa.stattools import acf, pacf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

sns.set_context('notebook')
sns.set_style("whitegrid")

#%% Read data
from Essential_functions import real_load

df=real_load()
df=df[df.index.year==2019]
#%% Plot auto correlation and partial auto correlation
# df['diff1']=df['Global_active_power'].diff(24)
# df.dropna(inplace=True)
# df['diff2']=df['diff1'].diff(8760)
# df.dropna(inplace=True)
plot_acf(df['Load'],lags=200)
plot_pacf(df['Load'],lags=200)

#%%
custom_params = {"axes.spines.right": False, "axes.spines.top": False}
csfont = {'fontname':'Times New Roman'}
# Create a figure with two subplots side by side
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

# Plot ACF in the first subplot
plot_acf(df['PV'], lags=200, ax=ax1)
ax1.set_title('Autocorrelation Function (ACF)')
ax1.set_xlabel('lags')

# Plot PACF in the second subplot
plot_pacf(df['PV'], lags=200, ax=ax2)
ax2.set_title('Partial Autocorrelation Function (PACF)')
ax2.set_xlabel('lags')

# Adjust layout and display the figure
plt.tight_layout()
#plt.savefig(r"C:\Users\Karthikeyan\Desktop\Thesis\Plots\ACF_PACF\correlogram_pv.jpeg",format="jpeg",dpi=1000)
plt.show()