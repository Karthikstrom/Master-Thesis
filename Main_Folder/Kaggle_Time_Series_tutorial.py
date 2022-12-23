# -*- coding: utf-8 -*-
"""
Created on Thu Dec 22 14:44:56 2022

@author: Karthikeyan
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
decomp_df=sm.tsa.seasonal_decompose(df['grid_import'],model="additive")
decomp_df.resid.plot()