# -*- coding: utf-8 -*-
"""
Created on Wed Jan 11 13:05:37 2023

@author: Karthikeyan
"""
#%% Loading packages


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from Essential_functions import load_data2

from statsforecast import StatsForecast

from statsforecast.models import (
    AutoARIMA,
    HoltWinters,
    CrostonClassic as Croston, 
    HistoricAverage,
    DynamicOptimizedTheta as DOT,
    SeasonalNaive
)
#%% Read data
df=load_data2()
#%% Formatting the data
df['unique_id']="load"
df['ds']=df.index
df.rename(columns={'Global_active_power':'y'},inplace=True)
df_s=df[['unique_id','ds','y']]


test=df_s.loc['2010-12-04':'2010-12-11']
df_s=df_s.loc[:'2010-12-04']
#%% Basic EDA
StatsForecast.plot(df_s,engine='matplotlib')
plt.show()
#%% Calling the statistical models
models = [
    AutoARIMA(season_length=24),
    HoltWinters(),
    Croston(),
    SeasonalNaive(season_length=24),
    HistoricAverage(),
    DOT(season_length=24)
]

#%% Creating a sf object of statsforecast

sf = StatsForecast(
    df=df_s, 
    models=models,
    freq='H', 
    n_jobs=-1,
    fallback_model = SeasonalNaive(season_length=12)
)

#%% Forecasting 
forecasts_df = sf.forecast(h=48)
#%% Plotting forecast
sf.plot(df_s,forecasts_df,engine='matplotlib')
fig,ax=plt.subplots()
#%% Crossvalidation
crossvaldation_df = sf.cross_validation(
    df=df_s,
    h=24,
    step_size=24,
    n_windows=7
  )