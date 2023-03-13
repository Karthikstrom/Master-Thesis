# -*- coding: utf-8 -*-
"""
Created on Fri Jan 13 15:25:49 2023

@author: Karthikeyan
"""

import os
import sys
import path

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import kpss
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.graphics.tsaplots import plot_predict
import matplotlib.dates as mdates
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.seasonal import seasonal_decompose

import pmdarima as pm

current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)
from statsmodels.tsa.seasonal import MSTL
from statsmodels.tsa.seasonal import DecomposeResult

from Essential_functions import load_data,metrics,data_split

"""

It is worth noting that differencing only remove additive seasonality, and if the series has a multiplicative seasonality,
 then we need to use log-differencing.
 
There are several ways to check if a time series has additive or multiplicative seasonality. Some common methods include:

Visual inspection: Plot the original time series and look for patterns that repeat
 at regular intervals. If the patterns are consistent across all levels of the series, 
 it is likely additive seasonality. If the patterns change in amplitude as the level of 
 the series changes, it is likely multiplicative seasonality.

Seasonal decomposition: Use a statistical method such as the seasonal decomposition
 of time series (STL) to decompose the original time series into its trend, seasonal, 
 and residual components. If the seasonal component is consistent across all levels of 
 the series, it is likely additive seasonality. If the seasonal component changes in 
 amplitude as the level of the series changes, it is likely multiplicative seasonality.

Scatter plot: Create a scatter plot of the original time series with the seasonal 
component of the series on one axis and the remainder (trend + residual) on the other 
axis. If the points on the scatter plot fall along a diagonal line, it is likely additive 
seasonality. If the points on the scatter plot form a cloud, it is likely multiplicative seasonality.

It is worth noting that identifying the type of seasonality is important because 
it affects how the series should be handled for further analysis.

"""
#%% Read Data|
df=load_data()
#%% Seasonal decompose
result=seasonal_decompose(df['Global_active_power'], model='multiplicable' )
#%% Adfuller test

"""
Null Hypothesis: It is not stationary
Alternate Hypothesis: It is stationary

if p<0.05 || t-statistic<critical value - cannot reject null hypothesis

meaning it is non stationary

p could be extremely small as seen from adf statistic much less than 1% critical 
value


"""
result = adfuller(df['Load'])
print('ADF Statistic: {}'.format(result[0]))
print('p-value: {}'.format(result[1]))
print('Critical Values:')
for key, value in result[4].items():
    print('\t{}: {}'.format(key, value))

# reject null hyp as p value lower than 0.05
# cannot reject null hyp as abs(t-value) > Critical value

#%% KPSS test

"""
Null Hypothesis: It is stationary
Alternate Hypothesis: It is not stationary

"""
kpss_test = kpss(df['Global_active_power'])

print('KPSS Statistic: %f' % kpss_test[0])
print('Critical Values @ 0.05: %.2f' % kpss_test[3]['5%'])
print('p-value: %f' % kpss_test[1])

