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


import pmdarima as pm

current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

from Essential_functions import load_data2,metrics,data_split
#%% Read Data|
df=load_data2()
#%% Adfuller test

"""
Null Hypothesis: It is not stationary
Alternate Hypothesis: It is stationary

if p<0.05 || t-statistic<critical value - cannot reject null hypothesis

meaning it is non stationary

p could be extremely small as seen from adf statistic much less than 1% critical 
value


"""
result = adfuller(df['Global_active_power'])
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

