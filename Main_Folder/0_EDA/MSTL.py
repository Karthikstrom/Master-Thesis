# -*- coding: utf-8 -*-
"""
Created on Mon Jan 23 17:46:08 2023

@author: Karthikeyan
"""

#%% Loading packages

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

from Essential_functions import load_data2,metrics,data_split
#%% Read Data
df=load_data2()
#%% Decompose
mstl=MSTL(df['Global_active_power'],periods=[24,24*365])
res=mstl.fit()
#%% Plotting 
res.plot()
plt.savefig("Decompose.jpeg",dpi=500)
plt.show()