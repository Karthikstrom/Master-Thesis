# -*- coding: utf-8 -*-
"""
Created on Wed Jan 18 13:06:15 2023

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


import pmdarima as pm

current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

from Essential_functions import load_data2,metrics,data_split
#%% Reading data
df=load_data2()
#%%