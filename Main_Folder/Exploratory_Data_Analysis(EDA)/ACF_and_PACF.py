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

from Essential_functions import load_data2

from statsmodels.tsa.stattools import acf, pacf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

sns.set_context('notebook')
sns.set_style("whitegrid")

#%% Read data
df=load_data2()
#%% Plot auto correlation and partial auto correlation
plot_acf(df['Global_active_power'],lags=20)
plot_pacf(df['Global_active_power'],lags=20)