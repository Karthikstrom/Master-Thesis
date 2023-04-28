# -*- coding: utf-8 -*-
"""
Created on Tue Mar 14 14:14:50 2023

@author: Karthikeyan
"""

#%% Loading packages
import os
import sys
import path
import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
import itertools

current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

from Essential_functions import load_wholedata
from Cost_analysis import COE
#%% Read data

df= load_wholedata()
df= df[:8760]

#adding pv penetration and calculating mismatch
#df['PV']=7*df['PV']
df['Mismatch']=df['PV']-df['Load']

#%% Plot the TOU
 
#%% Strategy

for index, day in df.index.iterrows():
    #if weekday/weekedn and by hours
    dd=df[]

