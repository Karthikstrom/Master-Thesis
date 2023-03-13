# -*- coding: utf-8 -*-
"""
Created on Mon Mar 13 00:33:58 2023

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


current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

from Essential_functions import load_data
import pywt
#%% Read data
df=load_data()
df=df[:-1]
#%% Wavelet Transformation








