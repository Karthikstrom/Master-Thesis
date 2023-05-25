# -*- coding: utf-8 -*-
"""
Created on Tue Mar 14 17:50:39 2023

@author: Karthikeyan
"""

#%% Loading functions
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
import itertools
import math
#%% Functions

h=1
eff_imp=1
eff_exp=1

def pb_in_func(pb_min,E_b,soc,soc_max):
    pb_in_temp=min(pb_min,(E_b/h)*(soc_max-soc))
    #to take the charge efficiency into consideration
    pb_in_temp=eff_imp*pb_in_temp
    return pb_in_temp

def pb_out_func(pb_max,E_b,soc,soc_min):
    pb_out_temp=min(pb_max,(E_b/h)*(soc-soc_min))
    #to take the discharge efficiency into consideration
    pb_out_temp=eff_exp*pb_out_temp
    
    
    return pb_out_temp

#calculate SOC
def SOC(soc_last,pb_imp,pb_exp,E_b):
    soc_temp=soc_last + (((pb_imp*eff_imp)-(pb_exp/eff_exp))/(E_b/h))# double check
    return soc_temp
