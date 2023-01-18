# -*- coding: utf-8 -*-
"""
Created on Thu Jan 12 13:20:35 2023

@author: Karthikeyan
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

"""
IDEAS:
    - outliers outside 3 Std
"""

#%% Creating n lags dataframe

def lag_dataframe(ip_series,n):
    shifts = np.arange(1, n).astype(int)

    # Use a dictionary comprehension to create name: value pairs, one pair per shift
    shifted_data = {"lag_{}_day".format(day_shift): 
                ip_series.shift(day_shift) for day_shift in shifts}
    shifted_data_df=pd.DataFrame(shifted_data)
    return shifted_data_df