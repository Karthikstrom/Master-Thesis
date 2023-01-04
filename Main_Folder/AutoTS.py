# -*- coding: utf-8 -*-
"""
Created on Fri Dec 23 17:38:35 2022

@author: Karthikeyan

https://pypi.org/project/AutoTS/

"""

#%% Importing libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from autots import AutoTS,load_daily

from Essential_functions import load_data
#%% Importing load data

df=pd.DataFrame()
df['grid_import']=load_data('2015-11-01','2020-07-30')

long=False
#df= load_daily(long=long)

model = AutoTS(
    forecast_length=21,
    frequency='infer',
    prediction_interval=0.9,
    ensemble=None,
    model_list="fast",  # "superfast", "default", "fast_parallel"
    transformer_list="fast",  # "superfast",
    drop_most_recent=1,
    max_generations=4,
    num_validations=2,
    validation_method="backwards"
)
model = model.fit(
    df
    #date_col='datetime' if long else None,
    #value_col='value' if long else None,
    #id_col='series_id' if long else None,
)

prediction = model.predict()
# plot a sample
prediction.plot(model.df_wide_numeric,
                series=model.df_wide_numeric.columns[0],
                start_date="2015-11-01")
# Print the details of the best model
print(model)

# point forecasts dataframe
forecasts_df = prediction.forecast
# upper and lower forecasts
forecasts_up, forecasts_low = prediction.upper_forecast, prediction.lower_forecast

# accuracy of all tried model results
model_results = model.results()
# and aggregated from cross validation
validation_results = model.results("validation")