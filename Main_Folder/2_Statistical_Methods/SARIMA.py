# -*- coding: utf-8 -*-
"""
Created on Sun Mar 12 01:24:20 2023

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
import pickle
import itertools
import matplotlib.dates as mdates

current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

from statsmodels.graphics.tsaplots import plot_acf,plot_pacf
from pandas.plotting import autocorrelation_plot
import pmdarima as pm
import statsmodels.api as sm
import statsmodels.tsa.api as smt
from itertools import product   
from tqdm import tqdm_notebook

from sklearn.metrics import r2_score, median_absolute_error, mean_absolute_percentage_error
from sklearn.metrics import median_absolute_error, mean_squared_error, mean_squared_log_error


from Essential_functions import load_data,metrics,load_data2,real_load
sns.set_theme()
#%% Read Data
df=real_load()
df['Load']=df['Load'].astype(np.float32)
df=df[-8760:]
df1=df.copy()

#%% Exp

#Diff
df['Diff1']=df['Load'].diff(1)
df['Rev_Diff1']=df['Diff1']+df['Load'].shift(1)


#%%Analyse and determine parameter values
seasonal_diff=df['Load'].diff(1)
seasonal_diff.dropna(inplace=True)
seasonal_diff2=seasonal_diff.diff(24)
seasonal_diff2.dropna(inplace=True)
seasonal_diff3=seasonal_diff2.diff(168)
seasonal_diff3.dropna(inplace=True)
#%% Train Test split
train=df[:-720*3]
test=df[-720*3:]
#%% Functions
def tsplot(y, lags=None, figsize=(12, 7), style='bmh'):
    """
        Plot time series, its ACF and PACF, calculate Dickey–Fuller test
        
        y - timeseries
        lags - how many lags to include in ACF, PACF calculation
    """
    
    if not isinstance(y, pd.Series):
        y = pd.Series(y)
        
    with plt.style.context(style):    
        fig = plt.figure(figsize=figsize)
        layout = (2, 2)
        ts_ax = plt.subplot2grid(layout, (0, 0), colspan=2)
        acf_ax = plt.subplot2grid(layout, (1, 0))
        pacf_ax = plt.subplot2grid(layout, (1, 1))
        
        y.plot(ax=ts_ax)
        p_value = sm.tsa.stattools.adfuller(y)[1]
        ts_ax.set_title('Time Series Analysis Plots\n Dickey-Fuller: p={0:.5f}'.format(p_value))
        smt.graphics.plot_acf(y, lags=lags, ax=acf_ax)
        smt.graphics.plot_pacf(y, lags=lags, ax=pacf_ax)
        plt.tight_layout()
        
def optimizeSARIMA(parameters_list, d, D, s):
    """Return dataframe with parameters and corresponding AIC
        
        parameters_list - list with (p, q, P, Q) tuples
        d - integration order in ARIMA model
        D - seasonal integration order 
        s - length of season
    """
    
    results = []
    best_aic = float("inf")

    for param in tqdm_notebook(parameters_list):
        # we need try-except because on some combinations model fails to converge
        try:
            model=sm.tsa.statespace.SARIMAX(df['Load'], order=(param[0], d, param[1]), 
                                            seasonal_order=(param[2], D, param[3], s)).fit(disp=-1)
        except:
            continue
        aic = model.aic
        # saving best model, AIC and parameters
        if aic < best_aic:
            best_model = model
            best_aic = aic
            best_param = param
        results.append([param, model.aic])

    result_table = pd.DataFrame(results)
    result_table.columns = ['parameters', 'aic']
    # sorting in ascending order, the lower AIC is - the better
    result_table = result_table.sort_values(by='aic', ascending=True).reset_index(drop=True)
    
    return result_table

def plotSARIMA(series, model, n_steps):
    """Plots model vs predicted values
        
        series - dataset with timeseries
        model - fitted SARIMA model
        n_steps - number of steps to predict in the future    
    """
    
    # adding model values
    data = series.copy()
    data.columns = ['actual']
    data['sarima_model'] = model.fittedvalues
    # making a shift on s+d steps, because these values were unobserved by the model
    # due to the differentiating
    data['sarima_model'][:s+d] = np.NaN
    
    # forecasting on n_steps forward 
    forecast = model.predict(start = data.shape[0], end = data.shape[0]+n_steps)
    forecast = data.sarima_model.append(forecast)
    # calculate error, again having shifted on s+d steps from the beginning
    #error = mean_absolute_percentage_error(data['actual'][s+d:], data['sarima_model'][s+d:])

    plt.figure(figsize=(15, 7))
    #plt.title("Mean Absolute Percentage Error: {0:.2f}%".format(error))
    plt.plot(forecast, color='r', label="model")
    plt.axvspan(data.index[-1], forecast.index[-1], alpha=0.5, color='lightgrey')
    plt.plot(data.actual, label="actual")
    plt.legend()
    plt.grid(True) 
#%% Calling decomposition model (No Use)
#mst=pickle.load(open(r'C:\Users\Karthikeyan\Desktop\Thesis\Model Database\MSTL1.sav','rb'))
#%% Visualizing the decomposition (No Use)
# plot,ax=plt.subplots()
# ax.plot(mst.trend)

# plot,bx=plt.subplots()
# bx.plot(mst.seasonal['seasonal_24'].iloc[:168])
# bx.xaxis.set_major_formatter(mdates.DateFormatter('%a'))

# plot,cx=plt.subplots()
# cx.plot(mst.seasonal['seasonal_168'].iloc[:168])
# cx.xaxis.set_major_formatter(mdates.DateFormatter('%a'))

# plot,dx=plt.subplots()
# dx.plot(mst.seasonal['seasonal_8760'])

# plot,ex=plt.subplots()
# ex.plot(mst.resid)

# plt.show()
#%% Corralation Plots (Not Use)
# plot_acf(df['Load'],lags=100)
# plot_pacf(df['Load'],lags=100)
# autocorrelation_plot(df['Load'])

# #SARIMA Model without converting to stationary
# smodel = pm.auto_arima(train['Seasonaldiff1'], start_p=1, start_q=1,
#                          test='adf',
#                          max_p=3, max_q=3, m=24,
#                          start_P=0, seasonal=True,
#                          d=None, D=1, trace=True,
#                          error_action='ignore',  
#                          suppress_warnings=True, 
#                          stepwise=True)
#%%
"""
p- last significant value in pacf plot(3-7) 
d- how many differences have we done(here=1)
q- same as p but here it is around 6


P-around 4 because 
D=1 (for one differentiation its okay)

"""
#Setting initial values and some bounds for them from ACF and PACF
# ps = range(3, 7)
# d=1 
# qs = range(5, 7)
# Ps = range(3, 5)
# D=1 
# Qs = range(0, 1)
# s = 24 # season length is still 24

# # creating list with all the possible combinations of parameters
# parameters = product(ps, qs, Ps, Qs)
# parameters_list = list(parameters)
# len(parameters_list)


# add the grid search part
#%% Model Training

p=2
d=1
q=1
P=3
D=0
Q=2
s=24

best_model=sm.tsa.statespace.SARIMAX(train['Load'], order=(p, d, q), 
                                        seasonal_order=(P, D, Q, s)).fit(disp=-1)

d=1,#%% Predict 
forecast = best_model.predict(start = len(train), end = len(train)+len(test)-1)

#%% Save Model
filename=r'C:\Users\Karthikeyan\Desktop\Thesis\Model Database\SARIMA.sav'
pickle.dump(best_model,open(filename,'wb'))
#%% Plotting

test.loc[:,'predicted']=forecast

fig,ax=plt.subplots(figsize=(10,5))
ax.plot(test['Load'],label="Actual",color='b')
ax.plot(test['predicted'],label="Predicted",color='r')
ax.set_ylabel("Load (kW)")
ax.xaxis.set_major_formatter(mdates.DateFormatter('%d-%b\n%Y\n%a'))
#plt.title("SARIMA")

plt.xlim(datetime.datetime(2019, 6, 3), datetime.datetime(2019, 6, 10))
plt.legend()
#plt.savefig(r"C:\Users\Karthikeyan\Desktop\Github\Master-Thesis\Main_Folder\13_Plots\Conference_ISGT\SARIMA1.jpeg",format="jpeg",dpi=500)
plt.show()

