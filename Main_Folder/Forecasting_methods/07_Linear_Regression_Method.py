# -*- coding: utf-8 -*-
"""
Created on Sun Jan  8 19:02:10 2023

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

import statsmodels.api as sm


from sklearn.linear_model import LinearRegression
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

from Essential_functions import load_data2,metrics,data_split
#%% Read data
df=load_data2()

"""
Looking at min and max there are outliers

model should be robust to that

maybe multiple linear regression?
"""
#%% Feauture creation

# df['lag_1']=df['Global_active_power'].shift(1)
# #df['rolling mean']=df['Global_active_power'].rolling(24).mean()
# df.dropna(inplace=True)



train,test=data_split(df,0.9)

# y_train=pd.DataFrame()
# x_train=train.drop('Global_active_power',axis=1)
# y_train['Global_active_power']=train['Global_active_power']


# y_test=pd.DataFrame()
# X_test=test.drop('Global_active_power',axis=1)
# y_test['Global_active_power']=train['Global_active_power']

#%%
# SARIMA example
from statsmodels.tsa.statespace.sarimax import SARIMAX

# fit model
model = SARIMAX(train, order=(0, 1, 2), seasonal_order=(0, 0, 2, 24))
model_fit = model.fit(disp=False)

#%% 

predictions = model_fit.predict(start=len(train), end=(len(train)+len(test)-1))

#%% Building the model
lr_model=LinearRegression()

# X=np.asarray(train['lag_1'])
# y=np.asarray(train['Global_active_power'])

# X=np.reshape(X,(-1,1))
# y=np.reshape(y,(-1,1))

# X=np.reshape(np.asarray(train['lag_1']),(1,-1))
# y=np.reshape(np.asarray(train['Global_active_power']),(1-1))


lr_model.fit(x_train,y_train)
#%% Predicting and 

val=np.asarray(test['lag_1'])
val=np.reshape(val,(-1,1))

y_pred =lr_model.predict(val)
test['y_pred']=y_pred

#%% Metrics and plotting

metrics(test['Global_active_power'],test['y_pred'])

fig,ax=plt.subplots()
ax.plot(test['Global_active_power'],label="Actual")
ax.plot(test['y_pred'],label="Predicted",color='r')
#plt.xlim(500,600)
plt.legend()
plt.show()

#%% To check for outliers / Box plot 
sns.boxplot(df['Global_active_power'])

"""
The linear regression algorithm learns how to make a weighted sum from 
its input features. For two features, we would have:

    target = weight_1 * feature_1 + weight_2 * feature_2 + bias

"""
#%%

# with sns.plotting_context("notebook",font_scale=2.5):
#     g = sns.pairplot(dataset[['sqft_lot','sqft_above','price','sqft_living','bedrooms']], 
#                  hue='bedrooms', palette='tab20',size=6)
# g.set(xticklabels=[]);