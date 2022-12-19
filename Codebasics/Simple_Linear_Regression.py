# -*- coding: utf-8 -*-
"""
Created on Fri Dec  9 16:17:25 2022

@author: Karthikeyan
"""

#%% Importing packages

import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
from sklearn import linear_model

#%% Creating dataset

data=[[2600,550000],[3000,565000],[3200,610000],[3600,680000],[4000,725000]]
df=pd.DataFrame(data,columns=['area','price'])


#%% Building the model

reg=linear_model.LinearRegression()
reg.fit(df[['area']],df['price'])

#%% Prediction
reg.predict([[3300]])

#%% Scatter plot with best fit line

fig,ax=plt.subplots()
ax.scatter(df['area'],df['price'],color='red',marker='+')
ax.plot(df['area'],reg.coef_*df['area']+reg.intercept_)
ax.set_xlabel("Area")
ax.set_ylabel("Price")
plt.show()
