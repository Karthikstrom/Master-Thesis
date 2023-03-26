# -*- coding: utf-8 -*-
"""
Created on Thu Dec 15 13:44:46 2022

@author: Karthikeyan
"""

#%% Importing package
import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
from Essential_functions import load_data2,load_data
import matplotlib.dates as mdates
import seaborn as sns
sns.set_theme()
# sns.set_context('notebook')
# sns.set_style("whitegrid")
#%% Importing data
df=load_data()
#%% Clipping the data from first non-zero value to the last
#df=df.loc['2016-01-01':'2016-12-31']
df.dropna(inplace=True)
#%% Checking number of null values in the data set
# to check if there is any duplicated index use
print("Number of duplicated index = ",len(df)-len(df.index.unique()))
print("Number of nan values = ",df.isnull().sum()[0])
#%% Getting the differences to get the absolute value
df['grid_import']=df['grid_import'].diff()
#%% Removing Nan values
df.dropna(inplace=True)
#%% Plotting the whole data
fig,a1=plt.subplots(figsize=(10,5))
a1.plot(df.loc['22-08-2009':'28-08-2009'])

a1.xaxis.set_major_formatter(mdates.DateFormatter('%d-%b\n%Y\n%a'))
#plt.title("Clean data sample for a week")
plt.ylabel("Load (KW)")
#plt.tight_layout()
#plt.savefig(r"C:\Users\Karthikeyan\Desktop\Github\Master-Thesis\Main_Folder\13_Plots\Conference_ISGT\Week_data2.jpeg",format="jpeg",dpi=500)
plt.show()

#%% Plotting weekly average
fig,bx=plt.subplots()
bx.plot(df['grid_import'].resample('W').sum())
plt.show()
#%% Plotting Monthly average
fig,cx=plt.subplots()
cx.plot(df['grid_import'].resample('M').sum())
plt.show()
#%% Scatter plot of data
fig,dx=plt.subplots()
dx.scatter(df.index,df['grid_import'],1)
plt.show()
#%% Groupby functionality
df['grid_import'].groupby(df.index.strftime('%Y')).sum()
#%% Rolling mean plots
fig,ex=plt.subplots()
ex.plot(df['grid_import'].rolling(4*24*7).mean())
plt.show()
#%% Comparing performance

# Make a random dataset:
mae = [0.4065, 0.596, 0.4254, 0.3248, 0.3794,0.4465,0.3701]
rmse = [0.531,0.7377,0.589,0.5605,0.4999,0.5709,0.5015]
bars = ('LR', 'SARIMA', 'SVM','RF', 'MLP', 'CNN','LSTM')
y_pos = np.arange(len(bars))

# Create bars
plt.bar(y_pos, rmse)

# Create names on the x-axis
plt.xticks(y_pos, bars)
plt.title("Comparing performace (Root mean squared error)")
plt.ylabel("Root mean squared error (Kwh)")
plt.savefig("MAE_comparison.jpeg",dpi=500)
# Show graphic
plt.show()

#%%
df=pd.read_csv(r"C:\Users\Karthikeyan\Desktop\Github\Master-Thesis\Main_Folder\results.xlsx")
#%%
#%% Comparing performance

# Make a random dataset:
mae = [25.15480832,
11.97075944,
26.12640486,
14.13520033,
12.07517478,
13.83541342,
14.46162246,
16.47160216,
24.9791729,
14.36905718,
15.64041565,
14.5844732,
14.5844732,
]
rmse = [35.069598,
16.34369691,
32.37749085,
22.30904223,
17.338807,
19.11682393,
20.21142801,
20.30208753,
35.66119175,
20.79253201,
20.84914078,
19.77689019,
19.77689019,
]
mape = [9.552275985,
3.761813667,
9.406085567,
4.436490562,
3.88228065,
4.263283586,
4.651731739,
5.981574636,
9.497837432,
4.403867745,
5.059232113,
4.714961273,
4.714961273,

]
bars = ('LR','RF','SVM','Baseline','MLP','CNN','RNN','LSTM','SLSTM','BLSTM','CNN-LSTM','ConvLSTM','EncDec LSTM')
y_pos = np.arange(len(bars))

# Create bars
plt.bar(y_pos, mape)

# Create names on the x-axis
plt.xticks(y_pos, bars, rotation=45)
plt.title("Comparing Mean Percetage Absolute Error")
plt.ylabel("Mean Absolute Percentage Error (%)")
plt.tight_layout()
plt.savefig(r"C:\Users\Karthikeyan\Desktop\Github\Master-Thesis\Main_Folder\MAPE_comparison.jpeg",dpi=500)
# Show graphic
plt.show()

#%% Density plot of the data
#sns.histplot(data=df['Load'],kde=True)
sns.distplot(df['Load'])
plt.xlabel('Load (W)')
#plt.savefig(r"C:\Users\Karthikeyan\Desktop\Github\Master-Thesis\Main_Folder\13_Plots\Conference_ISGT\distt_plot.jpeg",format="jpeg",dpi=500)
plt.show()