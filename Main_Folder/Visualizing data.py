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
from Essential_functions import load_data2,load_data,real_load
import matplotlib.dates as mdates
import seaborn as sns
custom_params = {"axes.spines.right": False, "axes.spines.top": False}
csfont = {'fontname':'Times New Roman'}
sns.set_theme(style="ticks", rc=custom_params)
# sns.set_context('notebook')
# sns.set_style("whitegrid")
#%% Importing data
df=real_load()
df['day']=df.index.dayofweek
df['h']=df.index.hour
#%% Clipping the data from first non-zero value to the last
#df=df.loc['2016-01-01':'2016-12-31']
df.dropna(inplace=True)
#%% Checking number of null values in the data set
# to check if there is any duplicated index use
print("Number of duplicated index = ",len(df)-len(df.index.unique()))
print("Number of nan values = ",df.isnull().sum()[0])
#%% Getting the differences to get the absolute value
#df['grid_import']=df['grid_import'].diff()
#%% Removing Nan values
#df.dropna(inplace=True)
#%% Plotting the whole data
fig,a1=plt.subplots(figsize=(12,7))
a1.plot(df['PV'].loc['2019-06-03':'2019-06-10'])

a1.xaxis.set_major_formatter(mdates.DateFormatter('%d-%b\n%Y\n%a'))
#a1.set_xlabel("Time",fontsize=24,**csfont)
a1.set_ylabel("PV Generation (Kw)",fontsize=24,**csfont)
plt.yticks(fontsize=20,**csfont)
plt.xticks(fontsize=20,**csfont)
plt.tight_layout()
#plt.savefig(r"C:\Users\Karthikeyan\Desktop\Github\Master-Thesis\Main_Folder\13_Plots\Thesis_Report\EDA\ts_pv.jpeg",format="jpeg",dpi=1000)
plt.show()

#%%
fig,a1=plt.subplots(figsize=(12,7))
a1.plot(df['RTP'].loc['2019-06-03':'2019-06-10'])

a1.xaxis.set_major_formatter(mdates.DateFormatter('%d-%b\n%Y\n%a'))
#a1.set_xlabel("Time",fontsize=24,**csfont)
a1.set_ylabel("Price (\u20AC/Kw)",fontsize=24,**csfont)
plt.yticks(fontsize=20,**csfont)
plt.xticks(fontsize=20,**csfont)
plt.tight_layout()
#plt.savefig(r"C:\Users\Karthikeyan\Desktop\Github\Master-Thesis\Main_Folder\13_Plots\Thesis_Report\EDA\ts_price.jpeg",format="jpeg",dpi=1000)
plt.show()

#%%
fig,a1=plt.subplots(figsize=(12,7))
a1.plot(df['Load'].loc['2019-06-03':'2019-06-10'])

a1.xaxis.set_major_formatter(mdates.DateFormatter('%d-%b\n%Y\n%a'))
#a1.set_xlabel("Time",fontsize=24,**csfont)
a1.set_ylabel("Load (Kw)",fontsize=24,**csfont)
plt.yticks(fontsize=20,**csfont)
plt.xticks(fontsize=20,**csfont)
plt.tight_layout()
#plt.savefig(r"C:\Users\Karthikeyan\Desktop\Github\Master-Thesis\Main_Folder\13_Plots\Thesis_Report\EDA\ts_load.jpeg",format="jpeg",dpi=1000)
plt.show()

#%% Weekly average


std=df.groupby(['day','h'],as_index=False).std()
weekly_avg=df.groupby(['day','h'],as_index=False).mean()

x_lables=['Mon','Tue','Wed','Thur','Fri','Sat','Sun']
x_pos=[i for i in range(12,len(weekly_avg),24)]

fig,ax=plt.subplots(figsize=(12,7))
ax=sns.lineplot(x=weekly_avg.index, y=weekly_avg['PV'])
plt.fill_between(weekly_avg.index, weekly_avg['PV'] +std['PV'], weekly_avg['PV'] -std['PV'], alpha=0.3)
ax.set_xlabel("Time",fontsize=24,**csfont)
ax.set_ylabel("PV Generated (Kw)",fontsize=24,**csfont)
plt.xticks(x_pos, x_lables)
plt.yticks(fontsize=20,**csfont)
plt.xticks(fontsize=20,**csfont)
#plt.savefig(r"C:\Users\Karthikeyan\Desktop\Thesis\Mid_Term_Presentation\Common_plots\pv_ov.jpeg",format="jpeg",dpi=1000)
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

#%% Load R squared

base   =  [-0.356]
sarima =  [0.149]
mlp    =  [0.438]
cnn    =  [0.435]
rnn    =  [0.448]
lstm   =  [0.453]
bilstm =  [0.450]
cnnlstm=  [0.444]

x = np.arange(len(sarima))
width = 0.1
  
plt.figure(figsize=(20, 35))

# plot data in grouped manner of bar type

plt.bar(x-3*width, base, width, color='midnightblue')
plt.bar(x-2*width, sarima, width, color='mediumblue')
plt.bar(x-width, mlp, width, color='steelblue')
plt.bar(x, cnn, width, color='dodgerblue')
plt.bar(x+width, rnn, width, color='lavender')
plt.bar(x+2*width, lstm, width, color='royalblue')
plt.bar(x+3*width, bilstm, width, color='cornflowerblue')
plt.bar(x+4*width, cnnlstm, width, color='lightsteelblue')
plt.yticks(fontsize=45,**csfont)
plt.xticks(fontsize=45,**csfont)
plt.xticks([])
#plt.xlabel("ERROR METRICS",fontsize=50,**csfont)
plt.ylabel("R Squared values",fontsize=50,**csfont)
plt.legend(["Baseline", "SARIMA", "MLP","CNN","RNN","LSTM","Bidirectional LSTM","CNN-LSTM Hybrid"],prop = { "size": 35 })
plt.title("R squared of Actual vs Predicted Load ",fontsize=60,**csfont)
#plt.savefig(r"C:\Users\Karthikeyan\Desktop\Thesis\Mid_Term_Presentation\Common_plots\rs_load.jpeg",format="jpeg",dpi=200)
plt.show()

#%% PV R squared

base   =  [0.593]
sarima =  [0.188]
mlp    =  [0.921]
cnn    =  [0.926]
rnn    =  [0.921]
lstm   =  [0.927]
bilstm =  [0.928]
cnnlstm=  [0.923]

x = np.arange(len(sarima))
width = 0.1
  
plt.figure(figsize=(20, 35))

# plot data in grouped manner of bar type

plt.bar(x-3*width, base, width, color='midnightblue')
plt.bar(x-2*width, sarima, width, color='mediumblue')
plt.bar(x-width, mlp, width, color='steelblue')
plt.bar(x, cnn, width, color='dodgerblue')
plt.bar(x+width, rnn, width, color='lavender')
plt.bar(x+2*width, lstm, width, color='royalblue')
plt.bar(x+3*width, bilstm, width, color='cornflowerblue')
plt.bar(x+4*width, cnnlstm, width, color='lightsteelblue')
plt.yticks(fontsize=45,**csfont)
plt.xticks(fontsize=45,**csfont)
plt.xticks([])
#plt.xlabel("ERROR METRICS",fontsize=50,**csfont)
plt.ylabel("R Squared values",fontsize=50,**csfont)
#plt.legend(["Baseline", "SARIMA", "MLP","CNN","RNN","LSTM","Bidirectional LSTM","CNN-LSTM Hybrid"],prop = { "size": 35 })
plt.title("R squared of Actual vs Predicted PV Generation ",fontsize=60,**csfont)
#plt.savefig(r"C:\Users\Karthikeyan\Desktop\Thesis\Mid_Term_Presentation\Common_plots\rs_PV.jpeg",format="jpeg",dpi=300)
plt.show()

#%% Price R squared

base   =  [0.433]
sarima =  [0.142]
mlp    =  [0.920]
cnn    =  [0.950]
rnn    =  [0.956]
lstm   =  [0.951]
bilstm =  [0.960]
cnnlstm=  [0.916]

x = np.arange(len(sarima))
width = 0.1
  
plt.figure(figsize=(20, 35))

# plot data in grouped manner of bar type

plt.bar(x-3*width, base, width, color='midnightblue')
plt.bar(x-2*width, sarima, width, color='mediumblue')
plt.bar(x-width, mlp, width, color='steelblue')
plt.bar(x, cnn, width, color='dodgerblue')
plt.bar(x+width, rnn, width, color='lavender')
plt.bar(x+2*width, lstm, width, color='royalblue')
plt.bar(x+3*width, bilstm, width, color='cornflowerblue')
plt.bar(x+4*width, cnnlstm, width, color='lightsteelblue')
plt.yticks(fontsize=45,**csfont)
plt.xticks(fontsize=45,**csfont)
plt.xticks([])
#plt.xlabel("ERROR METRICS",fontsize=50,**csfont)
plt.ylabel("R Squared values",fontsize=50,**csfont)
#plt.legend(["Baseline", "SARIMA", "MLP","CNN","RNN","LSTM","Bidirectional LSTM","CNN-LSTM Hybrid"],prop = { "size": 35 },bbox_to_anchor=(1.02, 1))
plt.title("R squared of Actual vs Predicted Electricity ",fontsize=60,**csfont)
plt.savefig(r"C:\Users\Karthikeyan\Desktop\Thesis\Mid_Term_Presentation\Common_plots\rs_Price.jpeg",format="jpeg",dpi=300)
plt.show()
#%% Load

base   =  [0.251, 0.410]
sarima =  [0.231, 0.322]
mlp    =  [0.195, 0.281]
cnn    =  [0.179, 0.275]
rnn    =  [0.183, 0.277]
lstm   =  [0.175,0.270]
bilstm =  [0.173,0.269]
cnnlstm=  [0.181,0.271]

x = np.arange(len(sarima))
width = 0.1
  
plt.figure(figsize=(25, 30))

# plot data in grouped manner of bar type

plt.bar(x-3*width, base, width, color='midnightblue')
plt.bar(x-2*width, sarima, width, color='mediumblue')
plt.bar(x-width, mlp, width, color='steelblue')
plt.bar(x, cnn, width, color='dodgerblue')
plt.bar(x+width, rnn, width, color='lavender')
plt.bar(x+2*width, lstm, width, color='royalblue')
plt.bar(x+3*width, bilstm, width, color='cornflowerblue')
plt.bar(x+4*width, cnnlstm, width, color='lightsteelblue')
plt.yticks(fontsize=45,**csfont)
plt.xticks(fontsize=45,**csfont)
plt.xticks(x, ['MAE', 'RMSE'])
plt.xlabel("ERROR METRICS",fontsize=50,**csfont)
plt.ylabel("Scores (kW)",fontsize=50,**csfont)
plt.legend(["Baseline", "SARIMA", "MLP","CNN","RNN","LSTM","Bidirectional LSTM","CNN-LSTM Hybrid"],prop = { "size": 35 })
#plt.title("Load Prediction Results",fontsize=60,**csfont)
plt.savefig(r"C:\Users\Karthikeyan\Desktop\Thesis\Mid_Term_Presentation\Common_plots\metrics_load.jpeg",format="jpeg",dpi=300)
plt.show()

#%% Price

base   =  [43.3, 65.9]
sarima =  [49, 59.9]
mlp    =  [19,24.6 ]
cnn    =  [13.8,19.3 ]
rnn    =  [13.4, 18.1]
lstm   =  [13.6,19.3]
bilstm =  [12.5,17.4]
cnnlstm=  [17.3,25.2]

x = np.arange(len(sarima))
width = 0.1
  
plt.figure(figsize=(40, 30))

# plot data in grouped manner of bar type

plt.bar(x-3*width, base, width, color='midnightblue')
plt.bar(x-2*width, sarima, width, color='mediumblue')
plt.bar(x-width, mlp, width, color='steelblue')
plt.bar(x, cnn, width, color='dodgerblue')
plt.bar(x+width, rnn, width, color='lavender')
plt.bar(x+2*width, lstm, width, color='royalblue')
plt.bar(x+3*width, bilstm, width, color='cornflowerblue')
plt.bar(x+4*width, cnnlstm, width, color='lightsteelblue')
plt.yticks(fontsize=45,**csfont)
plt.xticks(fontsize=45,**csfont)
plt.xticks(x, ['MAE', 'RMSE'])
plt.xlabel("ERROR METRICS",fontsize=50,**csfont)
plt.ylabel("Price (€/Mwh)",fontsize=50,**csfont)
plt.legend(["Baseline", "SARIMA", "MLP","CNN","RNN","LSTM","Bidirectional LSTM","CNN-LSTM Hybrid"],prop = { "size": 35 })
plt.title("Electricity Price Prediction Results",fontsize=60,**csfont)
#plt.savefig(r"C:\Users\Karthikeyan\Desktop\Thesis\Mid_Term_Presentation\Common_plots\metrics_Price.jpeg",format="jpeg",dpi=300)
plt.show()


#%% PV

base   =  [0.268, 0.538]
sarima =  [0.274, 0.434]
mlp    =  [0.124, 0.237]
cnn    =  [0.114, 0.229]
rnn    =  [0.126, 0.237]
lstm   =  [0.117,0.227]
bilstm =  [0.118,0.226]
cnnlstm=  [0.128,0.233]

x = np.arange(len(sarima))
width = 0.1
  
plt.figure(figsize=(40, 30))

# plot data in grouped manner of bar type

plt.bar(x-3*width, base, width, color='midnightblue')
plt.bar(x-2*width, sarima, width, color='mediumblue')
plt.bar(x-width, mlp, width, color='steelblue')
plt.bar(x, cnn, width, color='dodgerblue')
plt.bar(x+width, rnn, width, color='lavender')
plt.bar(x+2*width, lstm, width, color='royalblue')
plt.bar(x+3*width, bilstm, width, color='cornflowerblue')
plt.bar(x+4*width, cnnlstm, width, color='lightsteelblue')
plt.yticks(fontsize=45,**csfont)
plt.xticks(fontsize=45,**csfont)
plt.xticks(x, ['MAE', 'RMSE'])
plt.xlabel("ERROR METRICS",fontsize=50,**csfont)
plt.ylabel("Scores (kW)",fontsize=50,**csfont)
plt.legend(["Baseline", "SARIMA", "MLP","CNN","RNN","LSTM","Bidirectional LSTM","CNN-LSTM Hybrid"],prop = { "size": 35 })
plt.title("PV Prediction Results",fontsize=60,**csfont)
#plt.savefig(r"C:\Users\Karthikeyan\Desktop\Thesis\Mid_Term_Presentation\Common_plots\metrics_PV.jpeg",format="jpeg",dpi=300)
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
fig,a2=plt.subplots(figsize=(12,7))
a2=sns.distplot(df['Load'])
a2.set_xlabel("Load (Kw)",fontsize=24,**csfont)
a2.set_ylabel("Density",fontsize=24,**csfont)
plt.xlim(0,2.5)
plt.yticks(fontsize=20,**csfont)
plt.xticks(fontsize=20,**csfont)
plt.tight_layout()
plt.savefig(r"C:\Users\Karthikeyan\Desktop\Github\Master-Thesis\Main_Folder\13_Plots\Thesis_Report\EDA\den_load.jpeg",format="jpeg",dpi=1000)
plt.show()

#%%

fig,a2=plt.subplots(figsize=(12,7))
a2=sns.distplot(df['RTP'])
a2.set_xlabel("Price (\u20AC/Kw)",fontsize=24,**csfont)
a2.set_ylabel("Density",fontsize=24,**csfont)
plt.xlim(0,2.5)
plt.yticks(fontsize=20,**csfont)
plt.xticks(fontsize=20,**csfont)
plt.tight_layout()
plt.savefig(r"C:\Users\Karthikeyan\Desktop\Github\Master-Thesis\Main_Folder\13_Plots\Thesis_Report\EDA\den_price.jpeg",format="jpeg",dpi=1000)
plt.show()

#%%
fig,a2=plt.subplots(figsize=(12,7))
a2=sns.distplot(df['PV'])
a2.set_xlabel("PV Generated (Kw)",fontsize=24,**csfont)
a2.set_ylabel("Density",fontsize=24,**csfont)
plt.xlim(0,1.5)
plt.yticks(fontsize=20,**csfont)
plt.xticks(fontsize=20,**csfont)
plt.tight_layout()
plt.savefig(r"C:\Users\Karthikeyan\Desktop\Github\Master-Thesis\Main_Folder\13_Plots\Thesis_Report\EDA\den_pv.jpeg",format="jpeg",dpi=1000)
plt.show()

#%% Heatmap
from Essential_functions import corr_heat_map
plt.subplots(figsize=(20,11))
corr_heat_map(df)
plt.savefig(r"C:\Users\Karthikeyan\Desktop\Results\Report\heat_map.jpeg",format="jpeg",dpi=1000)
plt.show()
