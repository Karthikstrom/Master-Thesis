# -*- coding: utf-8 -*-
"""
Created on Tue Mar 14 15:24:23 2023

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
import matplotlib.dates as mdates

current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

from Essential_functions import load_wholedata
from Cost_analysis import COE
from Battery_Functions import pb_in_func,pb_out_func,SOC

#%% Read data

df= load_wholedata()
df=df[df.index.year==2019]


#adding pv penetration 
df['PV']=7*df['PV']
#%% Battery specifications

#battery capacity
E_b=6.6

#grid limit
ps_max=3.2

#because time step is one hour?
h=1 

#Typical efficiencies/room for improvement
eff_imp=1
eff_exp=1

#Soc limits/ RFI
soc_max=0.9
soc_min=0.1

#Battery charging and discharging limits
#From tesla powerwall 
pb_max=2.8
pb_min=2.8
#%% Initianlizing control variables/Arbitary values


X1=0.299 #mean of differences from each day/ could be dymanic- each month
X2=10/100 #percentage of charging when price is the lowest
X3=10/100 #percentage of dischargin when price is the highest
#%% Control Stratgy

df[['pb_in','pb_out','pp','p_imp','p_exp','soc']]=0

#groups by date and returns a tuple with data and the dataframe
data_day= df.groupby(df.index.date)
daily_difference=[]
daily_index=[]
operation=[]
df_list=[]
x2_limits=[]
x3_limits=[]


soc=0.7

for date in data_day:
    temp_data=date[1] #returns a dataframe for each day
    pr_max=temp_data['RTP'].max()
    pr_min=temp_data['RTP'].min()
    pr_diff=pr_max-pr_min
    x2_limit=pr_min+(pr_diff*X2)
    x3_limit=pr_max-(pr_diff*X3)
    
    
    #appending limit values
    x2_limits.append(x2_limit)
    x3_limits.append(x3_limit)
    
    
    
    if pr_diff>X1: #if is higher than the limit 
        operation.append(1)#counter for how many days its functioning
        for index,row in temp_data.iterrows():
            
            #getting the avilable input and output power of the battery
            pb_in = pb_in_func(pb_max,E_b,soc,soc_max)
            pb_out= pb_out_func(pb_max,E_b,soc,soc_min)
            
            #appending available input and output power of the battery
            temp_data.at[index, 'pb_in'] = pb_in
            temp_data.at[index, 'pb_out'] = pb_out
            
            price=temp_data.at[index,'RTP']
            load=temp_data.at[index,'Load']
            
            if price<x2_limit:
                p_imp=pb_in
                p_exp=0
                pp=load+p_imp
                
            else:
                if price>x3_limit:
                    if pb_out>load:
                        p_imp=0
                        p_exp=load
                        pp=0
                    else:
                        p_exp=pb_out
                        p_imp=0
                        pp=load-p_exp
                
                else:
                    p_imp=0
                    p_exp=0
                    pp=load
            soc=SOC(soc,p_imp,p_exp,E_b)
            temp_data.at[index,'pp']=pp
            temp_data.at[index,'p_imp']=p_imp
            temp_data.at[index,'p_exp']=p_exp
            temp_data.at[index,'soc']=soc
                        
    else:
        operation.append(0)
        for index,row in temp_data.iterrows():
            
            #getting the avilable input and output power of the battery
            pb_in = pb_in_func(pb_max,E_b,soc,soc_max)
            pb_out= pb_out_func(pb_max,E_b,soc,soc_min)
            
            #appending available input and output power of the battery
            temp_data.at[index, 'pb_in'] = pb_in
            temp_data.at[index, 'pb_out'] = pb_out
            
            #updating values
            p_exp=0
            p_imp=0
            
            soc=SOC(soc,p_imp,p_exp,E_b)
            temp_data.at[index,'soc']=soc
            temp_data.at[index,'pp']=temp_data.at[index,'Load']
            temp_data.at[index,'p_imp']=p_imp
            temp_data.at[index,'p_exp']=p_exp
            

    #appending values 
    daily_difference.append(pr_diff)
    daily_index.append(date[0])
    df_list.append(temp_data)

df_daily=pd.DataFrame({'diff':daily_difference,'date':daily_index,'oper':operation,'x2_limits':x2_limits,'x3_limits':x3_limits})
df_daily.set_index('date',inplace=True)
df_final=pd.concat(df_list)
#%% Plotting 
fig,ax=plt.subplots()
ax.plot(df_daily['diff'].rolling(24).mean())
plt.title("Rolling mean of Daily price difference")
plt.ylabel("Price difference(Euro/Kwh)")
#plt.savefig("price_diff_s2.jpeg",dpi=500)
plt.plot()

print("No of days the battery is in operation:",df_daily['oper'].value_counts()[1])
print("No of days the battery is not used:",df_daily['oper'].value_counts()[0])

fig,bx=plt.subplots()
bx.plot(df_final['soc'].iloc[1500:1720])
bx.xaxis.set_major_formatter(mdates.DateFormatter('%b\n%d'))
plt.title("State of charge of the battery- Strategy 2")
plt.ylabel("SOC")
plt.savefig("soc_s2_zoomed.jpeg",dpi=500)
plt.plot()

#Cost analysis
df_final['Arbitrage']=(df_final['Load']-df_final['pp'])*df_final['RTP']

zero= [0]*len(df_final)

rtp_coe=COE(df_final['Load'],zero,df_final['RTP'],df_final['RTP'])

fig,cx=plt.subplots()
cx.plot(df_final['pp'])
cx.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
plt.title("Power purchased- Strategy 2")
plt.ylabel("Power (KW)")
plt.savefig("pp_s2.jpeg",dpi=500)
plt.plot()