# -*- coding: utf-8 -*-
"""
Created on Tue Apr 18 21:46:57 2023

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
X1=0 #mean of differences from each day/ could be dymanic- each month
X2=30/100 #percentage of charging when price is the lowest
X3=30/100 #percentage of dischargin when price is the highest
#%% Control Stratgy

df[['pb_in','pb_out','pp','ps','p_imp','p_exp','p_exp_l','p_exp_g','soc','counter']]=0.0000

#groups by date and returns a tuple with data and the dataframe
data_day= df.groupby(df.index.date)
daily_difference=[]
daily_index=[]
operation_on=[]
operation_off=[]
df_list=[]
x2_limits=[]
x3_limits=[]


soc=0.7

for date in data_day:
    #returns a dataframe for each day
    temp_data=date[1] 
    pr_max=temp_data['RTP'].max()
    pr_min=temp_data['RTP'].min()
    pr_diff=pr_max-pr_min
    x2_limit=pr_min+(pr_diff*X2)
    x3_limit=pr_max-(pr_diff*X3)
    
    
    #appending limit values
    x2_limits.append(x2_limit)
    x3_limits.append(x3_limit)
    
    
    
    if pr_diff>X1: #if is higher than the limit 
    
        #counter for how many days its functioning
        operation_on.append(1)
        
        
        #Iterating for each day
        for index,row in temp_data.iterrows():
            
            #getting the avilable input and output power of the battery
            pb_in = pb_in_func(pb_max,E_b,soc,soc_max)
            pb_out= pb_out_func(pb_max,E_b,soc,soc_min)
            
            #appending available input and output power of the battery
            temp_data.at[index, 'pb_in'] = pb_in
            temp_data.at[index, 'pb_out'] = pb_out
            
            #Extracting each value for time-step
            price=temp_data.at[index,'RTP']
            load=temp_data.at[index,'Load']
            pv=temp_data.at[index,'PV']
            
            
            #Checking if PV can satisfy load
            if pv>=load:
                #Excess flow
                
                
                #Low price based action
                if price<=x2_limit:
                    
                    #Checking battery capacity
                    if pb_in>(pv-load):
                        pp=pb_in-(pv-load)
                        p_imp=pp+(pv-load)
                        ps=0
                        p_exp_g=0
                        p_exp_l=0
                        counter=111
                    
                    else:
                        p_imp=pb_in
                        ps=(pv-load)-p_imp
                        pp=0
                        p_exp_g=0
                        p_exp_l=0
                        counter=112
                
                # High price based action
                elif price>=x3_limit:
                    counter=12
                    pp=0
                    p_imp=0
                    p_exp_l=0
                    p_exp_g=pb_out
                    ps=p_exp_g+(pv-load)
                    
                        
                # No price based action        
                else:
                    #Checking battery capacity
                    
                    if pb_in>(pv-load):
                        p_imp=pv-load
                        pp=0
                        ps=0
                        p_exp_g=0
                        p_exp_l=0
                        counter=131
                    
                    else:
                        p_imp=pb_in
                        ps=(pv-load)-p_imp
                        pp=0
                        p_exp_g=0
                        p_exp_l=0
                        counter=132
            
            
            
            else:
                #Deficit flow
                
                #Low price based action
                if price<=x2_limit:
                    
                    #Checking battery capacity
                    if pb_out>(load-pv):
                        p_exp_l=load-pv
                        p_exp_g=0
                        p_imp=0
                        pp=0
                        ps=0
                        counter=211
                    
                    else:
                        p_exp_l=pb_out
                        p_exp_g=0
                        pp=load-pv-p_exp_l
                        ps=0
                        p_imp=0
                        counter=212
                        
                #High price based action
                elif price>=x3_limit:
                    
                    #Battery check
                    if pb_out>(load-pv):
                        p_exp_l=load-pv
                        p_exp_g=pb_out-p_exp_l
                        p_imp=0
                        pp=0
                        ps=p_exp_g
                        counter=221
                        
                    else:
                        p_else_l=pb_out
                        p_exp_g=0
                        p_imp=0
                        ps=0
                        pp=load-pv-p_exp_l
                        counter=222
                        
                #No price based action        
                else:
                    
                    #Battery check
                    if pb_out>(load-pv):
                        p_exp_l=load-pv
                        p_exp_g=0
                        pp=0
                        ps=0
                        p_imp=0
                        counter=231
                    
                    else:
                        p_exp_l=pb_out
                        p_exp_g=0
                        ps=0
                        pp=load-pv-p_exp_l
                        p_imp=0
                        counter=232
    
    
            #Should update inside the for loop
            
            #Total exported/discharged from battery
            p_exp=p_exp_l+p_exp_g
            
            #Update SOC
            soc=SOC(soc,p_imp,p_exp,E_b)
            
            temp_data.at[index,'counter']=counter
            temp_data.at[index,'pp']=pp
            temp_data.at[index,'ps']=ps
            temp_data.at[index,'p_imp']=p_imp
            temp_data.at[index,'p_exp']=p_exp
            temp_data.at[index,'p_exp_l']=p_exp_l
            temp_data.at[index,'p_exp_g']=p_exp_g
            temp_data.at[index,'soc']=soc
            
    
    # If it does not satisfy X1 criteria
    else:
        #Iterating for each day
        operation_off.append(1)
        for index,row in temp_data.iterrows():
            
            #getting the avilable input and output power of the battery
            pb_in = pb_in_func(pb_max,E_b,soc,soc_max)
            pb_out= pb_out_func(pb_max,E_b,soc,soc_min)
            
            #appending available input and output power of the battery
            temp_data.at[index, 'pb_in'] = pb_in
            temp_data.at[index, 'pb_out'] = pb_out
            
            #Extracting each value for time-step
            price=temp_data.at[index,'RTP']
            load=temp_data.at[index,'Load']
            pv=temp_data.at[index,'PV']
        
        
        #Checking if PV can satisfy load
        # if pv>=load:
        #     #temp_data.at[index,'counter']=6
        # else:
        #     #temp_data.at[index,'counter']=6
            
    #appending values 
    daily_difference.append(pr_diff)
    daily_index.append(date[0])
    df_list.append(temp_data)

df_daily=pd.DataFrame({'diff':daily_difference,'date':daily_index,'x2_limits':x2_limits,'x3_limits':x3_limits})
df_daily.set_index('date',inplace=True)
df_final=pd.concat(df_list)


#%% Cost Analysis



print("Price_based control",df_final['COE_RTP'].sum(),"Euros")




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