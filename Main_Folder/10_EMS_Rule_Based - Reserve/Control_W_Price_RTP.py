# -*- coding: utf-8 -*-
"""
Created on Wed Apr 19 19:36:49 2023

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
eff_imp=0.9
eff_exp=0.9

#Soc limits/ RFI
soc_max=0.9
soc_min=0.1

#Battery charging and discharging limits
#From tesla powerwall 
pb_max=2.8
pb_min=2.8
#%% Initianlizing control variables/Arbitary values
X1=0 #mean of differences from each day/ could be dymanic- each month
X2=10/100 #percentage of charging when price is the lowest
X3=10/100 #percentage of dischargin when price is the highest
#%% Control Stratgy

df[['pb_in','pb_out','pp','ps','p_imp','p_exp','p_exp_l','p_exp_g','p_imp_s','p_imp_g','soc','counter']]=0.0

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
            
            
            #Excess flow
            if pv>=load:
                excess=pv-load
                
                #Checking if battery has capacity to store the excess
                if pb_in>excess:
                    #yes, battery has capacity to store excess and some more
                    #after storing the excess, do we need to import from grid?
                    #we decide by taking x2 limit into consideration
                    #it is only possible to charge as simultaneous charge and discharge is not allowed
                    
                    if price<x2_limit:
                        #import
                        p_imp_solar=excess
                        p_imp_grid=pb_in-p_imp_solar
                        p_exp_load=0
                        p_exp_grid=0
                        p_exp=0
                        ps=0
                        pp=p_imp_grid
                        counter=1
                        
                    else:
                        #Not importing/ No price based action
                        p_imp_solar=excess
                        p_imp_grid=0
                        p_exp_grid=0
                        p_exp_load=0
                        p_exp=0
                        ps=0
                        pp=0
                        counter=2
                else:
                    #No battery has no capacity to store excess
                    #Charging as much as it can handle
                    p_imp_solar=pb_in
                    p_imp_grid=0
                    p_exp_load=0
                    p_exp_grid=0
                    ps=excess-p_imp_solar
                    pp=0
                    counter=3
            
            #Deficit flow
            else:
                deficit=load-pv
                
                #Checking if battery has capacity to supply for the deficit
                if pb_out>deficit:
                    #yes,it can satisfy the deficit and have some extra as well
                    #should we sell the rest? check if the price is in X3 limit
                    #we can only discharge, as it is already in discharge mode
                    #not possible to charge and discharge at the same time-step
                    if price>x3_limit:
                        #export/sell
                        p_exp_load=deficit
                        p_exp_grid=pb_out-p_exp_load
                        p_imp_solar=0
                        p_imp_grid=0
                        p_imp=0
                        pp=0
                        ps=p_exp_grid
                        counter=4
                        
                    else:
                        #No exporting/price based action
                        p_exp_load=deficit
                        p_exp_grid=0
                        p_imp_solar=0
                        p_imp_grid=0
                        p_imp=0
                        pp=0
                        ps=0
                        counter=5
                else:
                   #if the battery does not have enough capacity to fill the deficit
                   #it has to be imported, since battery is being discharged
                   #no possibility of charging it
                   
                   #discharging max
                   p_exp_load=pb_out
                   p_exp_grid=0
                   p_imp_solar=0
                   p_imp_grid=0
                   ps=0
                   pp=deficit-p_exp_load
                   counter=5
                   
                   
                    
            
            #Update SOC
            p_exp=p_exp_load+p_exp_grid
            p_imp=p_imp_solar+p_imp_grid
            soc=SOC(soc,p_imp,p_exp,E_b)
            
            temp_data.at[index,'counter']=counter
            temp_data.at[index,'pp']=pp
            temp_data.at[index,'ps']=ps
            temp_data.at[index,'p_imp']=p_imp
            temp_data.at[index,'p_exp']=p_exp
            temp_data.at[index,'p_exp_l']=p_exp_load
            temp_data.at[index,'p_exp_g']=p_exp_grid
            temp_data.at[index,'p_imp_s']=p_imp_solar
            temp_data.at[index,'p_imp_g']=p_imp_grid
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
            
            
            if pv>=load: 
                excess=pv-load
                #Check battery capacity
                if pb_in>excess:
                    p_imp_solar=excess
                    p_imp_grid=0
                    p_exp_grid=0
                    p_exp_load=0
                    p_exp=0
                    ps=0
                    pp=0
                    counter=6
                
                else:
                    p_imp_solar=pb_in
                    p_imp_grid=0
                    p_exp_load=0
                    p_exp_grid=0
                    ps=excess-p_imp_solar
                    pp=0
                    counter=7
                    
            
            #Deficit flow
            else:
                deficit=load-pv
                #Check battery capacity
                if pb_out>deficit:
                    p_exp_load=deficit
                    p_exp_grid=0
                    p_imp_solar=0
                    p_imp_grid=0
                    p_imp=0
                    pp=0
                    ps=0
                    counter=8
                    
                else:
                     p_exp_load=pb_out
                     p_exp_grid=0
                     p_imp_solar=0
                     p_imp_grid=0
                     ps=0
                     pp=deficit-p_exp_load
                     counter=9
            
            
            temp_data.at[index,'counter']=counter
            temp_data.at[index,'pp']=pp
            temp_data.at[index,'ps']=ps
            temp_data.at[index,'p_imp']=p_imp
            temp_data.at[index,'p_exp']=p_exp
            temp_data.at[index,'p_exp_l']=p_exp_load
            temp_data.at[index,'p_exp_g']=p_exp_grid
            temp_data.at[index,'p_imp_s']=p_imp_solar
            temp_data.at[index,'p_imp_g']=p_imp_grid
            temp_data.at[index,'soc']=soc
            
            
            
    #appending values-out the first loop and inside the day loop
    daily_difference.append(pr_diff)
    daily_index.append(date[0])
    #appends dataframe from each day
    df_list.append(temp_data)

df_daily=pd.DataFrame({'diff':daily_difference,'date':daily_index,'x2_limits':x2_limits,'x3_limits':x3_limits})
df_daily.set_index('date',inplace=True)
df_final=pd.concat(df_list)

#%% Cost Analysis
from KPIs import EMS_KPI
EMS_KPI(df_final['pp'],df_final['ps'],df_final['RTP'])
