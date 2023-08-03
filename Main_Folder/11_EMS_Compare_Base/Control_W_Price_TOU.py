# -*- coding: utf-8 -*-
"""
Created on Tue Mar  7 13:25:04 2023

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

from KPIs import EMS_KPI

current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

from Essential_functions import load_wholedata
from Cost_analysis import COE
import matplotlib.dates as mdates

"""
 -after making the function could use itertools to loop and find the optimum 
values
 - puttting conditions in a array for bigger ones
 - use select when each condition has only one output/ create multiple selects
  (np.select)
"""

#%% Read data
df= load_wholedata()
df=df[df.index.year==2019]

#adding pv penetration and calculating mismatch
#df['Load']=df['Load']*50
df['PV']=7*df['PV']
#df['Mismatch']=df['PV']-df['Load']

#%%
"""
-------------------------------------------------------------------------------
"""
"""
lambda is a keyword in Python that is used to define anonymous 
functions, also known as lambda functions. A lambda function is 
a small, one-line function that can take any number of arguments,
 but can only have one expression.
 
Pb_max - Maximum available power of the battery---->?

"""
#Battery parameters

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


# Available input power & output power of the battery

def pb_in_func(pb_max,E_b,soc,soc_max):
    pb_in_temp=min(pb_max,(E_b/h)*(soc_max-soc))
    #to take the charge efficiency into consideration
    pb_in_temp=eff_imp*pb_in_temp
    return pb_in_temp

def pb_out_func(pb_max,E_b,soc,soc_min):
    pb_out_temp=min(pb_min,(E_b/h)*(soc-soc_min))
    #to take the discharge efficiency into consideration
    pb_out_temp=eff_exp*pb_out_temp
    return pb_out_temp

#calculate SOC
def SOC(soc_last,pb_imp,pb_exp):
    soc_temp=soc_last + (((pb_imp*eff_imp)-(pb_exp/eff_exp))/(E_b/h))# double check
    return soc_temp
    

# Function to compute output with one time step as input not vectorized
def PV_BES1(pv,load,soc,pb_in,pb_out,price):
    
    
    #Excess flow
    if pv>=load: 
        excess=pv-load
        #Check battery capacity
        if pb_in>excess:
            #After satisfying the load and charging the battery
            # we have some capacity in the battery
            #do we charge or not
            if price<0.6:
                #Import
                p_imp_solar=excess
                p_imp_grid=pb_in-excess
                p_exp_load=0
                p_exp_grid=0
                ps=0
                pp=p_imp_grid
                p_imp=p_imp_solar+p_imp_grid
                p_exp=p_exp_load+p_exp_grid
            
            else:
                #No price based action
                p_imp_solar=excess
                p_imp_grid=0
                p_exp_load=0
                p_exp_grid=0
                ps=0
                pp=0
                p_imp=p_imp_solar+p_imp_grid
                p_exp=p_exp_load+p_exp_grid
            
        else:
            p_imp_solar=pb_in
            p_imp_grid=0
            p_exp_load=0
            p_exp_grid=excess-p_imp_solar
            p_imp=p_imp_solar+p_imp_grid
            p_exp=p_exp_load+p_exp_grid
            pp=0
            ps=p_exp_grid
            
            
    #Deficit floe
    else:
        deficit=load-pv
        #Check battery capacity
        if pb_out>deficit:
            #Now we have battery capacity more than the deficit so should we
            #export or not? check the price range
            if price>0.6:
                #we export
                p_exp_load=deficit
                p_exp_grid=pb_out-p_exp_load
                p_imp_solar=0
                p_imp_grid=0
                p_exp=p_exp_load+p_exp_grid
                p_imp=p_imp_solar+p_imp_grid
                ps=p_exp_grid
                pp=0
            else:
                #No price based action
                #i.e no exporting
                p_exp_load=deficit
                p_exp_grid=0
                p_imp_solar=0
                p_imp_grid=0
                p_exp=p_exp_load+p_exp_grid
                p_imp=p_imp_solar+p_imp_grid
                ps=0
                pp=0
            
        else:
            #If we dont have the capacity to satisfy the load
            p_exp_load=pb_out
            p_exp_grid=0
            p_imp_solar=0
            p_imp_grid=deficit-p_exp_load
            p_exp=p_exp_load+p_exp_grid
            p_imp=p_imp_solar+p_imp_grid
            ps=0
            pp=p_imp_grid
            
    return pp,p_exp,p_imp,ps,p_exp_grid,p_exp_load,p_imp_solar,p_imp_grid
    

#%% PV-BES

#intial soc value
soc=0.7

#intializing needed columns to zero to append the values later
df[['pb_in','pb_out','ps','pp','p_imp','p_exp','soc']]=0.0000

#iterrows iterates through each row separately
for index, row in df.iterrows():
    #isolating values from each row
    pv=row['PV']
    load=row['Load']
    price=row['TOU']
    
    #getting the avilable input and output power of the battery
    pb_in = pb_in_func(pb_max,E_b,soc,soc_max)
    pb_out= pb_out_func(pb_max,E_b,soc,soc_min)
    
    #appending available input and output power of the battery
    df.at[index, 'pb_in'] = pb_in
    df.at[index, 'pb_out'] = pb_out
    
    #control strategy
    pp,p_exp,p_imp,ps,p_exp_grid,p_exp_load,p_imp_solar,p_imp_grid=PV_BES1(pv,load,soc,pb_in,pb_out,price)
    
    #appending the output from control strategy
    
    df.at[index,'pp']=pp
    df.at[index,'p_exp']=p_exp
    df.at[index,'p_imp']=p_imp
    df.at[index,'ps']=ps
    df.at[index,'p_exp_load']=p_exp_load
    df.at[index,'p_exp_grid']=p_exp_grid
    df.at[index,'p_imp_solar']=p_imp_solar
    df.at[index,'p_imp_grid']=p_imp_grid
    
     
    #update SOC
    soc=SOC(soc,p_imp,p_exp)

    #appending soc values to the df
    df.at[index,'soc']=soc

#%% Cost Analysis

EMS_KPI(df['pp'],df['ps'],df['TOU'])