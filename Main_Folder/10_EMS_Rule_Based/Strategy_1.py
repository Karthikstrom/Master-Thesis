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

current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

from Essential_functions import load_wholedata

"""
 -after making the function could use itertools to loop and find the optimum 
values
 - puttting conditions in a array for bigger ones
 - use select when each condition has only one output/ create multiple selects
  (np.select)
"""
#%% Read data
df= load_wholedata()
df= df[:168]

#adding pv penetration and calculating mismatch
df['PV']=7*df['PV']
df['Mismatch']=df['PV']-df['Load']

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
E_b=13.5

#grid limit
ps_max=3

#because time step is one hour?
h=1 

#Typical efficiencies/room for improvement
eff_imp=0.9
eff_exp=0.8

#Soc limits/ RFI
soc_max=0.9
soc_min=0.1

#Battery charging and discharging limits
#From tesla powerwall 
pb_max=5
pb_min=5 


# Available input power & output power of the battery
pb_in_func = lambda pb_max,E_b,soc,soc_max: min(pb_max,(E_b/h)*(soc_max-soc))

pb_out_func= lambda pb_max,E_b,soc,soc_min: min(pb_max,(E_b/h)*(soc-soc_min))


#calculate SOC
def SOC(soc_last,pb_imp,pb_exp):
    soc_temp=soc_last + (((pb_imp*eff_imp)-(pb_exp/eff_exp))/(E_b/h))# double check
    return soc_temp

#apply everytime charging or discharging the battery
def SOC_check(soc,pb_imp,pb_exp):
    soc_next=SOC(soc,pb_imp,pb_exp)
    if ((soc>=soc_min) ==True) & ((soc<=soc_max)== True):
        return True
    else:
        return False
    

# Function to compute output with one time step as input not vectorized
def PV_BES(pv,load,soc,pb_in,pb_out):
    
    #conditions
    c1= pv>=load
    c2= pv-load>=pb_in
    c3= (pv-load-pb_in)>=ps_max
    c4= pb_out>=load-pv
        
    if c1==False: # Deficit flow
        p_imp=0
        ps=0
        pd=0
        if c4==True:
            p_exp=load-pv
            pp=0
        else:
            p_exp=pb_out
            pp=load-pv+p_exp
            
    else: # Excess flow (c1==True)
        p_exp=0
        pp=0
        if c2==False:
            p_imp=pv-load
            ps=0
            pd=0
        else:
            p_imp=pb_in
            if c3==False:
                ps=pv-load-p_imp
                pd=0
            else:
                ps=ps_max
                pd=pv-load-pb_in-ps_max
        
    return pp,p_exp,p_imp,ps,pd
#%% PV-BES

#intial soc value
soc=0.7

#intializing needed columns to zero to append the values later
df[['pb_in','pb_out','ps','pp','pd','p_imp','p_exp','soc']]=0

#iterrows iterates through each row separately
for index, row in df.iterrows():
    
    #isolating values from each row
    pv=row['PV']
    load=row['Load']
    
    #getting the avilable input and output power of the battery
    pb_in = pb_in_func(pb_max,E_b,soc,soc_max)
    pb_out= pb_out_func(pb_max,E_b,soc,soc_min)
    
    #appending available input and output power of the battery
    df.at[index, 'pb_in'] = pb_in
    df.at[index, 'pb_out'] = pb_out
    
    #control strategy
    #ps,pp,pd,p_imp,p_exp=PV_BES(pv,load,soc,pb_in,pb_out)
    pp,p_exp,p_imp,ps,pd=PV_BES(pv,load,soc,pb_in,pb_out)
    
    #appending the output from control strategy
    
    df.at[index,'pp']=pp
    df.at[index,'p_exp']=p_exp
    df.at[index,'p_imp']=p_imp
    df.at[index,'ps']=ps
    df.at[index,'pd']=pd
     
    #appending soc values to the df
    df.at[index,'soc']=soc
    
    #update SOC
    soc=SOC(soc,p_imp,p_exp)




