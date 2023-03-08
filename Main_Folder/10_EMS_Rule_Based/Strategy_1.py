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
#%% Control strategy functions

def PV_only(pv,load,rtp,tou,ps_max,pv_p):    
    df=pd.DataFrame({'PV':pv,'Load':load,'RTP':rtp,'TOU':tou})
    df['PV']=df['PV']*pv_p
    df['Purchased']=np.where(df['PV']>=df['Load'],0,df['Load']-df['PV'])
    df['Sold'] = np.where(df['PV'] >= df['Load'],np.where((df['PV'] - df['Load']) >= ps_max, ps_max, (df['PV'] - df['Load'])),0)
    df['Dumped']= np.where(df['PV'] >= df['Load'],np.where((df['PV'] - df['Load']) >= ps_max, df['PV']-df['Load']-ps_max, 0),0)
    df['Arbitrage']=df['Purchased']-df['Sold']
    df['TOU_Scenario']=df['Arbitrage']*df['TOU']
    df['RTP_Scenario']=df['Arbitrage']*df['RTP']
    TOU_Sum=df['TOU_Scenario'].sum()
    RTP_Sum=df['RTP_Scenario'].sum()
    print("TOU total cost: {:.2f}".format(df['TOU_Scenario'].sum()),'\u20AC')
    print("RTP total cost: {:.2f}".format(df['RTP_Scenario'].sum()),'\u20AC')
    return TOU_Sum

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
h=1 # because time step is one hour?
eff_imp=0.9 
eff_exp=0.8
pb_max=5 #-----??
soc_max=0.9
soc_min=0.1
E_b=100
ps_max=3


# Available input power of the battery
# shouldn't this be maximum input power of the battery? pb_min?
pb_in_func = lambda pb_max,E_b,soc,soc_max: min(pb_max,(E_b/h)*(soc_max-soc))
# Available output power of the battery
pb_out_func= lambda pb_max,E_b,soc,soc_min: min(pb_max,(E_b/h)*(soc-soc_min))

def SOC(soc_last,pb_imp,pb_exp,eff_imp,eff_exp):
    soc_temp=soc_last + ((pb_imp*eff_imp)-(pb_exp/eff_exp))/(E_b/h)
    return soc_temp

# Function to compute output with one time step as input not vectorized
def PV_BES(pv,load,soc,pb_im,pb_out):
    
    #conditions
    c1= pv>=load
    c2= pv-load>=pb_in
    c3= (pv-load-pb_in)>=ps_max
    c4= pb_out>=load-pv
    
    
    #choices
    ch1= ps_max
    ch2= pv-load-pb_im-ps_max
    ch3= pv-load-pb_im
    ch4= pv-load
    ch5= pv+pb_exp-load
    ch6= load-pv

    return ps,pp,pd,p_imp,p_exp

#%% PV only
grid_limit=1 # max sent back to grid is 0.077KW
pv_penetration=5 # change this to increse in 10% steps
df_pvo=PV_only(df['PV'],df['Load'],df['RTP'],df['TOU'],grid_limit,pv_penetration)

#%% finding optimum values

#setting search iterations
grid_limit_values=[0.5,1,1.5,2,2.5,3]
pv_penetration_values=[1,2,3,4,5,6,7,8,9,10]

#itertools returns cartestian pairs of the above values
input_values=itertools.product(grid_limit_values,pv_penetration_values)

#computes total cost for each pair
output_values=[PV_only(df['PV'],df['Load'],df['RTP'],df['TOU'],x,y) for x, y in input_values]

#creating a dataframe with iterations and their total value
inpv=itertools.product(grid_limit_values,pv_penetration_values)
zipped_values=zip(inpv,output_values)
df_final=pd.DataFrame(zipped_values,columns=['iter','Total_cost'])
df_final[['grid_limit', 'pv_penetration']] = df_final['iter'].apply(lambda x: pd.Series(x))
df_final.drop('iter',axis=1,inplace=True)

#%% PV-BES

#intial soc value
soc=0.2

#intializing needed columns to zero to append the values later
df[['pb_in','pb_out','ps','pp','pd','p_imp','p_exp','soc']]

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
    ps,pp,pd,p_imp,p_exp=PV_BES(pv,load,soc,pb_in,pb_out)
    
    #appending the output from control strategy
    df.at[index,'ps']=ps
    df.at[index,'pp']=pp
    df.at[index,'pd']=pd
    df.at[index,'p_imp']=p_imp
    df.at[index,'p_out']=p_exp
        
    #update SOC
    soc=SOC(soc,pb_imp,pb_exp,eff_imp,eff_exp)
    
    #appending soc values to the df
    df.at[index,'soc']=soc




