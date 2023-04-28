# -*- coding: utf-8 -*-
"""
Created on Fri Mar 10 14:40:11 2023

@author: Karthikeyan
"""

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

df['Mismatch']=df['PV']-df['Load']
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


#%% PV only
grid_limit=3 # max sent back to grid is 0.077KW
pv_penetration=6 # change this to increse in 10% steps
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
