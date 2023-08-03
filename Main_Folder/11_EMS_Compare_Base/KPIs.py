# -*- coding: utf-8 -*-
"""
Created on Wed Apr 19 10:09:41 2023

@author: Karthikeyan
"""

#%% Importing packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#%% Functions

def EMS_KPI(pp,ps,price):
    df=pd.DataFrame()
    df['pp']=pp
    df['ps']=ps
    df['price']=price
    
    
    #Total grid purchase
    total_electricity_consumption=df['pp'].sum()
    print("Total Electricity Purchased:",total_electricity_consumption,"Kwh")
    
    #Total sold to grid
    total_electricity_sold=df['ps'].sum()
    print("Total Electricity Sold:",total_electricity_sold,"Kwh")
    
    #Net Electricity exchange
    net_electricity_exchange=(df['pp']-df['ps']).sum()
    print("Net Electricity Exchange:",net_electricity_exchange,"Kwh")
    
    #Total cost
    df['e_cost']=(df['pp']-df['ps'])*df['price']
    total_electricity_cost= df['e_cost'].sum()
    print("Net Electricity Cost:",total_electricity_cost,"Euros")
    
    
    #Average daily peak
    daily_peak=(df['pp']+df['ps']).groupby(df.index.strftime('%m-%d')).max()
    average_daily_peak=daily_peak.mean()
    print("Average Daily Peak:",average_daily_peak,"Kwh")
    
    #Ramping
    #So there is ramp up and ramp down
    df['ramp']=(df['pp']-df['pp'].shift(1)).abs()
    ramp=df['ramp'].mean()
    print("Total Ramping:",ramp,"Kwh")
    
    
    #Load factor
    daily_average=(df['pp']+df['ps']).groupby(df.index.strftime('%m-%d')).mean()
    load_factor=daily_average/daily_peak
    load_factor=load_factor.mean()
    print("Load Factor:",load_factor)
    
