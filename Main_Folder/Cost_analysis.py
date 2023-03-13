# -*- coding: utf-8 -*-
"""
Created on Fri Mar 10 23:20:11 2023

@author: Karthikeyan
"""

#%% Loading packages

import pandas as pd
import numpy as np

#%% Functions

def Maintenance(annual_maintenance,interest_rate,life_time):
    Maintenance=annual_maintenance*(((1+interest_rate)**life_time-1)/interest_rate*(1+interest_rate)**life_time)
    return Maintenance

def Replacement(replacement_cost,component_lifetime,life_time,interest_rate):
    replace=0
    for i in range(10):
        if (component_lifetime*i)<life_time:
            replace=replace + (1/(1+interest_rate)**i*component_lifetime)
    return replace

def COE(pp,ps,rtp,fit):
    
    life_time=20
    interest_rate=0.06
    inflation_rate=0.02
    
    real_interest_rate=(interest_rate-inflation_rate)/(1+inflation_rate)
    
    
    #NPC of components
    bat_num=1
    pv_num=12
    
    #For one battery
    bat_principal_cost=1000
    bat_annual_maintenance=100
    bat_replacement_cost=600
    bat_lifetime=7
    
    bat_maintenance=Maintenance(bat_annual_maintenance,interest_rate,life_time)
    bat_replacement=Replacement(bat_replacement_cost,bat_lifetime,life_time,interest_rate)
    
    bat_npc=bat_principal_cost+bat_maintenance+bat_replacement
    
    #For one PV
    pv_principal_cost=200
    pv_annual_maintenance=20
    pv_replacement_cost=100
    pv_lifetime=20
    
    pv_maintenance=Maintenance(pv_annual_maintenance,interest_rate,life_time)
    pv_replacement=Replacement(pv_replacement_cost,pv_lifetime,life_time,interest_rate)
    
    pv_npc=pv_principal_cost+pv_maintenance+pv_replacement
    
    bat_total=bat_num*bat_npc
    pv_total=pv_num*pv_npc
    
    component_npc=bat_total+pv_total
    
    elec_df=pd.DataFrame()
    elec_df['rtp']=rtp
    elec_df['fit']=fit
    elec_df['ps']=ps
    elec_df['pp']=pp
    
    
    elec_df['npc']=(elec_df['pp']*elec_df['rtp'])-(elec_df['fit']*elec_df['ps'])

    elec_npc=elec_df['npc'].sum()
    
    total_npc=component_npc+elec_npc
    
    #Cost of electricity (Euro/Kwh)
    #To compate multiple system configurations
    
    #Capital recovery factor
    #Gives the amount to be paid every year
    # yearly payouts of the NPC 
    
    crf_s=(interest_rate*(1+interest_rate)**life_time)/(((1+interest_rate)**life_time)-1)
    crf_e=(real_interest_rate*(1+real_interest_rate)**life_time)/(((1+real_interest_rate)**life_time)-1)
    
    E_annual=rtp.sum()
    
    Cost_of_elec=((component_npc*crf_s)+(elec_npc*crf_e))/E_annual
    
    return Cost_of_elec