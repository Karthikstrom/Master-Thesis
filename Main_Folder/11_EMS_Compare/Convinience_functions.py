# -*- coding: utf-8 -*-
"""
Created on Wed Apr 19 22:05:05 2023

@author: Karthikeyan
"""

#%% Importing package
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme()

#%% Functions

def KPI_compare(baseline,pv_only,wo_price,w_price):
    
    #Normalizing
    pv_only=pv_only/baseline
    wo_price=wo_price/baseline
    w_price=w_price/baseline
    
    #height = [pv_only,wo_price,w_price]
    height= [w_price,wo_price,pv_only]
    #bars = ('Control W/O price signal\n Only PV', 'Control W/O price signal\n BES & PV', 'Control with price signal \n BES & PV ')
    bars = ('Control with price signal \n BES & PV ','Control W/O price signal\n BES & PV', 'Control W/O price signal\n Only PV' )
    y_pos = np.arange(len(bars))
     
    # Create horizontal bars
    plt.barh(y_pos, height)
     
    # Create names on the x-axis
    plt.yticks(y_pos, bars)
    plt.axvline(x = 1, color = 'black',label = 'Baseline',linestyle='--')
    # Show graphic
    plt.legend()
    plt.xlim([0, 2])
    plt.title("RTP- Total Electricity purchased (Kwh)")
    plt.savefig(r"C:\Users\Karthikeyan\Desktop\Thesis presentations\Presentation_2_Plots\RTP_AR.jpeg",dpi=500)
    plt.show()


def KPI_compare2(baseline,pv_only,wo_price,w_price):
    
    #Normalizing
    # pv_only=pv_only/baseline
    # wo_price=wo_price/baseline
    # w_price=w_price/baseline
    
    #height = [pv_only,wo_price,w_price]
    height= [w_price,wo_price,pv_only]
    #bars = ('Control W/O price signal\n Only PV', 'Control W/O price signal\n BES & PV', 'Control with price signal \n BES & PV ')
    bars = ('Control with price signal \n BES & PV ','Control W/O price signal\n BES & PV', 'Control W/O price signal\n Only PV' )
    y_pos = np.arange(len(bars))
     
    # Create horizontal bars
    plt.subplots(figsize=(15,5))
    plt.barh(y_pos, height)
     
    # Create names on the x-axis
    plt.yticks(y_pos, bars)
    plt.axvline(x = baseline, color = 'black',label = 'Baseline',linestyle='--')
    # Show graphic
    plt.legend()
    #plt.xlim([0, 2])
    plt.title("TOU- Load Factor")
    plt.savefig(r"C:\Users\Karthikeyan\Desktop\Thesis presentations\Presentation_2_Plots\TOU_LF.jpeg",dpi=500)
    plt.show()