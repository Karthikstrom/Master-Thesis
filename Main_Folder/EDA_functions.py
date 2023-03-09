# -*- coding: utf-8 -*-
"""
Created on Wed Mar  8 21:38:18 2023

@author: Karthikeyan
"""

#%% Loading packages
import pandas as pd
import numpy as np 
import seaborn as sns
import matplotlib.pyplot as plt

from Essential_functions import load_data

from numpy.fft import rfft, irfft, rfftfreq
from scipy import pi, signal, fftpack
from scipy.signal import find_peaks

#%% Statistic Functions

def Fourier_Analysis(data):
    n = len(data)
    power_ft = np.abs(rfft(data))
    power_freq = rfftfreq(n)
    
    height_threshold=20000 
    
    peaks_index, properties = find_peaks(np.abs(power_ft), height=height_threshold)

    op=pd.DataFrame()
    op['index']=peaks_index
    op['freq']=power_freq[op['index']]
    op['power']=properties['peak_heights']
    op['freq_to_hours']=1/op['freq']
    op['freq_to_days']=(1/op['freq'])/24
    op.drop(['index'],axis=1,inplace=True)
    op.sort_values(by='power', ascending=False)
    
    print('Positions and magnitude of frequency peaks:')
    [print("%4.4f    \t %3.4f" %(power_freq[peaks_index[i]], properties['peak_heights'][i])) for i in range(len(peaks_index))]

    plt.plot(power_freq, np.abs(power_ft),'-', power_freq[peaks_index],properties['peak_heights'],'x')
    plt.xlabel("Frequency")
    plt.ylabel("Amplitude")
    plt.show()
    
    return op