 # -*- coding: utf-8 -*-
"""
Created on Wed Jan  4 01:39:34 2023

@author: Karthikeyan
"""

#%% Loading packages
import pandas as pd
import numpy as np 
import seaborn as sns
import matplotlib.pyplot as plt

import os
import sys
import path

current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

from Essential_functions import load_data2,load_data

from numpy.fft import rfft, irfft, rfftfreq
from scipy import pi, signal, fftpack
from scipy.signal import find_peaks

#%% Read data
df1=load_data()
#%%
n = len(df1['Load'])
power_ft = np.abs(rfft(df1['Load']))
power_freq = rfftfreq(n)

plt.figure(figsize=(10, 7))
plt.plot(power_freq[2:], power_ft[2: ])
    
plt.xlabel('frequency (1/hour)')
plt.show()
#%%
# Let's find the peaks with height_threshold >=0.05
# Note: We use the magnitude (i.e the absolute value) of the Fourier transform

height_threshold=200000 # We need a threshold. 


# peaks_index contains the indices in x that correspond to peaks:

peaks_index, properties = find_peaks(np.abs(power_ft), height=height_threshold)

op=pd.DataFrame()
op['index']=peaks_index
op['freq']=power_freq[op['index']]
op['power']=properties['peak_heights']
op['freq_to_hours']=1/op['freq']
op['freq_to_days']=(1/op['freq'])/24
op.drop(['index'],axis=1,inplace=True)
op.sort_values(by='power', ascending=False)
#%%
print('Positions and magnitude of frequency peaks:')
[print("%4.4f    \t %3.4f" %(power_freq[peaks_index[i]], properties['peak_heights'][i])) for i in range(len(peaks_index))]

#%%

plt.plot(power_freq, np.abs(power_ft),'-', power_freq[peaks_index],properties['peak_heights'],'x')
plt.xlabel("Frequency")
plt.ylabel("Amplitude")
plt.show()


"""
If a signal has a high magnitude frequency and half of that frequency has 
a high magnitude, it could indicate that the signal is a harmonic signal, 
which means that it is composed of multiple sinusoidal waves with frequencies 
that are integer multiples of a fundamental frequency.

In this case, the high magnitude frequency could be the fundamental frequency,
 and the frequency with half of that magnitude could be the second harmonic 
 frequency (i.e., twice the fundamental frequency).

Alternatively, the signal could also be a modulated signal, where the high
 magnitude frequency represents the carrier signal and the frequency with half 
 of that magnitude represents the modulation signal. This would indicate that
 the signal has been modulated using amplitude modulation, where the amplitude
 of the carrier signal varies in proportion to the amplitude of the modulation 
 signal.

"""
