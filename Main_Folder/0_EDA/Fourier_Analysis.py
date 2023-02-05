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

from Essential_functions import load_data2

from numpy.fft import rfft, irfft, rfftfreq
from scipy import pi, signal, fftpack
from scipy.signal import find_peaks

#%% Read data
df=load_data2()
#%%
n = len(df1['diff2'])
power_ft = np.abs(rfft(df1['diff2']))
power_freq = rfftfreq(n)

plt.figure(figsize=(10, 7))
plt.plot(power_freq[2:], power_ft[2: ])
    
plt.xlabel('frequency (1/hour)')
plt.show()

#%%

tmp = pd.DataFrame({'freqency [1/hour]':power_freq[2: ], 'y':power_ft[2: ]})
tmp['period [hours]'] = (1/tmp['freqency [1/hours]'])

tmp.sort_values(by=['y'], ascending=False).head()
#%%
# Let's find the peaks with height_threshold >=0.05
# Note: We use the magnitude (i.e the absolute value) of the Fourier transform

height_threshold=600 # We need a threshold. 


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
