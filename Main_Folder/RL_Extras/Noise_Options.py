# -*- coding: utf-8 -*-
"""
Created on Mon Jun  5 16:00:24 2023

@author: Karthikeyan
"""
import numpy as np
import matplotlib.pyplot as plt

# Parameters for OU process
theta = 0.5  # Mean reversion parameter
mu = 0.0  # Mean of the noise
sigma = 0.2  # Standard deviation of the noise
dt = 0.01  # Time step
T = 100.0  # Total time

# Generate OU noise
t = np.arange(0, T, dt)
n = len(t)
x = np.zeros(n)
x[0] = np.random.normal(mu, sigma)
for i in range(1, n):
    dx = theta * (mu - x[i-1]) * dt + sigma * np.sqrt(dt) * np.random.normal()
    x[i] = x[i-1] + dx

# Generate Gaussian noise
gaussian_noise = np.random.normal(mu, sigma, n)

# Plotting
plt.figure(figsize=(10, 6))
plt.subplot(2, 1, 1)
plt.plot(t, x)
plt.title("Ornstein-Uhlenbeck Noise")
plt.xlabel("Time")
plt.ylabel("Noise Value")
plt.grid(True)

plt.subplot(2, 1, 2)
plt.plot(t, gaussian_noise)
plt.title("Gaussian Noise")
plt.xlabel("Time")
plt.ylabel("Noise Value")
plt.grid(True)

plt.tight_layout()
plt.show()

#%%
import numpy as np
import matplotlib.pyplot as plt

# Parameters for OU process
theta = 0.1 # Mean reversion parameter
mu = 0  # Mean of the noise
sigma_initial = 0.5  # Initial standard deviation of the noise
sigma_final = 0.01  # Final standard deviation of the noise
dt = 1  # Time step
T = 24 # Total time

# Generate OU noise with reducing sigma
t = np.arange(0, T, dt)
n = len(t)
x = np.zeros(T)
x[0] = np.random.normal(mu, sigma_initial)
for i in range(1, n):
    sigma = sigma_initial - (sigma_initial - sigma_final) * (i / n)  # Linearly reduce sigma
    dx = theta * (mu - x[i-1]) * dt + sigma * np.sqrt(dt) * np.random.normal()
    x[i] = x[i-1] + dx

# Plotting
plt.plot(t, x)
plt.title("Ornstein-Uhlenbeck Noise with Reducing Sigma")
plt.xlabel("Time")
plt.ylabel("Noise Value")
plt.grid(True)
plt.show()


#%%

import numpy as np
import matplotlib.pyplot as plt

# Parameters for Gaussian noise
mu = 0.0  # Mean of the noise
sigma_initial = 0.7  # Initial standard deviation of the noise
sigma_final = 0.7  # Final standard deviation of the noise
dt = 1  # Time step
T = 24  # Total time

# Generate Gaussian noise with reducing sigma
t = np.arange(0, T, dt)
n = len(t)
gaussian_noise = np.zeros(n)
for i in range(n):
    sigma = sigma_initial - (sigma_initial - sigma_final) * (i / n)  # Linearly reduce sigma
    gaussian_noise[i] = np.random.normal(mu, sigma)

# Plotting
plt.plot(t, gaussian_noise)
plt.title("Gaussian Noise with Reducing Sigma")
plt.xlabel("Time")
plt.ylabel("Noise Value")
plt.grid(True)
plt.show()