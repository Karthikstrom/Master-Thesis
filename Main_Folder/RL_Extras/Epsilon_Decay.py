# -*- coding: utf-8 -*-
"""
Created on Wed Jul 12 13:52:24 2023

@author: Karthikeyan
"""

import numpy as np
import matplotlib.pyplot as plt

def epsilon_decay(start_value, end_value, decay_rate, num_episodes):
    epsilons = []
    epsilon = start_value

    for episode in range(num_episodes):
        epsilon = max(end_value, epsilon * decay_rate)
        epsilons.append(epsilon)

    return epsilons

# Parameters
start_value = 1.0
end_value = 0.1
decay_rate = 0.9945
num_episodes = 600

# Decay epsilon
epsilon_values = epsilon_decay(start_value, end_value, decay_rate, num_episodes)

# Plot epsilon decay
plt.plot(epsilon_values)
plt.xlabel('Episode')
plt.ylabel('Epsilon')
plt.title('Epsilon Decay')
plt.show()