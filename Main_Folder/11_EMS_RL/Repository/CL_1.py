# -*- coding: utf-8 -*-
"""
Created on Fri Apr 14 19:16:45 2023

@author: Karthikeyan
"""

#%% Importing packages
import citylearn
import ipywidgets
import matplotlib
import seaborn
import stable_baselines3

# System operations
import os
import sys

# type hinting
from typing import List, Mapping, Tuple

# Data visualization
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns

# User interaction
from IPython.display import clear_output
from ipywidgets import Button, FloatSlider, HBox, HTML, IntProgress

# Data manipulation
import math
import numpy as np
import pandas as pd
import random

# CityLearn
from citylearn.agents.rbc import HourRBC
from citylearn.agents.q_learning import TabularQLearning
from citylearn.citylearn import CityLearnEnv
from citylearn.data import DataSet
from citylearn.reward_function import RewardFunction
from citylearn.wrappers import NormalizedObservationWrapper 
from citylearn.wrappers import StableBaselines3Wrapper
from citylearn.wrappers import TabularQLearningWrapper

# baseline RL algorithms
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback
#%% Preparing data 
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

from Essential_functions import load_wholedata

df_main= load_wholedata()
df_main.drop('TOU',axis=1,inplace=True)
df_main.rename(columns={'RTP': 'electricity_pricing', 'Load': 'non_shiftable_load', 'PV': 'solar_generation'},inplace=True)

#Only using a weeks data for simulation but needs to be generalized
df=df_main.copy()
df['hour']=df.index.hour
df['day_type']=df.index.dayofweek
df[['average_unmet_cooling_setpoint_difference',
    'indoor_relative_humidity','dhw_demand','Indoor_Temperature',
    'cooling_demand','heating_demand','Month','DLS','DWH_Heating',
    '1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16']]=0



#%% CSVing the data for input
building_data=df[['Month','hour','day_type','DLS','Indoor_Temperature',
                  'average_unmet_cooling_setpoint_difference',
                  'indoor_relative_humidity', 'non_shiftable_load',
                  'solar_generation','DWH_Heating',
                  'cooling_demand','heating_demand']]
pricing_data=df[['electricity_pricing','1','2','3']]
weather_data=df[['1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16']]

building_data.to_csv(r'C:\Users\Karthikeyan\Desktop\Github\Master-Thesis\Main_Folder\11_EMS_RL\Database\building_data.csv', index=False)
pricing_data.to_csv(r'C:\Users\Karthikeyan\Desktop\Github\Master-Thesis\Main_Folder\11_EMS_RL\Database\pricing_data.csv',index=False)
weather_data.to_csv(r'C:\Users\Karthikeyan\Desktop\Github\Master-Thesis\Main_Folder\11_EMS_RL\Database\weather_data.csv',index=False)
#%% Setting up schema

sim_start=100
sim_end=368


schema={
        "root_directory":r'C:\Users\Karthikeyan\Desktop\Github\Master-Thesis\Main_Folder\11_EMS_RL\Database',
        "central_agent" :True,
        "simulation_start_time_step":sim_start,
        "simulation_end_time_step":sim_end,
        "episodes":1,
        "seconds_per_time_step":3600,
        
        "observations":{
                        "hour":{
                                "active":True,
                                "shared_in_central_agent":True
                               },
            
                        "day_type":{
                                "active":False,
                                "shared_in_central_agent":True
                               },
                        
                        "electricity_pricing":{
                                "active":False,
                                "shared_in_central_agent":True
                               },
                        
                        "solar_generation":{
                                "active":False,
                                "shared_in_central_agent":True
                               },
                        
                        "non_shiftable_load":{
                                "active":False,
                                "shared_in_central_agent":True
                               }
                       },
        
        "actions":     {
                        "electrical_storage": 
                            {
                            "active": True
                            }
                       },
        
        "agent":       {
                        "type": 'citylearn.agents.sac.SAC',
                        
                        "attributes": 
                            {
                            "hidden_dimension": [256, 256],
                            "discount": 0.99,
                            "tau": 0.005,
                            "lr": 0.003,
                            "batch_size": 256,
                            "replay_buffer_capacity": 100000.0,
                            "start_training_time_step": 6000,
                            "end_exploration_time_step": 7000,
                            "deterministic_start_time_step": 26280,
                            "action_scaling_coef": 0.5,
                            "reward_scaling": 5.0,
                            'update_per_time_step': 2
                            }
                        },
            
        "reward_function": 
                        {
                            "type": "citylearn.reward_function.IndependentSACReward",
                            "attributes": None
                        },
        "buildings" :
                        {
                        "Building_1": 
                            {
                            "include": True,
                            "energy_simulation": 'building_data.csv',
                            "weather":'weather_data.csv',
                            #"carbon_intensity": None,
                            "pricing":"pricing_data.csv",
                            "inactive_observations": [],
                            "inactive_actions": [],
                            "electrical_storage":
                                {
                                "type": "citylearn.energy_model.Battery",
                                "autosize": False,
                                "attributes": 
                                        {
                                            "capacity": 6.4,
                                            "efficiency": 0.9,
                                            "capacity_loss_coefficient": 1e-05,
                                            "loss_coefficient": 0.0,
                                            "nominal_power": 5.0}
                                        },
                            "pv": 
                                {
                                "type": "citylearn.energy_model.PV",
                                "autosize": False,
                                "attributes":
                                        {
                                            "nominal_power": 4.0
                                        }
                                        
                                },
                        }
        
        }
    }


#%% Setting random seed
RANDOM_SEED = 7
print('Random seed:', RANDOM_SEED)
#%% Initializing CityLearn Environment
env = CityLearnEnv(schema)
#%% Properties and methods check of environment

print('Current time step:', env.time_step)
print('environment number of time steps:', env.time_steps)
print('environment uses central agent:', env.central_agent)
print('Common (shared) observations amogst buildings:', env.shared_observations)
print('Number of buildings:', len(env.buildings))

#%% KPIs

def get_kpis(env: CityLearnEnv) -> pd.DataFrame:
    """Returns evaluation KPIs.

    Electricity consumption, cost and carbon emissions KPIs are provided
    at the building-level and average district-level. Average daily peak,
    ramping and (1 - load factor) KPIs are provided at the district level.

    Parameters
    ----------
    env: CityLearnEnv
        CityLearn environment instance.

    Returns
    -------
    kpis: pd.DataFrame
        KPI table.
    """

    kpis = env.evaluate()

    # names of KPIs to retrieve from evaluate function
    kpi_names = [
        'electricity_consumption', 'cost', 'carbon_emissions', 
        'average_daily_peak', 'ramping', '1 - load_factor'
    ]
    kpis = kpis[
        (kpis['cost_function'].isin(kpi_names))
    ].dropna()

    # round up the values to 3 decimal places for readability
    kpis['value'] = kpis['value'].round(3)
    
    # rename the column that defines the KPIs
    kpis = kpis.rename(columns={'cost_function': 'kpi'})

    return kpis

def plot_building_kpis(envs: Mapping[str, CityLearnEnv]) -> plt.Figure:
    """Plots electricity consumption, cost and carbon emissions 
    at the building-level for different control agents in bar charts.

    Parameters
    ----------
    envs: Mapping[str, CityLearnEnv]
        Mapping of user-defined control agent names to environments 
        the agents have been used to control.

    Returns
    -------
    fig: plt.Figure
        Figure containing plotted axes.
    """

    kpis_list = []

    for k, v in envs.items():
        kpis = get_kpis(v)
        kpis = kpis[kpis['level']=='building'].copy()
        kpis['building_id'] = kpis['name'].str.split('_', expand=True)[1]
        kpis['building_id'] = kpis['building_id'].astype(int).astype(str)
        kpis['env_id'] = k
        kpis_list.append(kpis)

    kpis = pd.concat(kpis_list, ignore_index=True, sort=False)
    kpi_names= kpis['kpi'].unique()
    column_count_limit = 3
    row_count = math.ceil(len(kpi_names)/column_count_limit)
    column_count = min(column_count_limit, len(kpi_names))
    building_count = len(kpis['name'].unique())
    env_count = len(envs)
    figsize = (3.0*column_count, 0.3*env_count*building_count*row_count)
    fig, _ = plt.subplots(
        row_count, column_count, figsize=figsize, sharey=True
    )

    for i, (ax, (k, k_data)) in enumerate(zip(fig.axes, kpis.groupby('kpi'))):
        sns.barplot(x='value', y='name', data=k_data, hue='env_id', ax=ax)
        ax.axvline(1.0, color='black', linestyle='--', label='Baseline')
        ax.set_xlabel(None)
        ax.set_ylabel(None)
        ax.set_title(k)

        if i == len(kpi_names) - 1:
            ax.legend(
                loc='upper left', bbox_to_anchor=(1.3, 1.0), framealpha=0.0
            )
        else:
            ax.legend().set_visible(False)

        for s in ['right','top']:
            ax.spines[s].set_visible(False)

        for p in ax.patches:
            ax.text(
                p.get_x() + p.get_width(), 
                p.get_y() + p.get_height()/2.0, 
                p.get_width(), ha='left', va='center'
            )
    
    plt.tight_layout()
    return fig

def plot_district_kpis(envs: Mapping[str, CityLearnEnv]) -> plt.Figure:
    """Plots electricity consumption, cost, carbon emissions,
    average daily peak, ramping and (1 - load factor) at the 
    district-level for different control agents in a bar chart.

    Parameters
    ----------
    envs: Mapping[str, CityLearnEnv]
        Mapping of user-defined control agent names to environments
        the agents have been used to control.

    Returns
    -------
    fig: plt.Figure
        Figure containing plotted axes.
    """

    kpis_list = []

    for k, v in envs.items():
        kpis = get_kpis(v)
        kpis = kpis[kpis['level']=='district'].copy()
        kpis['env_id'] = k
        kpis_list.append(kpis)

    kpis = pd.concat(kpis_list, ignore_index=True, sort=False)
    row_count = 1
    column_count = 1
    env_count = len(envs)
    kpi_count = len(kpis['kpi'].unique())
    figsize = (6.0*column_count, 0.225*env_count*kpi_count*row_count)
    fig, ax = plt.subplots(row_count, column_count, figsize=figsize)
    sns.barplot(x='value', y='kpi', data=kpis, hue='env_id', ax=ax)
    ax.axvline(1.0, color='black', linestyle='--', label='Baseline')
    ax.set_xlabel(None)
    ax.set_ylabel(None)

    for s in ['right','top']:
        ax.spines[s].set_visible(False)

    for p in ax.patches:
        ax.text(
            p.get_x() + p.get_width(), 
            p.get_y() + p.get_height()/2.0, 
            p.get_width(), ha='left', va='center'
        )

    ax.legend(loc='upper left', bbox_to_anchor=(1.3, 1.0), framealpha=0.0)
    plt.tight_layout()

    return fig

def plot_building_load_profiles(envs: Mapping[str, CityLearnEnv]) -> plt.Figure:
    """Plots building-level net electricty consumption profile 
    for different control agents.

    Parameters
    ----------
    envs: Mapping[str, CityLearnEnv]
        Mapping of user-defined control agent names to environments 
        the agents have been used to control.

    Returns
    -------
    fig: plt.Figure
        Figure containing plotted axes.
    """

    building_count = len(list(envs.values())[0].buildings)
    column_count_limit = 4
    row_count = math.ceil(building_count/column_count_limit)
    column_count = min(column_count_limit, building_count)
    figsize = (4.0*column_count, 1.75*row_count)
    fig, _ = plt.subplots(row_count, column_count, figsize=figsize)

    for i, ax in enumerate(fig.axes):
        for k, v in envs.items():
            y = v.buildings[i].net_electricity_consumption
            x = range(len(y))
            ax.plot(x, y, label=k)

        y = v.buildings[i].net_electricity_consumption_without_storage
        ax.plot(x, y, label='Baseline')
        ax.set_title(v.buildings[i].name)
        ax.set_xlabel('Time step')
        ax.set_ylabel('kWh')
        ax.xaxis.set_major_locator(ticker.MultipleLocator(24))

        if i == building_count - 1:
            ax.legend(
                loc='upper left', bbox_to_anchor=(1.0, 1.0), framealpha=0.0
            )
        else:
            ax.legend().set_visible(False)
        
    
    plt.tight_layout()

    return fig

def plot_district_load_profiles(envs: Mapping[str, CityLearnEnv]) -> plt.Figure:
    """Plots district-level net electricty consumption profile 
    for different control agents.

    Parameters
    ----------
    envs: Mapping[str, CityLearnEnv]
        Mapping of user-defined control agent names to environments 
        the agents have been used to control.

    Returns
    -------
    fig: plt.Figure
        Figure containing plotted axes.
    """

    figsize = (5.0, 1.5)
    fig, ax = plt.subplots(1, 1, figsize=figsize)

    for k, v in envs.items():
        y = v.net_electricity_consumption
        x = range(len(y))
        ax.plot(x, y, label=k)
    
    y = v.net_electricity_consumption_without_storage
    ax.plot(x, y, label='Baseline')
    ax.set_xlabel('Time step')
    ax.set_ylabel('kWh')
    ax.xaxis.set_major_locator(ticker.MultipleLocator(24))
    ax.legend(loc='upper left', bbox_to_anchor=(1.0, 1.0), framealpha=0.0)
    
    plt.tight_layout()
    return fig

def plot_battery_soc_profiles(envs: Mapping[str, CityLearnEnv]) -> plt.Figure:
    """Plots building-level battery SoC profiles fro different control agents.

    Parameters
    ----------
    envs: Mapping[str, CityLearnEnv]
        Mapping of user-defined control agent names to environments 
        the agents have been used to control.

    Returns
    -------
    fig: plt.Figure
        Figure containing plotted axes.
    """

    building_count = len(list(envs.values())[0].buildings)
    column_count_limit = 4
    row_count = math.ceil(building_count/column_count_limit)
    column_count = min(column_count_limit, building_count)
    figsize = (4.0*column_count, 1.75*row_count)
    fig, _ = plt.subplots(row_count, column_count, figsize=figsize)

    for i, ax in enumerate(fig.axes):
        for k, v in envs.items():
            soc = np.array(v.buildings[i].electrical_storage.soc)
            capacity = v.buildings[i].electrical_storage.capacity_history[0]
            y = soc/capacity
            x = range(len(y))
            ax.plot(x, y, label=k)

        ax.set_title(v.buildings[i].name)
        ax.set_xlabel('Time step')
        ax.set_ylabel('SoC')
        ax.xaxis.set_major_locator(ticker.MultipleLocator(24))
        
        if i == building_count - 1:
            ax.legend(
                loc='upper left', bbox_to_anchor=(1.0, 1.0), framealpha=0.0
            )
        else:
            ax.legend().set_visible(False)
        
    
    plt.tight_layout()
    
    return fig

def plot_simulation_summary(envs: Mapping[str, CityLearnEnv]):
    """Plots KPIs, load and battery SoC profiles for different control agents.

    Parameters
    ----------
    envs: Mapping[str, CityLearnEnv]
        Mapping of user-defined control agent names to environments 
        the agents have been used to control.
    """
    
    _ = plot_building_kpis(envs)
    print('Building-level KPIs:')
    plt.show()
    _ = plot_building_load_profiles(envs)
    print('Building-level load profiles:')
    plt.show()
    _ = plot_battery_soc_profiles(envs)
    print('Battery SoC profiles:')
    plt.show()
    _ = plot_district_kpis(envs)
    print('District-level KPIs:')
    plt.show()
    print('District-level load profiles:')
    _ = plot_district_load_profiles(envs)
    plt.show()
    
#%% Setting up the environment
tql_env = CityLearnEnv(schema)

#%% define active observations and actions and their bin sizes
observation_bins = {'hour': 24}
action_bins = {'electrical_storage': 12}

# initialize list of bin sizes where each building 
# has a dictionary in the list definining its bin sizes
observation_bin_sizes = []
action_bin_sizes = []

for b in tql_env.buildings:
    # add a bin size definition for the buildings
    observation_bin_sizes.append(observation_bins)
    action_bin_sizes.append(action_bins)
    
#%%

tql_env = TabularQLearningWrapper(
    tql_env.unwrapped,
    observation_bin_sizes=observation_bin_sizes, 
    action_bin_sizes=action_bin_sizes
)

#%%

class CustomTabularQLearning(TabularQLearning):
    def __init__(
        self, env: CityLearnEnv, loader: IntProgress,
        random_seed: int = None, **kwargs
    ):
        r"""Initialize CustomRBC.

        Parameters
        ----------
        env: Mapping[str, CityLearnEnv]
            CityLearn environment instance.
        loader: IntProgress
            Progress bar.
        random_seed: int
            Random number generator reprocucibility seed for 
            eqsilon-greedy action selection.
        kwargs: dict
            Parent class hyperparameters
        """
        
        super().__init__(env=env, random_seed=random_seed, **kwargs)
        self.loader = loader
        self.reward_history = []
    
    def next_time_step(self):
        if self.env.time_step == 0:
            self.reward_history.append(0)

        else:
            self.reward_history[-1] += sum(self.env.rewards[-1])
            
        self.loader.value += 1
        super().next_time_step()
        
def get_loader(**kwargs):
    """Returns a progress bar"""
    
    kwargs = {
        'value': 0,
        'min': 0,
        'max': 10,
        'description': 'Simulating:',
        'bar_style': '',
        'style': {'bar_color': 'maroon'},
        'orientation': 'horizontal',
        **kwargs
    }
    return IntProgress(**kwargs)
#%%

# ----------------- CALCULATE NUMBER OF TRAINING EPISODES -----------------
i = 1000
m = tql_env.observation_space[0].n
n = tql_env.action_space[0].n
t = tql_env.time_steps - 1
tql_episodes = m*n*i/t
tql_episodes = int(tql_episodes)
print('Q-Table dimension:', (m, n))
print('Number of episodes to train:', tql_episodes)

# ------------------------------- SET LOADER ------------------------------
loader = get_loader(max=tql_episodes*t)
print(loader)

# ----------------------- SET MODEL HYPERPARAMETERS -----------------------
tql_kwargs = {
    'epsilon': 0.9,
    'minimum_epsilon': 0.01,
    'epsilon_decay': 0.0001,
    'learning_rate': 0.005,
    'discount_factor': 0.99,
}

# ----------------------- INITIALIZE AND TRAIN MODEL ----------------------
tql_model = CustomTabularQLearning(
    env=tql_env, 
    loader=loader, 
    random_seed=RANDOM_SEED, 
    **tql_kwargs
)
_ = tql_model.learn(episodes=tql_episodes)

#%%

observations = tql_env.reset()

while not tql_env.done:
    actions = tql_model.predict(observations, deterministic=True)
    observations, _, _, _ = tql_env.step(actions)

# plot summary and compare with other control results
plot_simulation_summary({'TQL': tql_env})

#%%

fig, ax = plt.subplots(1, 1, figsize=(4, 2))
y = np.array([max(
    tql_model.minimum_epsilon,
    tql_model.epsilon_init*np.exp(-tql_model.epsilon_decay*e)
) for e in range(100_000)])
ref_x = len(y) - len(y[y <= 0.5]) - 1
ref_y = y[ref_x]
ax.plot(y)
ax.axvline(ref_x, color='red', linestyle=':')
ax.axhline(ref_y, color='red', linestyle=':')
ax.axvline(tql_episodes, color='green', linestyle=':')
ax.set_xlabel('Episode')
text = f'{ref_x} training episodes needed to get\nat least 50%'\
    ' exploitation probability.'
ax.text(ref_x + 1000, ref_y + 0.05, text, color='red')
ax.text(
    tql_episodes + 1000, 
    ref_y - 0.1, 
    f'Current training episodes = {tql_episodes}', 
    va='bottom', color='green'
)
ax.set_ylabel(r'$\epsilon$')
plt.show()

#%% SAC environment
sac_env = CityLearnEnv(schema)
sac_env = NormalizedObservationWrapper(sac_env)
sac_env = StableBaselines3Wrapper(sac_env)
sac_model = SAC(policy='MlpPolicy', env=sac_env, seed=RANDOM_SEED)
#%%

class CustomCallback(BaseCallback):
    def __init__(self, env: CityLearnEnv, loader: IntProgress):
        r"""Initialize CustomCallback.

        Parameters
        ----------
        env: Mapping[str, CityLearnEnv]
            CityLearn environment instance.
        loader: IntProgress
            Progress bar.
        """

        super().__init__(verbose=0)
        self.loader = loader
        self.env = env
        self.reward_history = [0]

    def _on_step(self) -> bool:
        r"""Called each time the env step function is called."""

        if self.env.time_step == 0:
            self.reward_history.append(0)

        else:
            self.reward_history[-1] += sum(self.env.rewards[-1])
            
        self.loader.value += 1

        return True
    
#%%

# ----------------- CALCULATE NUMBER OF TRAINING EPISODES -----------------
fraction = 0.5
sac_episodes = int(tql_episodes*fraction)
print('Fraction of Tabular Q-Learning episodes used:', fraction)
print('Number of episodes to train:', sac_episodes)
sac_episode_timesteps = sac_env.time_steps - 1
sac_total_timesteps = sac_episodes*sac_episode_timesteps

# ------------------------------- SET LOADER ------------------------------
sac_loader = get_loader(max=sac_total_timesteps)
print(sac_loader)

# ------------------------------- TRAIN MODEL -----------------------------
sac_callback = CustomCallback(env=sac_env, loader=sac_loader)
sac_model = sac_model.learn(
    total_timesteps=sac_total_timesteps, 
    callback=sac_callback
)
#%%

observations = sac_env.reset()
sac_actions_list = []

while not sac_env.done:
    actions, _ = sac_model.predict(observations, deterministic=True)
    observations, _, _, _ = sac_env.step(actions)
    sac_actions_list.append(actions)

# plot summary and compare with other control results
plot_simulation_summary({'TQL': tql_env, 'SAC-1': sac_env})