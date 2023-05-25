# -*- coding: utf-8 -*-
"""
Created on Mon May  1 19:45:04 2023

@author: Karthikeyan
"""

#%% Loading packages
from ray import tune
import numpy as np
import pandas as pd
from Env_custom import EMSenv
#%% 
tune.run("PPO",
         config={"env":EMSenv,
                 "evaluation_interval":10,
                 "evaluation_num_episodes":100
                 },
    
        #checkpoint_freq=1000,
        )
#%%
from ray.rllib.agents.dqn import DQNTrainer, DEFAULT_CONFIG

class DQNTrainable(tune.Trainable):
    def _setup(self, config):
        self.env = EMSenv
        self.agent = DQNTrainer(config, self.env.observation_space, self.env.action_space)

    def _train(self):
        result = self.agent.train()
        return {"episode_reward_mean": result["episode_reward_mean"]}

    def _save(self, checkpoint_dir):
        return self.agent.save(checkpoint_dir)

    def _restore(self, checkpoint_path):
        self.agent.restore(checkpoint_path)
#%%
if __name__ == "__main__":
    config = DEFAULT_CONFIG.copy()
    config.update({
        # update any desired hyperparameters here
    })

    analysis = tune.run(
        DQNTrainable,
        config=config,
        stop={"episode_reward_mean": 200},
        checkpoint_at_end=True,
        #local_dir="~/ray_results"
    )