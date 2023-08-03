# -*- coding: utf-8 -*-
"""
Created on Fri May 19 09:34:19 2023

@author: Karthikeyan
"""

#%% Import packages
import numpy as np
import os 
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.layers import Dense

from keras.optimizers import Adam
#%% Buffer class

class ReplayBuffer:
    
    # Since continuous action spaces, number of actions is a misnomer, its the components
    def __init__(self,max_size,input_shape,n_actions):
        self.mem_size=max_size
        self.mem_cntre=0
        self.state_memory=np.zeros((self.mem_size,*input_shape))
        self.new_state_memory=np.zeros((self.mem_size,*input_shape))
        self.action_memory= np.zeros((self.mem_size,n_actions))
        self.reward_memory= np.zeros(self.mem_size)
        self.terminal_memory=np.zeros(self.mem_size,dtype=np.bool)
        
    def store_transition(self,state,action,reward,new_state,done):
        index= self.mem_cntre % self.mem_size
        
        self.state_memory[index]=state
        self.new_state_memory[index]=new_state
        action=action.flatten()
        self.action_memory[index]=action
        self.reward_memory[index]=reward
        self.terminal_memory[index]=done
        
        self.mem_cntre+=1
     
    #Not useful to learn from zeros in the whole replay buffer - check with deque memory as well
    
    #Retriving values in batches of 34
    def sample_buffer(self,batch_size):
        
        max_mem= min(self.mem_cntre,self.mem_size)
        
        #replace avoids sampling the same values
        batch= np.random.choice(max_mem,batch_size,replace=False)
        
        states=self.state_memory[batch]
        states_=self.new_state_memory[batch]
        actions=self.action_memory[batch]
        rewards=self.reward_memory[batch]
        dones=self.terminal_memory[batch]

        return states,actions,rewards,states_,dones


#%% Critic Network
class CriticNetwork(keras.Model):
    def __init__(self,fc1_dims=64,fc2_dims=124,fc4_dims=32,name='critic',chkpt_dir='tmp/ddpg'):
        super(CriticNetwork,self).__init__()
        
        self.fc1_dims=fc1_dims
        self.fc2_dims=fc2_dims
        self.fc4_dims=fc4_dims
        
        
        self.model_name=name
        self.checkpoint_dir=chkpt_dir
        self.checkpoint_file=os.path.join(self.checkpoint_dir,self.model_name+'_ddpg.h5')
        
        
        self.fc1=Dense(self.fc1_dims,activation='relu')
        self.fc2=Dense(self.fc2_dims,activation='relu')
        self.fc4=Dense(self.fc4_dims,activation='relu')
        self.q=Dense(1, activation='linear')
    
    def call(self,state,action):
        
        #concatanating state and action
        action_value=self.fc1(tf.concat([state,action],axis=1))
        action_value=self.fc2(action_value)
        action_value=self.fc4(action_value)
        
        q=self.q(action_value)
        
        return q
        
#%% Actor Network

class ActorNetwork(keras.Model):
    def __init__(self,fc1_dims=64,fc2_dims=124,fc4_dims=32,n_actions=6,name='actor',chkpt_dir='tmp/ddpg'):
        super(ActorNetwork,self).__init__()
        self.fc1_dims=fc1_dims
        self.fc2_dims=fc2_dims
        self.fc4_dims=fc4_dims
        self.n_actions=n_actions
        
        self.model_name=name
        self.checkpoint_dir=chkpt_dir
        self.checkpoint_file=os.path.join(self.checkpoint_dir,self.model_name+'_ddpg.h5')
        
        self.fc1=Dense(self.fc1_dims,activation='relu')
        self.fc2=Dense(self.fc2_dims,activation='relu')
        self.fc4=Dense(self.fc4_dims,activation='relu')
        self.mu=Dense(self.n_actions,activation='linear')
        
    
    def call(self,state):
        prob=self.fc1=(state)
        prob=self.fc2(prob)
        prob=self.fc4(prob)
        mu=self.mu(prob)
        
        return mu
#%% Agent class

class Agent:
    def __init__(self,input_dims,alpha=0.00005,beta=0.00001,env=None,gamma=0.99,
                 n_actions=2,max_size=30000,tau=0.05,batch_size=32,no_of_episodes=5000):
    
        self.gamma=gamma
        self.tau=tau
        self.memory=ReplayBuffer(max_size, input_dims, n_actions)
        self.batch_size=batch_size
        self.n_actions=n_actions
        self.max_action=env.action_space.high[0]
        self.min_action=env.action_space.low[0]
        self.no_of_episodes=no_of_episodes
        
        #Instantiating the netoworks
        self.actor=ActorNetwork(n_actions=n_actions,name='actor')
        self.critic=CriticNetwork(name='critic')
        self.target_actor=ActorNetwork(n_actions=n_actions,name='target_actor')
        self.target_critic=CriticNetwork(name='target_critic')

        #Compiling the networks
        #Even target networks needs to be compiled 
        self.actor.compile(optimizer=Adam(lr=alpha))
        self.critic.compile(optimizer=Adam(lr=beta))
        self.target_actor.compile(optimizer=Adam(lr=alpha))
        self.target_critic.compile(optimizer=Adam(lr=beta))

        self.update_network_parameters(tau)
        
        
        #Loss list
        self.critic_loss_list=[]
        self.actor_loss_list=[]
        self.step_counter=0
        self.x =[]
        self.gaussian_noise=[]
        
        self.a1_noise=self.Gaussian_Noise()
        self.a2_noise=self.Gaussian_Noise()
        
    def update_network_parameters(self,tau=None):
        #Needs to be a hard copy for the first instance
        #Not sure whats happening
        
        if tau is None:
            tau=self.tau
            
        weights=[]
        targets=self.target_actor.weights
        for i, weight in enumerate(self.actor.weights):
            weights.append(weight*tau+targets[i]*(1-tau))
        self.target_actor.set_weights(weights)
        
        weights=[]
        targets=self.target_critic.weights
        for i, weight in enumerate(self.critic.weights):
            weights.append(weight*tau + targets[i]*(1-tau))
        self.target_critic.set_weights(weights)
        
    def remember(self,state,action,reward,new_state,done):
        self.memory.store_transition(state,action,reward,new_state,done)
    
    def save_models(self):
        print('....saving models......')
        self.actor.save_weights(self.actor.checkpoint_file)
        self.critic.save_weights(self.critic.checkpoint_file)
        self.target_actor.save_weights(self.target_actor.checkpoint_file)
        self.target_critic.save_weights(self.target_critic.checkpoint_file)
        
    def load_models(self):
        print('...loading models......')
        self.actor.load_weights(self.actor.checkpoint_file)
        self.critic.load_weights(self.critic.checkpoint_file)
        self.target_actor.load_weights(self.target_actor.checkpoint_file)
        self.target_critic.load_weights(self.target_critic.checkpoint_file)
        
        
    def OU_Noise(self):
        # Parameters for OU process
        theta = 0.5  # Mean reversion parameter
        mu = 0.0  # Mean of the noise
        sigma_initial = 0.7  # Initial standard deviation of the noise
        sigma_final = 0.05  # Final standard deviation of the noise
        dt = 1  # Time step
        T = (self.no_of_episodes+1)*24 # Total time

        # Generate OU noise with reducing sigma
        t = np.arange(0, T, dt)
        n = len(t)
        self.x = np.zeros(n)
        self.x[0] = np.random.normal(mu, sigma_initial)
        for i in range(1, n):
            sigma = sigma_initial - (sigma_initial - sigma_final) * (i / n)  # Linearly reduce sigma
            dx = theta * (mu - self.x[i-1]) * dt + sigma * np.sqrt(dt) * np.random.normal()
            self.x[i] = self.x[i-1] + dx
        
        return self.x

    def Gaussian_Noise(self):
        # Parameters for Gaussian noise
        mu = 0.0  # Mean of the noise
        sigma_initial = 0.7  # Initial standard deviation of the noise
        sigma_final = 0.01  # Final standard deviation of the noise
        dt = 1  # Time step
        T = (self.no_of_episodes+1)*24  # Total time

        # Generate Gaussian noise with reducing sigma
        t = np.arange(0, T, dt)
        n = len(t)
        self.gaussian_noise = np.zeros(n)
        for i in range(n):
            sigma = sigma_initial - (sigma_initial - sigma_final) * (i / n)  # Linearly reduce sigma
            self.gaussian_noise[i] = np.random.normal(mu, sigma)
        
        return self.gaussian_noise
        
    def choose_action(self,observation,evaluate=False):
        observation = np.asarray(observation, dtype=np.float32)
        state=tf.convert_to_tensor([observation],dtype=tf.float32)
        actions=self.target_actor(state)
        
        #adding noise for exploration and exploitation
        if not evaluate:
            
            noise_tens=[self.a1_noise[self.step_counter]]
    
            actions = actions + noise_tens
            self.step_counter+=1
        
        #might go over the limits so clip it
        #actions=tf.clip_by_value(actions,self.min_action,self.max_action)
        
        #to return a scaler value
        return actions
        
    def learn(self):
        if self.memory.mem_cntre<self.batch_size:
            return
        state,action,reward,new_state,done= self.memory.sample_buffer(self.batch_size)
        
        states=tf.convert_to_tensor(state,dtype=tf.float32)
        states_=tf.convert_to_tensor(new_state,dtype=tf.float32)
        actions=tf.convert_to_tensor(action,dtype=tf.float32)
        rewards=tf.convert_to_tensor(reward,dtype=tf.float32)
        
        #updating critic network
        with tf.GradientTape() as tape:
            target_actions=self.target_actor(states_)
            critic_value_=tf.squeeze(self.target_critic(states_,target_actions),1)
            critic_value=tf.squeeze(self.critic(states,actions),1)
            target=reward + self.gamma*critic_value_*(1-done)
            critic_loss=keras.losses.MSE(target,critic_value)
            self.critic_loss_list.append(critic_loss)
            
        critic_network_gradient=tape.gradient(critic_loss,self.critic.trainable_variables)
        self.critic.optimizer.apply_gradients(zip(critic_network_gradient,self.critic.trainable_variables))
        
        #updating actor network
        with tf.GradientTape() as tape:
            new_policy_actions=self.actor(states)
            actor_loss=-self.critic(states,new_policy_actions)
            actor_loss=tf.math.reduce_mean(actor_loss)
            self.actor_loss_list.append(actor_loss)
        
        actor_network_gradient=tape.gradient(actor_loss,self.actor.trainable_variables)
        self.actor.optimizer.apply_gradients(zip(actor_network_gradient,self.actor.trainable_variables))
        
        self.update_network_parameters()
#%% Main training loop
from Env_custom_DDPG_1 import EMSenv

if __name__ == '__main__':
    env=EMSenv()
    agent=Agent(input_dims=env.observation_space.shape,env=env,n_actions=env.action_space.shape[0])
    n_games=5000
    
    score_history=[]
    best_score=-1000
    load_checkpoint=False
    
    
    for i in range(n_games):
        observation=env.reset()
        done=False
        score=0
        
        
        #Excess
        if env.current_obs[2]>=env.current_obs[0]:
            excess=env.current_obs[2]-env.current_obs[0]
            if env.current_obs[13]>=excess:
                action=agent.choose_action(observation,evaluate)
                action=np.array(action)
                action=np.reshape(action,(2,1))
                action[0]=0
                action[1]=action[1]+excess
                observation_,reward,done,info=env.step(action[0],action[1])
                agent.remember(observation,action,reward,observation_,done)
                score+=reward
                
            else:
                
        
        
        #Deficit
        else
            deficit=env.current_obs[0]-env.current_obs[2]
    
    
        score+=reward
    
    
    
    
    
    
    
    if load_checkpoint:
        n_steps=0
        while n_steps<=agent.batch_size:
            observation=env.reset()
            action=env.action_space.sample()
            observation_,reward,done,info=env.step(action)
            agent.remember(observation,action,reward,observation_,done)
            n_steps+=1
        
        agent.learn()
        agent.load_models()
        evaluate=True
        
    else:
        evaluate=False
    
    for i in range(n_games):
        observation=env.reset()
        done=False
        score=0
        while not done:
            action=agent.choose_action(observation,evaluate)
            action=np.array(action)
            action=np.reshape(action,(2,1))
            observation_,reward,done,info=env.step(action[0],action[1])
            agent.remember(observation,action,reward,observation_,done)
            score+=reward
            
            
            if not load_checkpoint:
                agent.learn()
            observation=observation_
        score_history.append(score)
        avg_score=np.mean(score_history[-100:],dtype=np.float64)
        
        if avg_score > best_score:
            best_score=avg_score
            if not load_checkpoint:
                agent.save_models()
                print('episode',i,'score %.1f' % score,'avg score %.1f' % avg_score)
        
            plt.plot(score_history)
            score+=reward
            agent.remember(observation,action,reward,observation_,done)
            if not load_checkpoint:
                agent.learn()
            observation=observation_
        score_history.append(score)
        avg_score=np.mean(score_history[-100:],dtype=np.float64)
plt.plot(score_history)

#%% Model evaluation
plt.plot(agent.actor_loss_list)
plt.show()
plt.plot(agent.critic_loss_list)
plt.show()
#%% Testing loop
test_episode=7
a1=[]
a2=[]
a3=[]
a4=[]
a5=[]
a6=[]
batt_soc=[]


load=[]
pv=[]

#Initial Batttery capacity
temp_soc=0.3

for i in range(test_episode):
        current_state=env.reset()
        current_state[12]=temp_soc
        done=False
        hour=0
        while not done:
            action=agent.choose_action(current_state, evaluate=True)
            action=np.array(action)
            action=np.reshape(action,(6,1))
            new_state,reward,done,_=env.step(action[0],action[1],action[2],action[3],action[4],action[5])
            current_state=new_state
            a1.append(action[0])
            a2.append(action[1])
            a3.append(action[2])
            a4.append(action[3])
            a5.append(action[4])
            a6.append(action[5])
            batt_soc.append(current_state[12])
            load.append(current_state[0])
            pv.append(current_state[2])
            temp_cap=current_state[12]
            hour+=1
        
        #passing on the battery capacity to the next day    
        temp_bat_cap=temp_cap
        hour=0
"""

- How to choose output layer activation function?
- lr of critic should be slightly higher than actor network

"""
#%% Checking Load,Solar,Soc Limits
import pandas as pd
df_results = pd.DataFrame({'Load':load, 'PV': pv,'a1':a1,'a2':a2,'a3':a3,'a4':a4,'a5':a5,'a6':a6})
df_results['s_tot']=df_results['a1']+df_results['a2']+df_results['a4']
df_results['l_tot']=df_results['a1']+df_results['a2']+df_results['a3']
df_results['cum_bat']=df_results['a4']+df_results['a2']+df_results['a6']
df_results['load_diff']=df_results['Load']-df_results['l_tot']
df_results['s_diff']=df_results['PV']-df_results['s_tot']


fig,ax=plt.subplots()
ax.plot(df_results['Load'])
#ax.plot(df_results['l_tot'])
ax.plot(df_results['load_diff'])
plt.show()

fig,bx=plt.subplots()
bx.plot(df_results['PV'])
#bx.plot(df_results['s_tot'])
bx.plot(df_results['s_diff'])
plt.show()

fig,cx=plt.subplots()
cx.plot(df_results['cum_bat'])
plt.show()
#%%
import os
import sys
import path
import pandas as pd

current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

from Essential_functions import real_load

test_start=31
test_end=31+test_episode-1

df_final=real_load()
df_final=df_final[df_final.index.year==2019]
start_idx=df_final[df_final.index.dayofyear==test_start].index.min()
end_idx=df_final[df_final.index.dayofyear==test_end].index.max()

df_final=df_final[start_idx:end_idx]
df_final['Bat_actions']=all_actions
df_final['Bat_capacity']=batt_cap
df_final['Grid_ex']=df_final['Load']-df_final['PV']+df_final['Bat_actions']
df_final['Arbitrage']=df_final['Grid_ex']*df_final['RTP']

fig,ax=plt.subplots()
ax.plot(df_final['Bat_actions'])
ax2=ax.twinx()
ax2.plot(df_final['RTP'],color='g')
plt.show()

#%% Multiple axis plot
import seaborn as sns
sns.set_style("white")
fig,host=plt.subplots(figsize=(17,9),layout='constrained')

ax2 = host.twinx()
ax3 = host.twinx()
ax4 = host.twinx()

host.set_xlabel("Time (hours)")
host.set_ylabel("Battery Actions")
ax2.set_ylabel("RTP")
ax3.set_ylabel("Load")
ax4.set_ylabel("PV")

p1 = host.plot(df_final['Bat_actions'],    color='g', label="Battery Actions")
p2 = ax2.plot(df_final['RTP'],    color='r', label="RTP")
p3 = ax3.plot(df_final['Load'], color='b', label="Load")
p4 = ax4.plot(df_final['PV'], color='y', label="PV")


host.legend(handles=p1+p2+p3+p4, loc='best')
ax3.spines['right'].set_position(('outward', 60))

plt.grid(visible=False)
plt.show()

print(df_final['Arbitrage'].sum())
print(df_final['Grid_ex'].sum())
#%%

# df_final['Grid_ex'].sum()
# Out[381]: <tf.Tensor: shape=(1,), dtype=float32, numpy=array([15.690247], dtype=float32)>

# df_final['Arbitrage'].sum()
# Out[382]: <tf.Tensor: shape=(1,), dtype=float32, numpy=array([10.344871], dtype=float32)>



# df_final['Grid_ex'].sum()
# Out[418]: <tf.Tensor: shape=(1,), dtype=float32, numpy=array([9.414416], dtype=float32)>

# df_final['Arbitrage'].sum()
# Out[419]: <tf.Tensor: shape=(1,), dtype=float32, numpy=array([5.6768975], dtype=float32)>