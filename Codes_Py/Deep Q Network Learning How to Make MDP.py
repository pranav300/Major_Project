#!/usr/bin/env python
# coding: utf-8

# # Cartpole DQN

# #### Import Dependencies

# In[1]:


import random
import gym
import numpy as np
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
import os


# #### Set Parameters

# In[2]:


env = gym.make("CartPole-v0")
state_size =  env.observation_space.shape[0]
action_size = env.action_space.n
batch_size=32    #For Gradient Descent 
n_episodes = 1001 #Maximum No Of Games we want the User To Play, We are going to make the machine remember random detaails from a few episodes
output_dir = "/Users/maharshichattopadhyay/Documents/Study/Major_Project/Cartpole/" 


# #### Define Agent 

# In[3]:


class DQNAgent:
    
    def __init__(self,state_size,action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000) #Bi-ended List, Used to remove old elements when new information comes after list is full
        self.gamma = 0.95 #Discount Factor
        self.epsilon = 1.0 #Exploration Rate of Agent (Exploration vs Exploitation)
        self.epsilon_decay = 0.995 
        self.epsilon_min = 0.01
        self.learning_rate = 0.001
        self.model = self._build_model()
        
    def _build_model(self):
        model = Sequential()
        model.add(Dense(24, input_dim = self.state_size, activation='relu'))
        model.add(Dense(24, activation = 'relu'))
        model.add(Dense(self.action_size,activation = 'linear'))
        model.compile(loss='mse', optimizer = Adam(lr = self.learning_rate))
        return model
    
    def remember(self,state,action,reward,next_state,done):#Done parameter let's us know if the episode has ended or not
        self.memory.append((state,action,reward,next_state,done))
        
    def act(self,state):
        if np.random.rand() <= self.epsilon: 
            return random.randrange(self.action_size)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])
    
    def replay(self, batch_size):        
        minibatch = random.sample(self.memory,batch_size)
        for state,action,reward,next_state,done in minibatch:
            if done:
                target = reward
            if not done:
                target = (reward + self.gamma * np.amax(self.model.predict(next_state)[0])) #target is Q-Value Function
            target_f = self.model.predict(state) 
            target_f[0][action] = target #map target from current state to future state
            self.model.fit(state,target_f,epochs=1,verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
    def load(self,name):
        self.model.load_weights(name)
       
    def save(self,name):
        self.model.save_weights(name)


# In[4]:


agent = DQNAgent(state_size,action_size)


# #### Interact With The Environment

# In[ ]:


done = False
for e in range(n_episodes):
    state = env.reset()
    state = np.reshape(state, [1, state_size])
    for time in range(5000):
        env.render()
        action = agent.act(state)
        next_state, reward, done, _ = env.step(action)
        reward = reward if not done else -10
        next_state = np.reshape(next_state, [1, state_size])
        agent.remember(state,action,reward,next_state,done)
        state=next_state
        if done:
            print("Episode: {}/{},score: {}, e: {:.2}".format(e,n_episodes,time,agent.epsilon))
            break
        if len(agent.memory) > batch_size:
            agent.replay(batch_size)
        if e%50 == 0:
            agent.save(output_dir+"weights_" + '{:04d}'.format(e)+".hdf5")


# In[ ]:




