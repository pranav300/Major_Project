#!/usr/bin/env python
# coding: utf-8

# ## Import Functions

# In[1]:


import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import os
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout


# ## Pre-Processing

# In[2]:


dataset_train = pd.read_csv("/Users/maharshichattopadhyay/Documents/Study/Major_Project/DataSet/Final_Dataset/MCB_Train.csv")
dataset_test = pd.read_csv("/Users/maharshichattopadhyay/Documents/Study/Major_Project/DataSet/Final_Dataset/MCB_Test.csv") 


# In[3]:


dataset_train.head()


# In[4]:


#dataset_test.head()


# In[5]:


trainset = dataset_train.iloc[:,8:9].values
sc = MinMaxScaler(feature_range = (0,1))
training_scaled = sc.fit_transform(trainset)
trainset


# In[6]:


x_train = []
y_train = []


# In[7]:


for i in range(1,1476):
    x_train.append(training_scaled[i-1:i, 0])
    y_train.append(training_scaled[i,0])
x_train,y_train = np.array(x_train),np.array(y_train)


# In[8]:


#x_train.shape


# In[9]:


x_train = np.reshape(x_train, (x_train.shape[0],x_train.shape[1],1))
x_train.shape


# In[10]:


#type(training_scaled)


# ## LSTM-RNN Model

# In[11]:


model = Sequential()
model.add(LSTM(units = 25,return_sequences = True,input_shape = (x_train.shape[1],1)))
model.add(Dropout(0.1))
model.add(LSTM(units = 25,return_sequences = True))
model.add(Dropout(0.1))
model.add(LSTM(units = 25,return_sequences = True))
model.add(Dropout(0.1))
model.add(LSTM(units = 25))
model.add(Dropout(0.1))
model.add(Dense(units = 1))
model.compile(optimizer = 'adam',loss = 'mean_squared_error')#,metrics = ['accuracy'])
model.fit(x_train,y_train,epochs = 100, batch_size = 32, validation_split=0.20, verbose=1)


# ## Predictions

# In[12]:


testset = dataset_test.iloc[:,8:9].values
dataset_total = pd.concat((dataset_train['Open'],dataset_test['Open']),axis = 0)
dataset_total.head()


# In[13]:


inputs = dataset_total[len(dataset_total) - len(dataset_test)-60:].values
#inputs
inputs = inputs.reshape(-1,1)
#inputs


# In[14]:


inputs = sc.transform(inputs)
inputs.shape


# In[15]:


x_test = []
for i in range(1,367):
    x_test.append(inputs[i-1:i,0])
x_test = np.array(x_test)
#x_test.shape
x_test = np.reshape(x_test, (x_test.shape[0],x_test.shape[1],1))
x_test.shape


# In[16]:


predicted_price = model.predict(x_test)
predicted_price = sc.inverse_transform(predicted_price)


# ## Plot

# In[17]:


plt.plot(testset,color = 'red', label = 'Real Price')
plt.plot(predicted_price, color = 'blue', label = 'Predicted Price')
plt.legend()
plt.show()


# In[ ]:




