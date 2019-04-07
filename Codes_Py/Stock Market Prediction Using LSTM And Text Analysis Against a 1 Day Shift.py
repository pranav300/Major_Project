#!/usr/bin/env python
# coding: utf-8

# ## Import Functions

# In[1]:


import numpy as np
import pandas as pd
from sklearn import preprocessing

from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.layers.recurrent import LSTM
from keras.layers import  Dropout
from keras.models import model_from_json
from keras.models import load_model
from keras import regularizers

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
fmt = '$%.0f'
tick = mtick.FormatStrFormatter(fmt)

get_ipython().run_line_magic('matplotlib', 'inline')


# ## Plotting The Close Prices 

# In[4]:


data_csv = pd.read_csv("/Users/maharshichattopadhyay/Documents/Study/Major_Project/DataSet/Final_Dataset/Final_Data_MCB.csv")
data_csv[['Close']].plot()
plt.show()
plt.clf()


# ## Calculating the length of data to use

# In[5]:


percentage_of_data = 1.0
data_to_use = int(percentage_of_data*(len(data_csv)-1))

# 80% of data will be of training
train_end = int(data_to_use*0.8)

total_data = len(data_csv)
print("total_data:", total_data)


# ## Re-Arranging the data

# In[12]:


start = total_data - data_to_use

# Currently doing prediction only for 1 step ahead
steps_to_predict = 1

#close, compund, neg, neu, pos, open, high, low, volume
# Order -> 8,1,2,3,4,5,6,7,9
close = data_csv.iloc[start:total_data,8] #close
compound = data_csv.iloc[start:total_data,1] #compund
neg = data_csv.iloc[start:total_data,2] #neg
neu = data_csv.iloc[start:total_data,3] #neu
pos = data_csv.iloc[start:total_data,4] #pos
open = data_csv.iloc[start:total_data,5] #open
high = data_csv.iloc[start:total_data,6] #high
low = data_csv.iloc[start:total_data,7] #low
volume = data_csv.iloc[start:total_data,9] #volume


# In[13]:


#shift next day close and next day compund
shifted_close = close.shift(-1) #shifted close
shifted_compound = compound.shift(-1) #shifted compund

#taking only: close, next_close, compund, next_compund, volume, open, high, low
data = pd.concat([close, shifted_close, compound, shifted_compound, volume, open, high, low], axis=1)
data.columns = ['close', 'shifted_close', 'compound', 'shifted_compound','volume', 'open', 'high', 'low']

data = data.dropna()
     
print(data[:10])


# ## Separating data into x and y

# In[16]:


#Approach: Training the machine using compund, close price, and next_compund to predict next_close price.
y = data['shifted_close'] #next_close
# close, compund, next_compund, volume, open, high, low   
cols = ['close', 'compound', 'shifted_compound','volume', 'open', 'high', 'low']
x = data[cols]


# ## Preprocessing

# In[17]:


scaler_x = preprocessing.MinMaxScaler (feature_range=(-1, 1))
x = np.array(x).reshape((len(x) ,len(cols)))
x = scaler_x.fit_transform(x)

scaler_y = preprocessing.MinMaxScaler (feature_range=(-1, 1))
y = np.array (y).reshape ((len( y), 1))
y = scaler_y.fit_transform (y)


# ## Data-Splitting

# In[23]:


X_train = x[0 : train_end,]
X_test = x[train_end+1 : len(x),]    
y_train = y[0 : train_end] 
y_test = y[train_end+1 : len(y)]  

X_train = X_train.reshape (X_train. shape + (1,)) 
X_test = X_test.reshape(X_test.shape + (1,))
print(X_train.shape)
print(X_test.shape)


# ## LSTM-RNN Model

# In[25]:


batch_size = 32
nb_epoch = 100
neurons = 25
dropout = 0.1
seed = 2016
np.random.seed(seed)
model = Sequential ()
model.add(LSTM(neurons, return_sequences=True, activation='tanh', inner_activation='hard_sigmoid', input_shape=(len(cols), 1)))
model.add(Dropout(dropout))
model.add(LSTM(neurons, return_sequences=True,  activation='tanh'))
model.add(Dropout(dropout))
model.add(LSTM(neurons, activation='tanh'))
model.add(Dropout(dropout))
model.add(Dense(activity_regularizer=regularizers.l1(0.00001), output_dim=1, activation='linear'))
model.add(Activation('tanh'))
model.compile(loss='mean_squared_error' , optimizer='RMSprop')
model.fit(X_train, y_train, batch_size=batch_size, nb_epoch=nb_epoch, verbose=1, validation_split=0.2)


# ## Predictions

# In[26]:


pred = model.predict(X_test) 
pred = scaler_y.inverse_transform(np.array(pred).reshape((len(pred), 1)))
prediction_data = pred[-1]   
X_test = scaler_x.inverse_transform(np.array(X_test).reshape((len(X_test), len(cols))))


# ## Plot

# In[27]:


plt.plot(pred, label="predictions")
y_test = scaler_y.inverse_transform(np.array(y_test).reshape((len( y_test), 1)))
plt.plot([row[0] for row in y_test], label="actual")
plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), fancybox=True, shadow=True, ncol=2)
ax = plt.axes()
ax.yaxis.set_major_formatter(tick)
plt.show()
plt.clf()


# In[ ]:




