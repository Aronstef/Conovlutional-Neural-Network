#!/usr/bin/env python
# coding: utf-8

# In[1]:


from tensorflow.keras.datasets.mnist import load_data
from tensorflow.keras.utils import to_categorical

import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np


# In[2]:


(x_train,y_train),(x_test,y_test)=load_data()


# In[3]:


print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)


# In[4]:


plt.imshow(x_train[0],cmap='gray')
plt.show()


# In[5]:


y_train[0]


# In[6]:


for (i) in range(5):
    plt.imshow(x_train[i],cmap='gray')
    plt.title(f"Target: {str(y_train[i])}")
    plt.show()


# # Data Processing

# In[7]:


x_train=x_train.reshape(60000,784)
x_test=x_test.reshape(10000,784)


# In[8]:


y_train_ohe = to_categorical(y_train)
y_test_ohe = to_categorical(y_test)


# In[9]:


y_train[0]


# In[10]:


y_train_ohe[0]


# In[11]:


print(x_train.shape)
print(x_test.shape)
print(y_train_ohe.shape)
print(y_test_ohe.shape)


# # Tensorflow Model

# In[12]:


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input,Dense
from tensorflow.keras.optimizers import Adam,RMSprop


# In[13]:


model = Sequential()
model.add(Input(784))
model.add(Dense(units=10,activation='softmax')) 


# In[14]:


model.compile(optimizer='rmsprop',loss='categorical_crossentropy',metrics='accuracy')


# In[15]:


#model fit
model.fit(x=x_train,y=y_train_ohe,epochs=100,batch_size=1000,validation_data=(x_train,y_train_ohe))


# In[16]:



model.history.history.keys()


# In[17]:


train_loss= model.history.history["loss"]
val_loss = model.history.history["val_loss"]


# In[18]:


train_loss= model.history.history["loss"]
val_loss = model.history.history["val_loss"]

plt.train(train_loss)
plt.plot(val_loss)
plt.xlabel("Epochs")
plt.ylabel("MSE")
plt.legend(["train","test"])
plt.grid()
plt.show()

