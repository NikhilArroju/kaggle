# -*- coding: utf-8 -*-
"""
Created on Tue Apr  2 17:39:01 2019

@author: Nik
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

train_dataset = pd.read_csv('train.csv')
test_dataset = pd.read_csv('test.csv')
sample_submission = pd.read_csv('sample_submission.csv')

X_train_df = train_dataset.drop(columns = ['label'])
y_train_df = train_dataset.label

X_test_df = test_dataset

X_train = np.array(X_train_df,dtype = 'float32')
y_train = np.array(y_train_df,dtype = 'float32')

X_test = np.array(X_test_df,dtype = 'float32')

X_train = X_train/255
y_train = y_train/255

X_test = X_test/255


#plt.imshow(X_train[np.random.randint(X_train_df.shape[0]),:].reshape(28,28)
#          ,cmap='gray')

from sklearn.model_selection import train_test_split
X_train,X_validate,y_train,y_validate = train_test_split(X_train,y_train,
                                                         test_size = 0.2)

X_train = X_train.reshape(X_train.shape[0],28,28,1)
X_validate = X_validate.reshape(X_validate.shape[0],28,28,1)
X_test = X_test.reshape(X_test.shape[0],28,28,1)

import keras
from keras.models import Sequential
from keras.layers import Conv2D,MaxPooling2D,Dropout,Flatten,Dense
from keras.optimizers import Adam,RMSprop
flag = 2
if flag == 1:
  model = Sequential()
  model.add(Conv2D(32,(4,4),input_shape = (28,28,1),activation = 'relu'))
  model.add(MaxPooling2D(pool_size = (2,2)))
  model.add(Dropout(rate = 0.3))
  
  model.add(Conv2D(32,(3,3),activation = 'relu'))
  model.add(MaxPooling2D(pool_size = (2,2)))
  model.add(Dropout(rate = 0.3))

  model.add(Conv2D(32,(3,3),activation = 'relu'))
  model.add(MaxPooling2D(pool_size = (2,2)))
  model.add(Dropout(rate = 0.3))
  
  model.add(Flatten())

  model.add(Dense(units = 512,activation='relu'))
  
  model.add(Dense(units = 256,activation = 'relu'))
  
  model.add(Dense(units = 10,activation = 'softmax'))
  
  model.compile(loss = 'sparse_categorical_crossentropy',
              optimizer = Adam(lr = 0.01),
              metrics = ['accuracy'])
else:
  
  model = Sequential()
  model.add(Conv2D(32,(5,5),input_shape = (28,28,1),activation = 'relu'))
  model.add(Conv2D(32,(5,5),input_shape = (28,28,1),activation = 'relu'))
  model.add(MaxPooling2D(pool_size = (2,2)))
  model.add(Dropout(rate = 0.5))
  
  model.add(Conv2D(64,(3,3),input_shape = (28,28,1),activation = 'relu'))
  model.add(Conv2D(64,(3,3),input_shape = (28,28,1),activation = 'relu'))
  model.add(MaxPooling2D(pool_size = (2,2)))
  model.add(Dropout(rate = 0.5))
  
  model.add(Flatten())
  
  model.add(Dense(units = 8192,activation = 'relu'))
  model.add(Dropout(rate = 0.5))
  
  model.add(Dense(units = 2048,activation = 'relu'))
  model.add(Dropout(rate = 0.5))
  
  model.add(Dense(units = 10,activation = 'softmax'))
  
  model.compile(optimizer=RMSprop(lr=0.0001,rho=0.9,epsilon=1e-08,decay=0.00001),
                loss="categorical_crossentropy",metrics=["accuracy"])

#Image data generator
from keras.preprocessing.image import ImageDataGenerator
data_generator = ImageDataGenerator(rotation_range = 10,width_shift_range=0.1 ,
                                    height_shift_range = 0.1 ,
                                    zoom_range = 0.1,)

data_generator.fit(X_train)

train_generator  = data_generator.flow(X_train,y_train,batch_size = 32)
validation_generator = data_generator.flow(X_validate,y_validate,batch_size = 32)

history = model.fit_generator(data_generator.flow(X_train,y_train,batch_size = 32),
                               epochs = 100,verbose = 1,
                               validation_data = (X_validate,y_validate),
                               steps_per_epoch=X_train.shape[0]//32)

y_pred = model.predict(X_test)

predicted_df = pd.DataFrame({'ImageId':X_test,
                             'Lable':y_pred})
predicted_df.to_csv(columns = ['ImageId','Label'])


