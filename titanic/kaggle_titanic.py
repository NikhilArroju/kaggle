# -*- coding: utf-8 -*-
"""
Created on Fri Jan 11 18:50:26 2019
@author: Nikhil
"""

import numpy as np
import pandas as pd

train_dataset = pd.read_csv('train.csv')
test_dataset = pd.read_csv('test.csv')

#Looking for nan values
hasnan = False
if train_dataset.isnull().sum().max() != 0 and test_dataset.isnull().sum().max() != 0:
    hasnan = True

if hasnan:
    nan_details = pd.concat([train_dataset.isnull().sum(),test_dataset.isnull().sum()],
                     axis =1,
                     keys = ['train_dataset','test_dataset'],sort = True)
    print(nan_details[nan_details.sum(axis=1)>0])

#dealing with embarked column
embarked_mode = train_dataset.Embarked.mode()
print(embarked_mode)
train_dataset.Embarked.fillna(embarked_mode[0],inplace = True)

#dealing with ages
train_age_mean = train_dataset.Age.mean()
train_age_std = train_dataset.Age.std()
print(train_age_mean,train_age_std)
train_ages = np.random.randint(train_age_mean - train_age_std,
                                 train_age_mean + train_age_std,
                                 size = train_dataset.Age.isnull().sum())

test_age_mean = test_dataset.Age.mean()
test_age_std = test_dataset.Age.std()
print(test_age_mean,test_age_std)
test_ages = np.random.randint(test_age_mean - test_age_std,
                                 test_age_mean + test_age_std,
                                 size = test_dataset.Age.isnull().sum())

train_dataset.Age[np.isnan(train_dataset.Age)] = train_ages
test_dataset.Age[np.isnan(test_dataset.Age)] = test_ages

#dealing with the fare in test data
test_dataset.Fare.fillna(test_dataset.Age.median(),inplace = True)

#feature engineering
'''
Have to decide which features to combine for the prediction
available are Pclass,Name,Sex,Age,SibSip,Parch,Ticket,Fare,Cabin,Embarked

pclass and fare are correlated so no use using both 
similarly higher fares have a cabin so it is kind of skewed

sibsip and parch can be used to determine family size\
'''







