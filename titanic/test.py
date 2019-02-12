# -*- coding: utf-8 -*-
"""
Created on Mon Jan  7 11:40:43 2019

@author: Nikhil
"""
import pandas as pd
import numpy as np

train_dataset = pd.read_csv('train.csv')
test_dataset = pd.read_csv('test.csv')

print("---- Train Dataset .head()----")
print(train_dataset.head())

print('Train dataset column types info')

dtype_info = train_dataset.dtypes.reset_index()
dtype_info.columns = ['Column','Column Type']
print(dtype_info.groupby("Column Type").aggregate('count').reset_index())
print(dtype_info)

#Checking for missing valeues

datahasnan = False
if train_dataset.count().min() == train_dataset.shape[0] and test_dataset.count().min() == test_dataset.shape[0]:
    datahasnan = False
else:
    datahasnan = True
    
if datahasnan == True:
    nas = pd.concat([train_dataset.isnull().sum(), test_dataset.isnull().sum()],axis = 1,keys=['Train Dataset','Test Dataset'])
    
print(nas)
print(nas[nas.sum(axis =1 )>0])

#Pclass vs survived
print(train_dataset[['Pclass','Survived']].groupby(['Pclass'],as_index = False).mean().sort_values(by='Survived',ascending=False))

# sex vs Survived
print(train_dataset[["Sex", "Survived"]].groupby(['Sex'], as_index=False).mean().sort_values(by='Survived', ascending=False))

# SibSp vs Survived
#Sibling = brother, sister, stepbrother, stepsister
#Spouse = husband, wife (mistresses and fiancés were ignored)
print(train_dataset[["SibSp", "Survived"]].groupby(['SibSp'], as_index=False).mean().sort_values(by='Survived', ascending=False))


# Parch vs Survived
#Parent = mother, father
#Child = daughter, son, stepdaughter, stepson
#Some children travelled only with a nanny, therefore parch=0 for them.
print(train_dataset[["Parch", "Survived"]].groupby(['Parch'], as_index=False).mean().sort_values(by='Survived', ascending=False))


#Cleaning data and adding missing values

train_random_ages = np.random.randint(train_dataset['Age'].mean()-train_dataset['Age'].std(),
                                      train_dataset['Age'].mean()+train_dataset['Age'].std(),
                                      size = train_dataset['Age'].isnull().sum())

test_random_ages = np.random.randint(test_dataset['Age'].mean()-test_dataset['Age'].std(),
                                      test_dataset['Age'].mean()+test_dataset['Age'].std(),
                                      size = test_dataset['Age'].isnull().sum())

train_dataset['Age'][np.isnan(train_dataset['Age'])] = train_random_ages
test_dataset['Age'][np.isnan(test_dataset['Age'])] = test_random_ages

train_dataset['Age'] = train_dataset['Age'].astype(int)

#For embarked

print(train_dataset['Embarked'].mode())

train_dataset['Embarked'].fillna('S',inplace = True)
test_dataset['Embarked'].fillna('S',inplace = True)

#Creating a new column by mapping embarked to integers
'''
from sklearn.preprocessing import LabelEncoder
label_encoder_1 = LabelEncoder()
train_dataset[:,-1] = label_encoder_1.fit_transform(train_dataset[:,-1])
X = train_dataset.values
X[:,-1] = label_encoder_1.fit_transform(X[:,-1])
'''
train_dataset['Port'] = train_dataset['Embarked'].map({'S':0,'C':1,'Q':2}).astype(int)
test_dataset['Port'] = test_dataset['Embarked'].map({'S':0,'C':1,'Q':2}).astype(int)

#For fare
train_dataset['Fare'].fillna(train_dataset['Fare'].median,inplace = True)
