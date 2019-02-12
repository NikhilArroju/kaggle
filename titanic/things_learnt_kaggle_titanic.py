
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 10 11:50:34 2019

@author: Nikhil
"""
str_variable = "This is string"
str_variable2 = "this is no 2"
print(str_variable)
print("The string is %s"%str_variable)
print("The strings are %s and %s"%(str_variable,str_variable2))

import pandas as pd
import numpy as np

train_dataset = pd.read_csv('train.csv')
test_dataset = pd.read_csv('test.csv')
print(train_dataset.PassengerId)
print(train_dataset['PassengerId'])
#above two give the same result which incledes rows with index and at the 
#end name,length and dtpye

print(train_dataset.shape) #gives (891,12)
print(train_dataset.shape[0]) #gives no of rows ie 891
print(train_dataset.count()) #gives each column name and no of non empty rows in each
cnt = train_dataset.count() #gives a series and .min() gives out a integer

train_dataset.isnull().sum()

nas = pd.concat([train_dataset.isnull().sum(),test_dataset.isnull().sum()],axis =1,keys = ['column_name','count'])
train_dataset.Age.mean()

#.astype(int) used to conver all values to int obviously
print(train_dataset['Embarked'])
print(train_dataset.loc[:,'Survived':'Embarked']) #loc is useful like this and also by boolean operations
print(train_dataset['Embarked'][61])
print(train_dataset['Embarked'].fillna('S',inplace = False)[61])
#inplace =True modifies the dataframe itself

x = np.zeros((3,3))

full_dataset = [train_dataset,test_dataset]
