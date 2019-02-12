# -*- coding: utf-8 -*-
"""
Created on Sun Jan 20 14:46:41 2019

@author: Nikhil
"""

#from numpy cheatsheet
import numpy as np

a = np.array([1,2,3])

b = np.array([[1.5,2,3], [4,5,6]], dtype = float)

c = np.array([[(1.5,2,3), (4,5,6)], [(3,2,1), (4,5,6)]],dtype = float)

np.zeros((3,4))

d = np.arange(10,25,5)

np.linspace(0,2,9)
np.corrcoef(a)

b.reshape(2,-1)

x = np.array([1, 2, 3])
y=x
z = np.copy(x)

x[0]=10
w = np.copy(x).deepcopy(x)

x[1] = 20
