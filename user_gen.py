#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 29 20:26:17 2019

@author: apro2929
"""

import numpy as np
from matplotlib.pyplot import hist
import random as rdm
from sklearn import preprocessing
import matplotlib.pyplot as plt
import pandas as pd
    
values = hist((np.random.rayleigh(16, 100000)+15), bins=10000, density=True)
Rayleigh_x = np.array(values[1]).astype(np.int8)
Rayleigh_y = np.array(values[0]).astype(np.int8)
for i in range(0,len(Rayleigh_x)):
    if Rayleigh_x[i] > 60:
        Rayleigh_x[i] -=15
rdm.seed(42)
Gender = []
for i in range(10000):
	Gender.append(rdm.randint(0,1))
    
Gender = np.array(Gender)
Gender = Gender.astype(np.int8)


rdm.seed(43)
Employment = []
for i in range(10000):
    if Rayleigh_x[i] < 22:
        if rdm.randint(0,100)>95:
            Employment.append(1)
        else:
            Employment.append(0)
    else:
        Employment.append(rdm.randint(1,2))
    
lb = preprocessing.LabelBinarizer()
lb.fit(range(max(Employment)+1))
Employment = lb.transform(Employment)
Employment = np.array(Employment)
Employment = Employment.astype(np.int8)


Marital_status = []
for i in range(0,10000):
    if Gender[i] : ## true is female
        if rdm.randint(1,10)>4:
            Marital_status.append(1)
        else:
            Marital_status.append(0)
    else:
        if rdm.randint(1,10)>6:
            Marital_status.append(1)
        else:
            Marital_status.append(0)
Marital_status = np.array(Marital_status).astype(np.int8)        

mu, sigma = 0, 0.1 # mean and standard deviation
s = np.random.normal(mu, sigma, 100000)     
count, bins, ignored = plt.hist(s, 10000, normed=True)
normal_y = 1/(sigma * np.sqrt(2 * np.pi)) * np.exp( - (bins - mu)**2 / (2 * sigma**2)) 
plt.plot(bins, normal_y,linewidth=2, color='r')
plt.show()   
normal_y = normal_y*100
normal_y = np.array(normal_y).astype(np.int16)    

mu, sigma = 0.3, 0.5 # mean and standard deviation
s = np.random.normal(mu, sigma, 100000)     
count, bins, ignored = plt.hist(s, 10000, normed=True)
normal_y_target = 1/(sigma * np.sqrt(2 * np.pi)) * np.exp( - (bins - mu)**2 / (2 * sigma**2)) 
plt.plot(bins, normal_y_target,linewidth=2, color='r')
plt.show()
normal_y_target = np.array(np.abs(20*np.log(normal_y_target))).astype(np.int16)
day = []
for i in range(0,10000):
    day.append(rdm.randint(1,7))
data = {'day':day, 'Age':Rayleigh_x[:10000], 'Location':normal_y[:10000], 'Gender':Gender, 'Student':Employment[:,1],
        'Employed':Employment[:,2], 'Unemployed':Employment[:,0],
        'Marital_status':Marital_status, 'Restaurant':normal_y_target[:10000]}
user_df = pd.DataFrame(data = data)
user_df.to_csv('User_data.csv', index=False)

