#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 30 02:06:16 2019

@author: apro2929
"""
from sklearn import preprocessing
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.layers import LSTM, Dense, Activation, ThresholdedReLU, MaxPooling2D, Embedding, Dropout
from keras.optimizers import Adam, SGD, RMSprop
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping
# Multilayer Perceptron
from keras.utils import plot_model
from keras.models import Model
from keras.layers import Input
from keras.layers import Dense
from matplotlib.pyplot import hist
import time

user_data = pd.read_csv('User_data.csv')

target = user_data['Restaurant']
user_data.drop('Restaurant', axis = 1, inplace = True)

lb = preprocessing.LabelBinarizer()
lb.fit(range(max(target)+1))
target = lb.transform(target)
target = np.array(target)
target = target.astype(np.int8)
for i in range(0, target.shape[1]):
    user_data['ResID'+str(i)] = target[:,i]
features = ['day', 'Age', 'Location', 'Gender', 'Student',
                               'Employed', 'Unemployed', 'Marital_status']
user_data = user_data.groupby(features).sum().reset_index()
user_values = user_data[features].values    
user_target = user_data.drop(features,axis = 1)
columns = list(user_target)
for col in columns:
    user_target[col] = user_target[col].astype('bool')
        
columns =list(user_target)
single_outs = []
for col in columns:
    single_outs.append(np.array(user_target[col]))

visible = Input(shape=(8,))
hidden1 = Dense(32)(visible)
hidden2 = Dense(64)(hidden1)
hidden3 = Dense(128)(hidden2)
hidden4 = Dense(256)(hidden3)

outputs = []
for i in range(0, user_target.shape[1]):
    outputs.append(Dense(1, activation='sigmoid')(hidden4))
model = Model(inputs=visible, outputs=outputs)

# plot graph
#plot_model(model, to_file='multiple_outputs.png')
bin_crossentropy = ['binary_crossentropy']*user_target.shape[1]
model.compile(optimizer = 'rmsprop', loss = bin_crossentropy, metrics = ['accuracy'])
#callbacks_list=[EarlyStopping(monitor="val_loss",min_delta=.001, patience=3,mode='auto')]
model.summary()
model.fit(user_values, single_outs, epochs = 10, batch_size = 128 )

pred2 = []
for i in range(0,user_target.shape[1]):
    pred2.append(0)
ts = time.time()  
for i in range(0, 100): 
    record = (user_values[i]).reshape(1,8)
    pred = model.predict(record)
    for i in range(0,len(pred)):
        pred2[i]+=pred[i][0][0]
time.time() - ts        

hist(pred2, bins=50, density=False)
scaler = preprocessing.MinMaxScaler()
pred2 = np.log(np.array(pred2)).reshape(-1,1)        
scaler.fit(pred2)
pred3 = scaler.transform(pred2)
pred4 = []
for i in pred3:
    if i >0.5:
        pred4.append(i[0])
pred4 = (np.array(pred2)).reshape(-1,1)        
scaler = preprocessing.MinMaxScaler()
scaler.fit(pred4)
pred4 = scaler.transform(pred4)
fig, ax = plt.subplots()
N, bins, patches = hist(pred3, bins=50, density=False)
for i in range(0,15):
    patches[i].set_facecolor('#EF1E1E')
for i in range(20,30):    
    patches[i].set_facecolor('#EFEB1E')
for i in range(30,48):    
    patches[i].set_facecolor('#41EF1E')
for i in range(48,50):    
    patches[i].set_facecolor('#1EBEEF')
ax.set_title('Restaurants likelihood')
plt.show() 
#observation:
# record prediction takes 0.004 second
