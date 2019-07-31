#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 30 17:15:20 2019

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
from xgboost import XGBRegressor, XGBClassifier
from xgboost import plot_importance
from imblearn.over_sampling import SMOTE

def plot_features(booster, figsize):    
    fig, ax = plt.subplots(1,1,figsize=figsize)
    return plot_importance(booster=booster, ax=ax)


user_data = pd.read_csv('User_data.csv')

target = user_data['Restaurant']
profile = user_data[user_data.Location == 0][user_data.Age == 24][user_data.Gender == 0][user_data.Student == 0][user_data.Marital_status == 0].reset_index(drop = True)
profile = (profile.drop([0,1,2,3,4],axis = 0)).reset_index(drop = True) 
user_data.drop('Restaurant', axis = 1, inplace = True)
personal_restaurants = np.array(profile['Restaurant'])
len(np.unique(personal_restaurants))
hist(personal_restaurants, bins=len(np.unique(personal_restaurants)), density=False)
profile = profile.drop(['Age','Location','Gender','Student','Unemployed','Marital_status', 'Employed'], axis = 1)
list(set([1,2,3,4,5,6,7]) - set(profile['day'].values))
profile2 = profile.copy()
profile2['day'] -=1
profile3= profile.copy()
profile3['day'] +=1 

def lag_feature(df, lags):
    tmp = df[['day','Restaurant','day']]
    for i in lags:
        shifted = tmp.copy()
        shifted.columns = ['day','Restaurant', 'day'+'_lag_'+str(i)]
        shifted['day'+'_lag_'+str(i)] = ((shifted['day']+i)%7)
        for j in range(0,len(shifted['day'+'_lag_'+str(i)])):
            if shifted['day'+'_lag_'+str(i)][j] == 0:
                shifted['day'+'_lag_'+str(i)][j] = 1
        df = pd.merge(df, shifted, on=['day','Restaurant'], how='left')
    return df

lags = [4,5]
p2_lag = lag_feature(profile2,lags)
p3_lag = lag_feature(profile3,lags)
profile = lag_feature(profile,lags)
profile = profile.append(p2_lag, ignore_index = True)
profile = profile.append(p3_lag, ignore_index = True)

target_restaurants = profile['Restaurant']
profile = profile.drop('Restaurant', axis = 1)
ye = profile.values
X_resampled, y_resampled = SMOTE(k_neighbors=2).fit_resample(profile.values, target_restaurants.values)
model = XGBClassifier(   
    seed=42)

ts = time.time()
model.fit(
    X_resampled, 
    y_resampled, 
    eval_metric="logloss",  
    verbose=True)
time.time() - ts

preds = []
for i in range(0,48):
    test = np.array(X_resampled[i,:]).reshape(1,X_resampled.shape[1])
    Y_pred = model.predict(test)
    preds.append(Y_pred)

plot_features(model, (10,14))