# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import tensorflow as tf
import numpy as np
import pandas as pd
import keras
from sklearn import preprocessing
import csv
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.recurrent import LSTM



# Algorithm parameters
TIME_STEPS = 3
INPUT_SIZE = 28
BATCH_SIZE = 50
#BATCH_INDEX = 1
OUTPUT_SIZE = 10
CELL_SIZE = 50 # how many hidden units in hidden layer
LR = 0.001
Train_rate = 0.9
data_list = []

foxconndf= pd.read_csv('0050.csv', index_col=0 )
# Drop the row which contains NA value, how='any' if there's at least one blank, how='all' if all data values are blank
foxconndf.dropna(how='any',inplace=True)
foxconndf = foxconndf.drop(['Date'], axis = 1)
# Initiate
def normalize(df):
    newdf= df.copy()
    min_max_scaler = preprocessing.MinMaxScaler()
    for feature in df.columns:
        newdf[feature] = min_max_scaler.fit_transform(df[feature].values.reshape(-1,1))
    return newdf

foxconndf_norm = normalize(foxconndf)
#print(foxconndf_norm)

# Split the data into training and testing data set
def data_helper(df, time_frame):
    
    number_features = len(df.columns)
    # list the data as matrix
    datavalue = df.as_matrix()
    result = []
    
    for index in range( len(datavalue) - (time_frame + 1) ): 
        result.append(datavalue[index: index + (time_frame+1) ]) 
    
    result = np.array(result)
    # number_train = training data numbers
    number_train = round(Train_rate * result.shape[0]) 
    
    x_train = result[:int(number_train), :-1] 
    y_train = result[:int(number_train), -1][:,-1] 
    
    
    x_test = result[int(number_train):, :-1]
    y_test = result[int(number_train):, -1][:,-1]
    
    
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], number_features))
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], number_features))  
    return [x_train, y_train, x_test, y_test]

X_train, y_train, X_test, y_test = data_helper(foxconndf_norm, TIME_STEPS)


def build_model(input_length, input_dim):
    d = 0.5
    model = Sequential()
    model.add(LSTM(256, input_shape=(input_length, input_dim), return_sequences=True))
    model.add(Dropout(d))
    model.add(LSTM(256, input_shape=(input_length, input_dim), return_sequences=False))
    model.add(Dropout(d))
    model.add(Dense(16,kernel_initializer="uniform",activation='relu'))
    model.add(Dense(1,kernel_initializer="uniform",activation='linear'))
    model.compile(loss='mse',optimizer='adam', metrics=['accuracy'])
    return model

model = build_model(TIME_STEPS, 67 )
model.fit( X_train, y_train, batch_size=BATCH_SIZE, epochs=500, validation_split=0.1, verbose=1)
def denormalize(df, norm_value):
    original_value = df['ClosePrice'].values.reshape(-1,1)
    norm_value = norm_value.reshape(-1,1)
    
    min_max_scaler = preprocessing.MinMaxScaler()
    min_max_scaler.fit_transform(original_value)
    denorm_value = min_max_scaler.inverse_transform(norm_value)
    
    return denorm_value

pred = model.predict(X_test)
denorm_pred = denormalize(foxconndf, pred)
denorm_ytest = denormalize(foxconndf, y_test)

import matplotlib.pyplot as plt
#%matplotlib inline  
plt.plot(denorm_pred,color='red', label='Prediction')
plt.plot(denorm_ytest,color='blue', label='Answer')
plt.legend(loc='best')
plt.show()