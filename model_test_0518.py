# -*- coding: utf-8 -*-
"""
Created on Tue May 15 11:36:54 2018

@author: Ken
"""

#import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn import preprocessing
from keras.models import Sequential, load_model
from keras.layers.core import Dense, Dropout
from keras.layers.recurrent import LSTM

# Load files
foxconndf= pd.read_csv('0051.csv', index_col=0 )
foxconndf.dropna(how='any',inplace=True)

foxconndf = foxconndf.drop(['Date'], axis=1)

TIME_STEPS = 5
TRAIN_RATE = 0.8
CELL_SIZE = 50
BATCH_SIZE = None
EPOCHS_NUM = 100
LAYERS_NUM = 2
DROPOUT_RATE = 0.3
FEATURE_NUM = len(foxconndf.columns) # Feature num

def normalize(df):
    newdf= df.copy()
    min_max_scaler = preprocessing.MinMaxScaler()
    for feature in df.columns:
        newdf[feature] = min_max_scaler.fit_transform(df[feature].values.reshape(-1,1))
    return newdf
foxconndf_norm= normalize(foxconndf)

def data_helper(df, time_frame):
    # Feature number
    number_features = len(df.columns)
    datavalue = df.as_matrix()
    x_result = []    
    y_result = []
    for index in range(len(datavalue)-TIME_STEPS):
        x_result.append(datavalue[index: index + time_frame])
        y_result.append(datavalue[index + time_frame][4])  # test for closeprice
    x_result = np.array(x_result)
    y_result = np.array(y_result)

    number_train = round(TRAIN_RATE * x_result.shape[0])
    x_train, x_test = x_result[:int(number_train)], x_result[int(number_train) :]
    y_train, y_test = y_result[:int(number_train)], y_result[int(number_train) :]
    
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], number_features))
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], number_features))

    return [x_train, y_train, x_test, y_test]

X_train, y_train, X_test, y_test = data_helper(foxconndf_norm, TIME_STEPS)

data_helper(foxconndf_norm, TIME_STEPS)


def build_model(input_length, input_dim):
    model = Sequential()
    for _ in range(LAYERS_NUM):
        model.add(LSTM(CELL_SIZE, input_shape=(input_length, input_dim), return_sequences=True))
        model.add(Dropout(DROPOUT_RATE))
    model.add(LSTM(CELL_SIZE, input_shape=(input_length, input_dim), return_sequences=False))
    model.add(Dropout(DROPOUT_RATE))
    model.add(Dense(16,kernel_initializer="uniform",activation='relu'))
    model.add(Dense(1,kernel_initializer="uniform",activation='linear')) # CELL_SIZE 5, output dim
    model.compile(loss='mse',optimizer='adam', metrics=['accuracy'])
    return model

model = build_model( TIME_STEPS, FEATURE_NUM )
model.fit( X_train, y_train, batch_size=BATCH_SIZE, epochs=EPOCHS_NUM, validation_split=0.1, verbose=1)


pred = model.predict(X_test) # model.predict(data set)

def denormalize(df, norm_value):
    original_value = df['ClosePrice'].values.reshape(-1,1)
    norm_value = norm_value.reshape(-1,1)

    min_max_scaler = preprocessing.MinMaxScaler()
    min_max_scaler.fit_transform(original_value)
    denorm_value = min_max_scaler.inverse_transform(norm_value)

    return denorm_value

denorm_pred = denormalize(foxconndf, pred)
denorm_ytest = denormalize(foxconndf, y_test)
print(model.summary()) # model summary
print(model.get_config()) # model configuration
model.save('model_test_closeonly.h1')
my_model = load_model('model_test_closeonly.h1')
score = model.evaluate(X_train, y_train, batch_size=BATCH_SIZE)

import matplotlib.pyplot as plt
#%matplotlib inline  
plt.plot(denorm_pred,color='red', label='Prediction')
plt.plot(denorm_ytest,color='blue', label='Answer')
plt.legend(loc='best')
plt.show()

#denorm_pred = denormalize(foxconndf, pred)
#denorm_ytest = denormalize(foxconndf, y_test)
