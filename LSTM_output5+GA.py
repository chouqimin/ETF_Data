# -*- coding: utf-8 -*-
"""
Created on Fri May 25 11:35:40 2018

@author: Fatty
"""
#Fitness Funcition
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing
from keras.models import Sequential, load_model
from keras.layers.core import Dense, Dropout
from keras.layers.recurrent import LSTM

chromosom_value=[[10,20,30],[0.1,0.2,0.3],[16,32,64,128,256],[1,2,3],[100,200,300],[100,200,300],[0.1,0.2,0.3,0.4,0.5],["sigmoid","relu","linear"],["sigmoid","relu","linear"],[0.1,0.2,0.3]]

def normalize(df):
    newdf= df.copy()
    min_max_scaler = preprocessing.MinMaxScaler()
    for feature in df.columns:
        newdf[feature] = min_max_scaler.fit_transform(df[feature].values.reshape(-1,1))
    return newdf
    
def data_helper(df, time_frame, train_rate):
    # Feature number
    number_features = len(df.columns)
    datavalue = df.as_matrix()
    x_result = []    
    y_result = []
    for index in range(len(datavalue)-time_frame):
        x_result.append(datavalue[index: index + time_frame])
        y_result.append(datavalue[index + time_frame][:5])  # test for all value
    x_result = np.array(x_result)
    y_result = np.array(y_result)

    number_train = round((1 - train_rate) * x_result.shape[0])
    x_train, x_test = x_result[:int(number_train)], x_result[int(number_train) :]
    y_train, y_test = y_result[:int(number_train)], y_result[int(number_train) :]
    
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], number_features))
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], number_features))

    return [x_train, y_train, x_test, y_test]

def build_model(input_length, df, cell, layers, dropout):
    print(len(df.columns))
    number_features = len(df.columns)
    model = Sequential()
    for _ in range(layers):
        model.add(LSTM(cell, input_shape=(input_length, number_features), return_sequences=True))
        model.add(Dropout(dropout))
    model.add(LSTM(cell, input_shape=(input_length, number_features), return_sequences=False))
    model.add(Dropout(dropout))
    model.add(Dense(16,kernel_initializer="uniform",activation='relu'))
    model.add(Dense(5,kernel_initializer="uniform",activation='linear')) # CELL_SIZE 5, output dim
    model.compile(loss='mse',optimizer='adam', metrics=['accuracy']) # mae
    return model

def denormalize(df, norm_value):
    denorm_value = []
    pre_df = pd.DataFrame(norm_value, columns = df.columns[:5])
    for feature in df.columns[:5]:
        
        original_value = df[feature].values.reshape(-1,1)
        denorm_column = pre_df[feature].values.reshape(-1,1)
    
        min_max_scaler = preprocessing.MinMaxScaler()
        min_max_scaler.fit_transform(original_value)
        denorm_value.append(min_max_scaler.inverse_transform(denorm_column))
    return denorm_value

def generate_model(parameters, df, csvfile):
    time_step, train_rate, cell_size, layers_num, batchsize, epochs_num, dropout_rate, activation_1, activation_2, validation_rate = parameters
    df_norm= normalize(df)
    X_train, y_train, X_test, y_test = data_helper(df_norm, time_step, train_rate)
    model = build_model(time_step, df, cell_size, layers_num, dropout_rate)
    model.fit( X_train, y_train, batch_size=batchsize, epochs=epochs_num, validation_split=validation_rate, verbose=1)
    
    pred = model.predict(X_test) # model.predict(data set)

    denorm_pred = denormalize(df, pred)
    denorm_ytest = denormalize(df, y_test)
    
    model_path = csvfile[:-4] + '_model.md' 
    model.save(model_path)
    
    #%matplotlib inline  
    for i in range(5):
        plt.plot(denorm_pred[i],color='red', label='Prediction')
        plt.plot(denorm_ytest[i],color='blue', label='Answer')
        plt.legend(loc='best')
        plt.show()
    score = model.evaluate(X_test, y_test, batch_size=batchsize)
    print(denorm_pred)
    return score[0]

#GA Function
import random
def createrandomList(create_number,category):
    list_random = [ random.randint(0,category) for i in range(create_number)]
    return(list_random)

#init create population chromosome(100)--function input：filename & population, output：population_list 
def create_population(population):
    population_list=[]
    for popu in range(population):
        chromosome_create=[]
        for index in chromosom_value:
            chromosome_create.append(index[createrandomList(1,len(index)-1)[0]]) #從chromosom_value隨機產生一個值塞到
        population_list.append(chromosome_create)
    return population_list

def cross(cross_rate,chromosome_1,chromosome_2):
    random_rate=random.random()
    if(random_rate>cross_rate):
        point_1=random.randint(0,len(chromosome_1)-1)
        point_2=random.randint(0,len(chromosome_1)-1)

        while(point_1>=point_2):
            point_1=random.randint(0,len(chromosome_1)-1)
            point_2=random.randint(0,len(chromosome_1)-1)

        #print("cross_point1: "+str(point_1))
        #print("cross_point2: "+str(point_2))
        tmp=0
        for index in range(point_1,point_2+1):
            tmp=chromosome_1[index]
            chromosome_1[index]=chromosome_2[index]
            chromosome_2[index]=tmp
    return chromosome_1

def mutation(mutation_rate,chromosome):
    random_rate=random.random()
    if(random_rate>mutation_rate):
        #print("mutation_start")
        point_1=random.randint(0,len(chromosome)-1)#產生二個隨機point1 & point2 為了到時候mutation
        point_2=random.randint(0,len(chromosome)-1)
        while(point_1==point_2):#如果point1 point2 一樣要重新產生
            point_1=random.randint(0,len(chromosome)-1)
            point_2=random.randint(0,len(chromosome)-1)
        #print("mu_point1= "+str(point_1))
        #print("mu_point2= "+str(point_2))

        tmp=chromosome[point_1]#改變染色體中第point1的值，如果產生的新值與舊的一樣要再重新產生
        #chromosome[point_1]值等於chromosom_value裡面隨機產生的一個值createrandomList(1,len(chromosom_value[point_1])-1)[0]
        chromosome[point_1]=chromosom_value[point_1][createrandomList(1,len(chromosom_value[point_1])-1)[0]]#
        while(tmp==chromosome[point_1]):
            chromosome[point_1]=chromosom_value[point_1][createrandomList(1,len(chromosom_value[point_1])-1)[0]]

        tmp_2=chromosome[point_2]#改變染色體中第point2的值，如果產生的新值與舊的一樣要再重新產生
        chromosome[point_2]=chromosom_value[point_2][createrandomList(1,len(chromosom_value[point_2])-1)[0]]
        while(tmp_2==chromosome[point_2]):
            chromosome[point_2]=chromosom_value[point_2][createrandomList(1,len(chromosom_value[point_2])-1)[0]]
    return chromosome

def calculateY_rank(rank_number,population_list):
    result_all_x_fitness=[]#記錄所有population與產生的y
    best_result=[]#記錄最好的population&y
    
    foxconndf= pd.read_csv('0051.csv', index_col=0 )
    foxconndf.dropna(how='any',inplace=True) # remove the rows with blank
    foxconndf = foxconndf.drop(['Date'], axis=1)
    
    for index in population_list:
        fitness_value=generate_model(index, foxconndf, '0051.csv')#calculate fintness_value
        result_all_x_fitness.append([fitness_value,index])
    
    result_rank=[]#record before rank"s x & fitness_value
    for index in range(0,rank_number):
        result_rank.append(sorted(result_all_x_fitness,reverse=True)[index])#sort:big->small
    best_result=result_rank[0]
    
    x_final=[]#記錄前十五名的染色體
    for index in result_rank:
        x_final.append(index[1])
    return [x_final,best_result]

#產生新的X(上一代最好的前15個交配+突變產生新的15個+隨機產生85個新的x)--function
def create_new_x(rank_number,population,cross_rate,mutation_rate,x_final):
    new_x=[]
    #先把前15名的染色體做交配&突變->產生新15個染色體
    for index in range(0,rank_number):
        other_ch=createrandomList(1,rank_number-1)[0]
        while(index==other_ch):
            other_ch=createrandomList(1,rank_number-1)[0] #變矩陣要減1 if index= other_ch -> create other_ch again
        cross_result=cross(cross_rate,x_final[index],x_final[other_ch])
        new_x.append(mutation(mutation_rate,cross_result))

    #隨機產生第16~100的染色體
    for index in range(0,population-rank_number):
        new_x.append(create_population(1)[0])
    return new_x

#GA + LSTM ->main
import datetime
#init parameter
iteration=2
iteration_now=0
population=2
rank_number=1

cross_rate=0.1
mutation_rate=0.1

y_target=0#fitnessvalue的目標，到達就停止迭代->望小
y_best=100000#目前為止(所有迭代中)最好的fitnessvalue
y_now=100000#當前此迭代中最好的fitnessvalue
x_best=[]

population_list=create_population(population)#產生初始母體染色體

Totalstarttime=datetime.datetime.now()
print("Start GA: "+str(Totalstarttime))
while(not(iteration_now>=iteration or y_best<y_target)):
    iteration_now=iteration_now+1
    print("\niteration_now= "+str(iteration_now))
    Y=calculateY_rank(rank_number,population_list) #return X_final(前15名的染色體) & best_result(此代最好的fitness跟染色體)
    y_now=Y[1][0]
    print("y_now= "+str(y_now))
    if y_now<y_best:
        y_best=y_now
        print("y_best= "+str(y_best))
        x_best=Y[1][1]
        #print("x_best= "+str(x_best))
    population_list=create_new_x(rank_number,population,cross_rate,mutation_rate,Y[0])

print("iteration_now= "+str(iteration_now))
print("y_best= "+str(y_best))
print("x_best= "+str(x_best))

print("GA's end time is "+str(datetime.datetime.now()))
print("All time spends:  "+str(datetime.datetime.now()-Totalstarttime))