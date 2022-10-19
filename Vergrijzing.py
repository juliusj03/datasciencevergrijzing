#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn import linear_model
import matplotlib.pyplot as plt

import seaborn as sns
import scipy.stats as sps
vergrijzing = pd.read_csv('vergrijzing.csv',sep=',', index_col=0, usecols=[0, 5,8])
display(vergrijzing)
deuli=[]
irlli=[]
nedli=[]
deuto=[]
deu50=[]
irlto=[]
irl50=[]
nedto=[]
ned50=[]
nedper=[]
deuper=[]
irlper=[]
p=0
agel= vergrijzing['Age']
valuel = vergrijzing['Value']
print(agel)
for x in vergrijzing.index:
    
    
    if x == 'DEU':
        if agel[p] == 'Total':
            deuto.append(valuel[p])
        else:
            deu50.append(valuel[p])
        
    if x == 'IRL':
        if agel[p] == 'Total':
            irlto.append(valuel[p])
        else:
            irl50.append(valuel[p])
    if x == 'NLD':
        if agel[p] == 'Total':
            nedto.append(valuel[p])
        else:
            ned50.append(valuel[p])
    p+=1
dat=[x for x in range(1972,2021)]
for x in range(len(ned50)):
    nedper.append((ned50[x]/nedto[x]*100))
    irlper.append((irl50[x]/irlto[x]*100))
    deuper.append((deu50[x]/deuto[x]*100))


# In[6]:


import pandas_ta

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.linear_model import LinearRegression
import tensorflow as tf
from tensorflow.keras.layers import Dense, LSTM
from tensorflow.keras.models import Sequential
from sklearn.preprocessing import MinMaxScaler


def vergrijzing(x):
    df = pd.DataFrame(x, index =dat, columns =['percen'])
    y = df['percen'].fillna(method='ffill').values.reshape(- 1, 1)

# scale the data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler = scaler.fit(y)
    y = scaler.transform(y)

# generate the training sequences
    n_forecast = 1
    n_lookback = 45

    X = []
    Y = []

    for i in range(n_lookback, len(y) - n_forecast + 1):
        X.append(y[i - n_lookback: i])
        Y.append(y[i: i + n_forecast])

    X = np.array(X)
    Y = np.array(Y)

# train the model
    tf.random.set_seed(0)

    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(X.shape[1], X.shape[2])))
    model.add(LSTM(50, return_sequences=False))
    model.add(Dense(25))
    model.add(Dense(1))

    model.compile(loss='mse', optimizer='adam')
    model.fit(X, Y, epochs=100, batch_size=128, validation_split=0.2, verbose=0)

# generate the multi-step forecasts
    n_future = 40
    y_future = []

    x_pred = X[-1:, :, :]  # last observed input sequence
    y_pred = Y[-1]         # last observed target value

    for i in range(n_future):

    # feed the last forecast back to the model as an input
        x_pred = np.append(x_pred[:, 1:, :], y_pred.reshape(1, 1, 1), axis=1)

    # generate the next forecast
        y_pred = model.predict(x_pred)

    # save the forecast
        y_future.append(y_pred.flatten()[0])

# transform the forecasts back to the original scale
    y_future = np.array(y_future).reshape(-1, 1)
    y_future = scaler.inverse_transform(y_future)

# organize the results in a data frame
    df_past = df[['percen']].reset_index()

    df_past.rename(columns={'index': 'Date'}, inplace=True)



    df_past['Forecast'] = np.nan
    pat =df_past['Date'].iloc[-1]
    dats=[]
    df_future = pd.DataFrame(columns=['Date', 'percen', 'Forecast'])
    for x in range(n_future):
        dats.append(pat+x+1)
    df_future['Date'] = dats
    df_future['Forecast'] = y_future.flatten()
    df_future['percen'] = np.nan

    resultsnl = df_past.append(df_future).set_index('Date')
    
    resultsnl['total'] = resultsnl['percen'].combine_first(resultsnl['Forecast'])
    return resultsnl['total']
    

# plot the results


dat1=[x for x in resultsnl.index]

# plot the results
plt.xlabel('jaren')

plt.ylabel('percentage 50+ jaar (%)')
plt.title('vergrijzing van verschillende eu landen ')
plt.scatter(dat, nedper, label = "ned", color='b')
plt.scatter(dat, deuper, label = "deu", color='r')
plt.scatter(dat, irlper, label = "irl", color='g')
plt.plot(dat1, vergrijzing(nedper),label = "ned voor", color='b')
plt.plot(dat1, vergrijzing(deuper),label = "deu voor", color='r')
plt.plot(dat1, vergrijzing(irlper),label = "irl voor", color='g')
plt.legend()
plt.show()


# In[ ]:




