# America-Energy-Usage-Stochastic-ARIMA-
Stochastic Processing for America Energy Usage and its prediction
by
Bayu Satria Persada              (1706985924)
Anandwi Ghuran Muhajallin Arreto (1706985911)
Fadhilah Rheza P                 (1706042863) 
Yudhistira Indyka                (1706042895)

OVERVIEW

* [Description](#Description)
* [Process](#Predicting_Process)

## Description

ARIMA, short for ‘Auto Regressive Integrated Moving Average’ is actually a class of models that ‘explains’ a given time series based on its own past values, that is, its own lags and the lagged forecast errors, so that equation can be used to forecast future values. Any ‘non-seasonal’ time series that exhibits patterns and is not a random white noise can be modeled with ARIMA models.

An ARIMA model is characterized by 3 terms: p, d, q 
where,
p is the order of the AR term
q is the order of the MA term
d is the number of differencing required to make the time series stationary

To make the data stationary first we need to differenciate it, that is the I (integrate) part of ARIMA, so if your data is already Stationary then its only ARMA

## Predicting Process

First we check the data by its stationary or not, by checking with adfuller test, if the value is above 0,05 then it is not stationary
we include this code


```
import numpy as np, pandas as pd
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima_model import ARIMA
from numpy import log
from statsmodels.tsa.stattools import acf
df2 = pd.read_csv('https://raw.githubusercontent.com/Bayusatriapersada/America-Energy-Usage-Stochastic-ARIMA-/master/Energy%20Usage%20Data.csv', names=['AEP'],header=0)
result = adfuller(df2.AEP.dropna())
print('ADF Statistic: %f' % result[0])
print('p-value: %f' % result[1])
```


