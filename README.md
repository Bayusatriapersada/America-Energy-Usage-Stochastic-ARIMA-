# America-Energy-Usage-Stochastic-ARIMA-
Stochastic Processing for America Energy Usage and its prediction
by
Bayu Satria Persada              (1706985924)
Anandwi Ghuran Muhajallin Arreto (1706985911)
Fadhilah Rheza P                 (1706042863) 
Yudhistira Indyka                (1706042895)

OVERVIEW

* [Description](#Description)
* [Process](#Predicting Process)

## Description

ARIMA, short for ‘Auto Regressive Integrated Moving Average’ is actually a class of models that ‘explains’ a given time series based on its own past values, that is, its own lags and the lagged forecast errors, so that equation can be used to forecast future values. Any ‘non-seasonal’ time series that exhibits patterns and is not a random white noise can be modeled with ARIMA models.

An ARIMA model is characterized by 3 terms: p, d, q 
where,
p is the order of the AR term
q is the order of the MA term
d is the number of differencing required to make the time series stationary

To make the data stationary first we need to differenciate it, that is the I (integrate) part of ARIMA, so if your data is already Stationary then its only ARMA

## Predictiing Process

First we check the data by
