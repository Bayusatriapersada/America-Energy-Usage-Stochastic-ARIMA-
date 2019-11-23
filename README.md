# America-Energy-Usage-Stochastic-ARIMA-
Stochastic Processing for America Energy Usage and its prediction
by
* Bayu Satria Persada              (1706985924)
* Anandwi Ghuran Muhajallin Arreto (1706985911)
* Fadhilah Rheza P                 (1706042863) 
* Yudhistira Indyka                (1706042895)

OVERVIEW

* [Description](#Description)
* [Process](#Predicting_Process)
* [Code](https://github.com/Bayusatriapersada/America-Energy-Usage-Stochastic-ARIMA-/blob/master/ARIMA.ipynb)

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
The result of our test is
![Adfuller](https://github.com/Bayusatriapersada/America-Energy-Usage-Stochastic-ARIMA-/blob/master/Image/Adfuller.png)
From the data above it can be seen as the P value is lower than 0,05 it means that the daa is alread stationary, by its MEAN or by its STD Deviation.

Next, we try to find the P,D,Q Value for Arima
P can be find by using PACF that is Partial autocorrelation 

Partial Autocorrelation can be imagined as the correlation between the series and its lag, after excluding the contributions from the intermediate lags. So, PACF sort of conveys the pure correlation between a lag and the series. That way, you will know if that lag is needed in the AR term or not.

Include this Code
```
plt.rcParams.update({'figure.figsize':(12,6), 'figure.dpi':120})
fig, axes = plt.subplots(1, 2, sharex=True)
axes[0].plot(df2.AEP); axes[0].set_title('Original Series')
plot_acf(df2.AEP, ax=axes[1])
plt.show()
```

Our result on finding
![PACF](https://github.com/Bayusatriapersada/America-Energy-Usage-Stochastic-ARIMA-/blob/master/Image/PACF%20Autocorellating.png)

Our result is not so great because the autocorrelating graph is too close, it cant be read , and we cant find the P value

next, we find the D Value, D value is differencing, but we dont need that because our data is already Stationary so its D = 0;

After that we need to find the Q value by using ACF, Just like how we looked at the PACF plot for the number of AR terms, you can look at the ACF plot for the number of MA terms. An MA term is technically, the error of the lagged forecast.

The ACF tells how many MA terms are required to remove any autocorrelation in the stationarized series.

Include this code
```
fig, axes = plt.subplots(1, 2, sharex=True)
axes[0].plot(df2.AEP); axes[0].set_title('Original Series')
axes[1].set(ylim=(0,5))
plot_pacf(df2.AEP.dropna(), ax=axes[1])

plt.show()
```

our data ACF

![ACF](https://github.com/Bayusatriapersada/America-Energy-Usage-Stochastic-ARIMA-/blob/master/Image/ACF%20Autocorellating.png)

its as the same as PACF, it cant be read so, for testing purposes we gonna try all the p and q, and we find the best BIC, AIC and that is using P = 3, and Q = 3; on Auto Arima too its using p = 3 and q = 3, Auto Arima gonna be explained later

so we made the model

include this code
```
model = ARIMA(df2.AEP, order=(3,0,3))
model_fit = model.fit(disp=0)
print(model_fit.summary())
```

and we find the table for our model summary

![model](https://github.com/Bayusatriapersada/America-Energy-Usage-Stochastic-ARIMA-/blob/master/Image/Model%20Make.png)

thats the summary of our model, for simplification we did not put the test model make so it would be less complicated to see

next, we gonna train the data

using
```
train = df2.AEP[:90000]
test = df2.AEP[90000:]
```
we train the data for exactyly 90000 data, and we gonna test 90000 and the rest of the data to see if our prediction is good or not

the include this code
```
fitted = model.fit(disp=-1)  

# Forecast
fc, se, conf = fitted.forecast(121273-90000, alpha=0.05)  # 95% conf

# Make as pandas series
fc_series = pd.Series(fc, index=test.index)
lower_series = pd.Series(conf[:, 0], index=test.index)
upper_series = pd.Series(conf[:, 1], index=test.index)

# Plot
plt.figure(figsize=(12,5), dpi=100)
plt.plot(train, label='training')
plt.plot(test, label='actual')
plt.plot(fc_series, label='forecast')
plt.fill_between(lower_series.index, lower_series, upper_series, 
                 color='k', alpha=.15)
plt.title('Forecast vs Actuals')
plt.legend(loc='upper left', fontsize=8)
plt.show()
```

then we gonna predict

![predict](https://github.com/Bayusatriapersada/America-Energy-Usage-Stochastic-ARIMA-/blob/master/Image/Prediction.png)

then we try to predict the next data (untrained data) about 10000 hours of energy usage in america

![predict10000](https://github.com/Bayusatriapersada/America-Energy-Usage-Stochastic-ARIMA-/blob/master/Image/Predict%2010000%20hours.png)

after that we gonna make the auto arima, because in real industrial testing, we cant waste time on finding p d q value manually, so we use  this library

```
from statsmodels.tsa.arima_model import ARIMA
import pmdarima as pm
```
so we can use the auto arima
and here is the code to find the best optional p , d, q but limited to 3 because this is only for research purposes

```
df2 = pd.read_csv('https://raw.githubusercontent.com/Bayusatriapersada/America-Energy-Usage-Stochastic-ARIMA-/master/Energy%20Usage%20Data.csv', names=['AEP'],header=0)

model = pm.auto_arima(df2.AEP, start_p=1, start_q=1,
                      test='adf',       # use adftest to find optimal 'd'
                      max_p=3, max_q=3, # maximum p and q
                      m=1,              # frequency of series
                      d=None,           # let model determine 'd'
                      seasonal=False,   # No Seasonality
                      start_P=0, 
                      D=0, 
                      trace=True,
                      error_action='ignore',  
                      suppress_warnings=True, 
                      stepwise=True)

print(model.summary())
```
this the model summary

![modelauto](https://github.com/Bayusatriapersada/America-Energy-Usage-Stochastic-ARIMA-/blob/master/Image/automodel.png)

then we got the data we need

![finaltest](https://github.com/Bayusatriapersada/America-Energy-Usage-Stochastic-ARIMA-/blob/master/Image/Data.png)

from the image above we got the data we need, the residual data, the corellogram, the skew of the data,density of the data

then we finnaly predict with the auto arima

![finalpredict](https://github.com/Bayusatriapersada/America-Energy-Usage-Stochastic-ARIMA-/blob/master/Image/Final%20Prediction.png)

Thank you

#this is a Open source data #
