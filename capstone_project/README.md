# DelhiAQI2019

## Problem Background and Motivation

Delhi's airquality has been extremely poor over a couple of months this year in 2019 and this is a repeated pattern which was observed in 2017 and 2018 as well. Many reasons have been attributed to the increase in air pollution and mainly it is driven by Industrial, vehicle, stuble burning. Thus, our first step should be able to better understand and create forecasts about the impact of air pollution and create awareness.

## Problem Statement

In this analysis, we have obtained a time series dataset with following period (start: 2019-08-18, end: 2019-11-14) and I intend to predict PM2.5 feature using **ARIMAX and ARIMA** modelling approaches.


## Data source

It was difficult to find this data. However, I have sourced it from https://openaq.org/, the recording station is IGI Terminal 3 in Delhi.The air pollution features that are available are measurements of particulate matter and several gases.

* PM10 
* PM2.5
* NO2 (Nitrogen Dioxide)
* CO (Carbon Monoxide)
* O3 (Ozone)

you may also find some location and date features avaiable as well. Although, I have dowloaded a csv for delhi, India but you can download data through their API for not only India but many other countries and cities.

## Result

We find that the model forecasts that in next 30 days we should expect the pollution of pm 2.5 to rise higher. Only If we had more data and preferably from the last 2 years we could have been more confident of this forecast.

We also saw that the MAPE of model1 with regressors is around ~30% which is way better than our univariate model i.e. model2, which gives us a MAPE of ~43% 

## Further Improvements

One essential improvement that is required for this analysis is to get more data and prefereably for at least last 2 years or more. Once we have more data we can try to refit the same model or add seasonal component to it using SARIMA, I would also encourage to try some tree based machine learning techniques such as GBM, XGboost or catboost. One draw back that I see is that the we dont have many features, if we could get more features for the days it would add real value to our forecast. One such feature could be temperature, humidity etc.