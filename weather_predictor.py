# -*- coding: utf-8 -*-
"""
Created on Tue Sep 21 20:17:41 2021

@author: ASUS
"""
# Logistic regression
import numpy as np
import pandas as pd
import opendatasets as od 
# print(od.version())

dataset_url = "https://www.kaggle.com/jsphyg/weather-dataset-rattle-package" 
od.download(dataset_url) 

weather_df = pd.read_csv("C:/ML/Taking-a-Rain-Check/weatherAUS.csv")
# print(weather_df.shape)
# print(weather_df.info())
# print(weather_df.describe()) 

weather_df.dropna(subset = ['RainToday' , 'RainTomorrow'] , inplace = True) 
# weather_df.shape 

import matplotlib 
import matplotlib.pyplot as plt 
import seaborn as sns 
import plotly.express as px 
# %matplotlib inline

sns.set_style('darkgrid') 
matplotlib.rcParams['font.size'] = 14
matplotlib.rcParams['figure.figsize'] = (10,6) 
matplotlib.rcParams['figure.facecolor'] = '#00000000'


fig = px.histogram(weather_df , x = 'Location' , title= 'Location vs Rainy Day' , color = 'RainToday')
# fig.show()

fig = px.histogram(weather_df , x = 'Temp3pm' , title= 'Temperature at 3 pm vs. Rain Tomorrow' , color = 'RainTomorrow')
# fig.show() 

fig = px.histogram(weather_df , x = 'RainToday' , title = 'RainTOday vs RainTomorrow'  , color = 'RainTomorrow') 
# fig.show() 

fig = px.scatter(weather_df , x = 'MinTemp' , y = 'MaxTemp' , color = 'RainToday' , title = 'MinTemp Vs MaxTemp')
# fig.show()  

fig = px.scatter(weather_df.sample(2000) , x = 'Temp3pm' , y = 'Humidity3pm' , color = 'Humidity3pm' , title = 'Temp vs Humidity @3pm')
print(fig.show()) 
# .show()  

import sklearn 
from sklearn.model_selection import train_test_split

# train_val_df , test_df = train_test_split(weather_df , test_size = 0.2)
# train_df , val_df = train_test_split(train_val_df , test_size = 0.25)

# print(test_df.shape)
# print(train_df.shape)
# print(val_df.shape)

plt.title('NO OF ROWS PER YEAR') 
sns.countplot(x = pd.to_datetime(weather_df.Date).dt.year)

year = pd.to_datetime(weather_df.Date).dt.year 

train_df = weather_df[year < 2015] 
val_df = weather_df[year == 2015] 
test_df = weather_df[year > 2015] 

# print(train_df.head()) 
# print(test_df.head()) 
# print(val_df.head()) 

# train_df
# val_df
# test_df

# IDENTIFYING INPUT AND TARGET COLUMNS 
train_df.info()

# 0th column is the date and the last column of train_df is actually is target column, hence we dont include them while working as it can lead to 100% accuracy 

input_cols = list(train_df.columns)[1:-1] 
target_cols = 'RainTomorrow' 

print(input_cols)

train_input = train_df[input_cols].copy()
train_target = train_df[target_cols].copy()

val_input = train_df[input_cols].copy()
val_target = train_df[target_cols].copy()

test_input = train_df[input_cols].copy()
test_target = train_df[target_cols].copy()

# type(val_input)

# separating numerical and catergorical data 
numeric_cols = train_input.select_dtypes(include = np.number).columns.tolist()
catogorical_cols = train_input.select_dtypes('object').columns.tolist() 

# ---------------------------------------------------------------------------------
# cleaning numerical data (filling missing values and scaling the numbers)
from sklearn.impute import SimpleImputer 

# creating an imputer object 
imputer = SimpleImputer(strategy = 'mean') 

weather_df[numeric_cols].isna().sum()

# calculates the mean(in this case) of the colums we require to fill missing values in
imputer.fit(weather_df[numeric_cols])
print(numeric_cols)

# the mean of every column in the numeric_cols
list(imputer.statistics_) 

imputer.transform(train_input[numeric_cols]) 

val_input[numeric_cols]

train_input[numeric_cols] = imputer.transform(train_input[numeric_cols]) 
val_input[numeric_cols] = imputer.transform(val_input[numeric_cols])
test_input[numeric_cols] = imputer.transform(test_input[numeric_cols])

# train_input[numeric_cols]

train_input[numeric_cols].isna().sum()
val_input[numeric_cols].isna().sum()
test_input[numeric_cols].isna().sum()

# Scaling the data to give equal dominance/effect in the final result
# Usually columns having large range of values effects the result more
# brings the data in the range of 0 to 1 
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler() 

scaler.fit(weather_df[numeric_cols])

print('minimum')
list(scaler.data_min_)

print('maximum')
list(scaler.data_max_)

train_input[numeric_cols] = scaler.transform(train_input[numeric_cols])
val_input[numeric_cols] = scaler.transform(val_input[numeric_cols])
test_input[numeric_cols] = scaler.transform(test_input[numeric_cols])

# train_input.describe()

# --------------------------------------------------------------
# encoding catogorical data for numerical computation 

weather_df[catogorical_cols].nunique()

from sklearn.preprocessing import OneHotEncoder 
# ?OneHotEncoder

weather_df2 = weather_df[catogorical_cols].fillna('Unknown') 
encoder = OneHotEncoder(sparse = False , handle_unknown= 'ignore')

encoder.fit(weather_df2[catogorical_cols]) 

encoder.categories_
encoded_cols = list(encoder.get_feature_names(catogorical_cols))
print(encoded_cols)

train_input[encoded_cols] = encoder.transform(train_input[catogorical_cols].fillna('Unknown'))
val_input[encoded_cols] = encoder.transform(val_input[catogorical_cols].fillna('Unknown'))
test_input[encoded_cols] = encoder.transform(test_input[catogorical_cols].fillna('Unknown'))



