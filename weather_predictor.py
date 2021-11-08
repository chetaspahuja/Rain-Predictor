# -*- coding: utf-8 -*-
"""
Created on Tue Sep 21 20:17:41 2021

@author: ASUS
"""
# Logistic regression
import pandas as pd
import opendatasets as od 
# print(od.version())

dataset_url = "https://www.kaggle.com/jsphyg/weather-dataset-rattle-package" 
od.download(dataset_url) 

weather_df = pd.read_csv("C:\GIT\Datasets\weatherAUS.csv")
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

numeric_cols = train_input.select_dtypes(include = np.number).columns.tolist()
catogorical_cols = train_input.select_dtypes('object').columns.tolist() 


