# Logistic regression
import joblib
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
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

weather_df.dropna(subset=['RainToday', 'RainTomorrow'], inplace=True)
# weather_df.shape

# %matplotlib inline

sns.set_style('darkgrid')
matplotlib.rcParams['font.size'] = 14
matplotlib.rcParams['figure.figsize'] = (10, 6)
matplotlib.rcParams['figure.facecolor'] = '#00000000'


fig = px.histogram(weather_df, x='Location',
                   title='Location vs Rainy Day', color='RainToday')
# fig.show()

fig = px.histogram(weather_df, x='Temp3pm',
                   title='Temperature at 3 pm vs. Rain Tomorrow', color='RainTomorrow')
# fig.show()

fig = px.histogram(weather_df, x='RainToday',
                   title='RainTOday vs RainTomorrow', color='RainTomorrow')
# fig.show()

fig = px.scatter(weather_df, x='MinTemp', y='MaxTemp',
                 color='RainToday', title='MinTemp Vs MaxTemp')
# fig.show()

fig = px.scatter(weather_df.sample(2000), x='Temp3pm', y='Humidity3pm',
                 color='Humidity3pm', title='Temp vs Humidity @3pm')
print(fig.show())
# .show()


# train_val_df , test_df = train_test_split(weather_df , test_size = 0.2)
# train_df , val_df = train_test_split(train_val_df , test_size = 0.25)

# print(test_df.shape)
# print(train_df.shape)
# print(val_df.shape)

plt.title('NO OF ROWS PER YEAR')
sns.countplot(x=pd.to_datetime(weather_df.Date).dt.year)

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
numeric_cols = train_input.select_dtypes(include=np.number).columns.tolist()
catogorical_cols = train_input.select_dtypes('object').columns.tolist()

# ---------------------------------------------------------------------------------
# cleaning numerical data (filling missing values and scaling the numbers)

# creating an imputer object
imputer = SimpleImputer(strategy='mean')

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

# ?OneHotEncoder

weather_df2 = weather_df[catogorical_cols].fillna('Unknown')
encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')

encoder.fit(weather_df2[catogorical_cols])

encoder.categories_
encoded_cols = list(encoder.get_feature_names(catogorical_cols))
print(encoded_cols)

train_input[encoded_cols] = encoder.transform(
    train_input[catogorical_cols].fillna('Unknown'))
val_input[encoded_cols] = encoder.transform(
    val_input[catogorical_cols].fillna('Unknown'))
test_input[encoded_cols] = encoder.transform(
    test_input[catogorical_cols].fillna('Unknown'))

test_input[encoded_cols]

print('train input :', train_input.shape)
print('train target :', train_target.shape)

print('val input :', val_input.shape)
print('val target :', val_target.shape)

print('test input :', test_input.shape)
print('test target :', test_target.shape)

# Saving these modified files in parquet form(from pandas) (pyarrow reqd)

# !pip install pyarrow --quiet

train_input.to_parquet('train_input.parquet')
val_input.to_parquet('val_input.parquet')
test_input.to_parquet('test_input.parquet')

pd.DataFrame(train_target).to_parquet('train_target.parquet')
pd.DataFrame(val_target).to_parquet('val_target.parquet')
pd.DataFrame(test_target).to_parquet('test_target.parquet')


model = LogisticRegression(solver='liblinear')
model.fit(train_input[numeric_cols + encoded_cols], train_target)

print(model.coef_.tolist())

weight_df = pd.DataFrame(
    {'feature': (numeric_cols + encoded_cols), 'weights': model.coef_.tolist()[0]})

print('Intercept of the Model :', model.intercept_)

# plt.figure(figsize = (50 , 70))
sns.barplot(data=weight_df, x='weights', y='feature')
sns.barplot(data=weight_df.sort_values(
    'weights', ascending=False).head(10), x='weights', y='feature')


x_train = train_input[numeric_cols + encoded_cols]
x_val = val_input[numeric_cols + encoded_cols]
x_test = test_input[numeric_cols + encoded_cols]

train_preds = model.predict(x_train)

# predict_proba in logisitic regression gives out probabiltites of raining tmrw
train_probability = model.predict_proba(x_train)
# train_probability

# gives out accuracy of the model
accuracy_score(train_target, train_preds)

confusion_matrix(train_target, train_preds, normalize='true')


def predictPlot(inputs, targets, name=''):
    preds = model.predict(inputs)
    accuracy = accuracy_score(targets, preds)
    print("Accuracy : {:.2f}%".format(accuracy * 100))

    cf = confusion_matrix(targets, preds, normalize='true')
    plt.figure()
    sns.heatmap(cf, annot=True)
    plt.xlabel('Prediction')
    plt.ylabel('Target')
    plt.title('{} Confusion Matrix'.format(name))
    return preds


val_preds = predictPlot(x_val, val_target, 'Validation')
train_preds = predictPlot(x_train, train_target, 'Training')

# -------------------------------------------------------------------
# testing the model on a random input

new_input = {'Date': '2021-06-19',
             'Location': 'Katherine',
             'MinTemp': 23.2,
             'MaxTemp': 33.2,
             'Rainfall': 10.2,
             'Evaporation': 4.2,
             'Sunshine': np.nan,
             'WindGustDir': 'NNW',
             'WindGustSpeed': 52.0,
             'WindDir9am': 'NW',
             'WindDir3pm': 'NNE',
             'WindSpeed9am': 13.0,
             'WindSpeed3pm': 20.0,
             'Humidity9am': 89.0,
             'Humidity3pm': 58.0,
             'Pressure9am': 1004.8,
             'Pressure3pm': 1001.5,
             'Cloud9am': 8.0,
             'Cloud3pm': 5.0,
             'Temp9am': 25.7,
             'Temp3pm': 33.0,
             'RainToday': 'Yes'}

newInput_df = pd.DataFrame([new_input])

newInput_df[numeric_cols] = imputer.transform(newInput_df[numeric_cols])
newInput_df[numeric_cols] = scaler.transform(newInput_df[numeric_cols])
newInput_df[encoded_cols] = encoder.transform(newInput_df[catogorical_cols])

xNewInput = newInput_df[numeric_cols+encoded_cols]
type(xNewInput)

predicition = model.predict(xNewInput)[0]
print('Prediction for Input :', predicition)


# Creating a function for model to work for a single new input value

def predictForAnInput(singleInput):
    input_df = pd.DataFrame([singleInput])
    input_df[numeric_cols] = imputer.transform(input_df[numeric_cols])
    input_df[numeric_cols] = scaler.transform(input_df[numeric_cols])
    input_df[encoded_cols] = encoder.transform(input_df[catogorical_cols])
    xInput = input_df[numeric_cols + encoded_cols]
    pred = model.predict(xInput)[0]
    prob = model.predict_proba(xInput)[0][list(model.classes_).index(pred)]
    return pred, prob


new_input = {'Date': '2021-06-19',
             'Location': 'Launceston',
             'MinTemp': 23.2,
             'MaxTemp': 33.2,
             'Rainfall': 10.2,
             'Evaporation': 4.2,
             'Sunshine': np.nan,
             'WindGustDir': 'NNW',
             'WindGustSpeed': 52.0,
             'WindDir9am': 'NW',
             'WindDir3pm': 'NNE',
             'WindSpeed9am': 13.0,
             'WindSpeed3pm': 20.0,
             'Humidity9am': 89.0,
             'Humidity3pm': 58.0,
             'Pressure9am': 1004.8,
             'Pressure3pm': 1001.5,
             'Cloud9am': 8.0,
             'Cloud3pm': 5.0,
             'Temp9am': 25.7,
             'Temp3pm': 33.0,
             'RainToday': 'Yes'}

predictForAnInput(new_input)

# Saving model , all the proccess and parameters(weights and )


weather_predictor = {
    'model': model,
    'imputer': imputer,
    'scaler': scaler,
    'input_cols': input_cols,
    'target_cols': target_cols,
    'numeric_cols': numeric_cols,
    'categorical_cols': catogorical_cols,
    'encoded_cols': encoded_cols
}

joblib.dump(weather_predictor, 'weatherPredictor.joblib')
