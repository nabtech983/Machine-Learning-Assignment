# -*- coding: utf-8 -*-
"""
Created on Wed Dec  4 10:17:01 2019

@author: MUHAMMAD NABEEL
"""

# Simple Linear Regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
data = pd.read_csv('aids.csv')
X = data.iloc[:, :-1].values
y = data.iloc[:, 1].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3, random_state = 0)

# Fitting Simple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Predicting the Test set results
y_pred = regressor.predict(X_test)

# Visualising the Training set results
plt.scatter(X_train, y_train, color ='orange')
plt.plot(X_train, regressor.predict(X_train), color = 'purple')
plt.title('Years Vs Death (Training set)')
plt.xlabel('Death By aids')
plt.ylabel('years')
plt.show()

# Visualising the Test set results
plt.scatter(X_test, y_test, color = 'orange')
plt.plot(X_train, regressor.predict(X_train), color = 'purple')
plt.title('Years Vs Death (Test set)')
plt.xlabel('Death By aids')
plt.ylabel('years')
plt.show()

print(regressor.predict([[2019]]))