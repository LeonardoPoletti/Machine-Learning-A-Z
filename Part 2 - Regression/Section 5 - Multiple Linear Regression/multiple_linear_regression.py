# -*- coding: utf-8 -*-
"""
Created on Mon Apr 30 13:55:12 2018

@author: Leonardo
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Importing the dataset
dataset = pd.read_csv('50_Startups.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 4].values



# Endoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

# Tranformar textos em numeros para facilitar a leitura
labelencoder_X = LabelEncoder()
X[:, 3] = labelencoder_X.fit_transform(X[:, 3]) 

# Separar os paises em outras colunas
# Cria-se varias colunas para expressar os paises.
# Fica como um codigo binario
onehotencoder = OneHotEncoder(categorical_features = [3])
X = onehotencoder.fit_transform(X).toarray()


# Avoiding the Dummy Variable Trap
X = X[:, 1:]


# Splitting the dataset into the Training set an Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)


# Fitting Multiple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)


# Predicting the Test set results
y_pred = regressor.predict(X_test)

# Building the optimal model using Backward Elimination
import statsmodels.formula.api as sm
X = np.append(arr = np.ones((50, 1)).astype(int), values = X, axis = 1)
X_opt = X[:, [0, 1, 2, 3, 4, 5]]
regressor_OLS =  sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()
# Remova a coluna que apresenta o maior Pvalue
X_opt = X[:, [0, 1, 3, 4, 5]]
regressor_OLS =  sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()
# Remova a coluna que apresenta o maior Pvalue
X_opt = X[:, [0, 3, 4, 5]]
regressor_OLS =  sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()
# Remova a coluna que apresenta o maior Pvalue
X_opt = X[:, [0, 3, 5]]
regressor_OLS =  sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()
# Remova a coluna que apresenta o maior Pvalue
X_opt = X[:, [0, 3]]
regressor_OLS =  sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()


















