# -*- coding: utf-8 -*-
"""
Created on Sat Apr 14 14:22:08 2018

@author: Leonardo
"""

# Data Processing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# Importing the dataset
dataset = pd.read_csv('Data.csv')
#   iloc [Linhas - Primeira : Ultima , Colunas - Primeira : Ultima]
#   Iremos setar para o X receber todas as linhas,
#   Porem pegara todas as colunas do dataset menos a ultima.
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 3].values


# Taking care of missing data
from sklearn.preprocessing import Imputer

#    Ira informar qual o valor faltante e se será utilziado a coluna ou linha
#    para verificação.
    
#    Selecionando a função e pressionando Ctrl + I verá os atributos
imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
#   Passando as colunas que apresentam valores faltantes, passe uma coluna
#    para frente pois é um intervalo aberto, o ultimo não conta
imputer = imputer.fit(X[:, 1:3])
#  Irá calcular a medica de toda a coluna e jogar no campo faltante
X[:, 1:3] = imputer.transform(X[:, 1:3])  


# Endoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

# Tranformar textos em numeros para facilitar a leitura
labelencoder_X = LabelEncoder()
X[:, 0] = labelencoder_X.fit_transform(X[:, 0]) 

# Separar os paises em outras colunas
# Cria-se varias colunas para expressar os paises.
# Fica como um codigo binario
onehotencoder = OneHotEncoder(categorical_features = [0])
X = onehotencoder.fit_transform(X).toarray()

labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y) 

# Splitting the dataset into the Training sert and Test set
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2
                                                    , random_state = 0)
# Feature Scaling
from sklearn.preprocessing import StandardScaler

# Escalonando os valores para não apresentavam tanta vavial entre eles.
# Não deixar o salario tão alto quanto a idade .
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)



 

































