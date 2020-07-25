#Importing the Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn sa sn

#Importing the dataset
dataset = pd.read_csv('toy_dataset.csv')

#Knowing the dataset

#Shows the mean, standard deviation of each variable
dataset.describe()

#Shape of the dataset
dataset.shape

#Distinct values of each variable
dataset.nunique()

#Names of each column in the dataset
dataset.columns

#First 5 samples
dataset.head()

#Distinct values of a specific variable
dataset.City.unique()

#Boxplot
sn.boxplot(x=dataset['Income'])
