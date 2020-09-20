#Importing the required libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

#Importing the dataset
dataset = pd.read_csv('HeartFailure.csv')

#ANALYZING THE DATA
#Looking at the data
dataset.head()
dataset.tail()
dataset.describe()
dataset.info()

#Checking for null values
dataset.isnull().sum()

#Checking for unique values
dattaset.nunique()

sns.lineplot(dataset['ejection_fraction'],dataset['platelets'],hue=dataset['sex'],ci=None)
sns.lineplot(dataset['ejection_fraction'],dataset['platelets'],hue=dataset['DEATH_EVENT'],ci=None)

#Correlation Matrix
dataser.corr()

#Visualizing the correlation matrix using a heatmap
sns.heatmap(dataset.corr(),annot=True)

#Looking at the distribution of age
sns.distplot(dataset['age'])

#Distribution of creatinine_phosphokinase
sns.distplot(dataset['creatinine_phosphokinase'])

#LOGISTIC REGRESSION
#Creating a dictionary for creating a new dataset in order for us to scale the data easier by ordering the categorical variables on one side of the dataset and the others on the other side
dict = {'anaemia': dataset.iloc[:, 1], 'diabetes':dataset.iloc[:, 3], 'high_blood_pressure':dataset.iloc[:, 5], 'sex':dataset.iloc[:, 9], 'smoking':dataset.iloc[:, 10], 'age':dataset.iloc[:, 0], 'creatinine_phosphokinase':dataset.iloc[:, 2], 'ejection_fraction':dataset.iloc[:, 4], 'platelets':dataset.iloc[:, 6], 'serum_xreatinine':dataset.iloc[:, 7], 'serum_sodium':dataset.iloc[:, 8], 'time':dataset.iloc[:, 11]}
new_dataset = pd.DataFrame(dict)

#Features
x = new_dataset.values
y = dataset.iloc[:,-1].values

#Splitting the dataset into training and test set
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2)

#Scaling the features
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
x_train[:,5:] = scaler.fit_transform(x_train[:,5:])
x_test[:, 5:] = scaler.fit_transform(x_test[:, 5:])

#Fitting the logistic model to the features
from sklearn.linear_model import LogisticRegression
logistic = LogisticRegression()
logistic.fit(x_train,y_train)
y_pred = logistic.predict(x_test)

#Evaluating the model
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test,y_pred)
print(cm,accuracy_score(y_test,y_pred))

#NEURAL NETWORK
#Importing tensorflow
import tensorflow as tf

#Creating the neural network
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Dense(100,activation='relu'))
model.add(tf.keras.layers.Dense(1,activation='sigmoid'))

#Compiling the model
model.compile(optimizer='SGD',loss='binary_crossentropy',metrics=['accuracy'])

#Fitting the model
model.fit(x_train,y_train,epochs=50,batch_size=64)
model.evaluate(x_test,y_test,batch_size=64)

#Trying to improve the model
#Improving the model with hidden layers
ann = tf.keras.models.Sequential()
ann.add(tf.keras.layers.Dense(100,activation='relu'))
ann.add(tf.keras.layers.Dense(100,activation='relu'))
ann.add(tf.keras.layers.Dense(100,activation='relu'))
ann.add(tf.keras.layers.Dense(1,activation='sigmoid'))
ann.compile(optimizer='SGD',loss='binary_crossentropy',metrics=['accuracy'])
ann.fit(x_train,y_train,epochs=50,batch_size=64)
ann.evaluaye(x_test,y_test,batch_size=64)

#Adding dropout layers to improve the model
new_ann = tf.keras.models.Sequential()
new_ann.add(tf.keras.layers.Dense(100,activation='relu'))
new_ann.add(tf.keras.layers.Dropout(0.2))
new_ann.add(tf.keras.layers.Dense(500,activation='relu'))
new_ann.add(tf.keras.layers.Dropout(0.25))
new_ann.add(tf.keras.layers.Dense(100,activation='relu'))
new_ann.add(tf.keras.layers.Dense(1,activation='sigmoid'))
new_ann.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])

new_ann.fit(x_train,y_train,epochs=50,batch_size=64)
new_ann.evaluate(x_test,y_test,batch_size=64)
