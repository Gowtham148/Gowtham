#Importing the Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
  
#Importing the dataset
dataset = pd.read_csv('happiness.csv')
x = dataset.iloc[:, 5:].values
y = dataset.iloc[:, 3].values
  
#Splitting the dataset into training set and test set
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2)

#Feature scaling is not necessary as all the datas are in between the values -3 and 3

#Plotting a scatter plot to get an idea of what the data looks like
plt.scatter(x_train[:, 0],y_train)

#Training the Linear Regression model
from sklearn.linear_model import LinearRegression
linear_regression = LinearRegression()
linear_regression.fit(x_train,y_train)

#Predicting the output of the test set
y_linear = linear_regression.predict(x_test)

#Comparing the real values with the predicted ones
print(np.concatenate((y_test.reshape(len(y_test),1),y_linear.reshape(len(y_linear),1)),1))

#Evaluating the r2 score
from sklearn.metrics import r2_score
r2_score(y_test,y_linear)

#Training the model on other regressors in order to evaluate the best one

#Training the Support Vector Regressor
from sklearn.tree import DecisionTreeRegressor
decision_tree = DecisionTreeRegressor()
decision_tree.fit(x_train,y_train)

#Predicting the test set results(decision tree)
y_decision = decision_tree.predict(x_test)

#Comparing the values
print(np.concatenate((y_test.reshape(len(y_test),1),y_decision.reshape(len(y_linear),1)),1))

#Evaluating the r2 score
r2_score(y_test,y_decision)

#Training the SVR model
from sklearn.svm import SVR
support_vector = SVR()
support_vector.fit(x_train,y_train)

#Predicting the test set results(SVR)
y_svm = support_vector.predict(x_test)

#Evaluatuing the r2 score
r2_score(y_test,y_svm)
