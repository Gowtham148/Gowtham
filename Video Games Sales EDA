#Importing the required libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import statistics

#Reading the dataset
dataset = pd.read_csv('../input/videogamesales/vgsales.csv')

#Looking at the first 5 elements of the dataset to get an idea of how it looks 
dataset.head()

#To see how many unique values are in each column
dataset.nunique()

#As we can see here, there are 16598 rows here(as each rank is unique). Other columns have lesser number of unique values.

#Checking for null values
dataset.isnull().sum()

#The only columns here with null values are 'Year' and 'Publisher'. Let us see how to deal with these null values. But first let's see the correlation matrix for this dataset.
dataset.corr()

#Visulaizing the correlation matrix
sns.heatmap(dataset.corr(),annot=True)

#As we can see Year is highly correlated with JP_Sales (ignoring Rank as it doesn't provide any value). So let us fill the missing values of the Year column with respect to the JP_Sales column.

def missing_values(x):
    #In order for us to fill the missing values in the year column with respect to the Genre column we have to find the number of unique values in the Genre column
    #We achieve this by invoking the .unique() function which returns all the 12 unique values in a list
    #We go through all the values and then compare them with the year and find the most occuring year for each Genre and then fill the missing values with the mode
    for i in dataset['Genre'].unique():
        mode = statistics.mode(dataset[dataset['Genre']==i]['Year'])
        dataset.loc[dataset['Genre']==i,'Year'] = dataset[dataset['Genre']==i]['Year'].fillna(mode)
        

#Now let us check the number of missing values in the dataset
missing_values(dataset)
dataset.isnull().sum()

#Let us describe the dataset
dataset.describe()

#Now let us visuaize the global sales of the games in each year. In order to do that we have group the global sales variable with the year variable.

y = dataset.groupby(['Year'])['Global_Sales'].sum()

#Let's see what y looks like
print(y,type(y))

#As we can see the year columns are in float so let;s change that to int and then plot the graph
x = y.index.astype(int)
import matplotlib.pyplot as plt
plt.figure(figsize=(12,8))
sns.barplot(x,y)

#The years are not visible to the eyes. So a better way to visualize this is to rotate the values in the x-axis by 60 degrees.
plt.figure(figsize=(12,8))
c = sns.barplot(x,y)
c.set_xticklabels(labels=x,rotation=60)

#Now the values in the x-axis are clear. As we can see from the graph the global sales in the year 2008 has been the highest followed by 2009. In order to get a better understanding of why this is the case let us see the number of games released each year.

x = dataset['Year'].sort_values().astype(int)
y = list(set(x))
plt.figure(figsize=(12,8))
c = sns.countplot(x)
c.set_xticklabels(labels=y,rotation=60)

#As we can see the number of video games released in the year 2008 has been the highest which directly correlates to its increase in the global sales. This indiactes a direct proportionlity to the number of video games released to the global sales.
