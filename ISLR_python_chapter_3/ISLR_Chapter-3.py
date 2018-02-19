# -*- coding: utf-8 -*-
"""
Created on Sun Feb 18 11:30:47 2018

@author: Aparajit Koshal
"""

import numpy as np
import pandas as pd

##ISLR Chapter 3 
boston=pd.read_excel('C:\Texas A&M Spring Semester\ISLR\MASS.xlsx')
boston.columns
boston.describe()

##Counting null values
boston.isnull().sum()
##There are no null values
##Performing linear regression
from sklearn.linear_model import LinearRegression

###Not dividing the dataset into different sets
y=boston['medv'].values
y.shape
x=boston[['lstat']].values##Please note that there is two dimensional list function used
x##returns a pandas dataframe and not a series
regression = LinearRegression()
regression.fit(x,y)
##As can be seen from the ISLR, the results for regression are similar
print('The estimated equation of the regression line is: {} + {} * x'.format(regression.intercept_, regression.coef_[0]))
regression_line = lambda x: regression.intercept_ + regression.coef_ * x

##Plotting a least square regression line 
import matplotlib.pyplot as plt
%matplotlib inline
plt.style.use('seaborn-whitegrid')
##Advanced plotting # create an axes object in the figure
fig, ax = plt.subplots(1, 1, figsize=(10, 10))
x_val=np.linspace(0,40,100)
ax.plot(x_val,regression_line(x_val),color='red', linewidth=1, label='regression line')
ax.scatter(x,y,color='grey',alpha=0.5)
ax.set_xlabel('lstat')
ax.set_ylabel('medv')
ax.set_title('Regression of medv~lstat')

##We can get the similar table as present in R
import statsmodels.api as sm
X2 = sm.add_constant(x)
X2##A constant 1 has been added to the predictor which aligns with assumptions of OLS
est = sm.OLS(y, X2)
est2 = est.fit()
print(est2.summary())

##Confidence Interval

y.

z_critical = stats.norm.ppf(q = 0.975)  # Get the z-critical value*

print("z-critical value:")              # Check the z-critical value
print(z_critical)                        

pop_stdev = population_ages.std()  # Get the population standard deviation

margin_of_error = z_critical * (pop_stdev/math.sqrt(sample_size))

confidence_interval = (sample_mean - margin_of_error,
                       sample_mean + margin_of_error)  

print("Confidence interval:")
print(confidence_interval)
