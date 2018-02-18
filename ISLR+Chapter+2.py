
# coding: utf-8

# Chapter 2
# 
import numpy as np
import pandas as pd
x=np.array([1,2,3,4])
x
len(x)
del x
np.random.randint(0,10,(3,3))
np.eye(3)
np.empty(3)
# Creating a 2X2 matrix,
# Don't forget the extra square bracket
y=np.array([[12,3],[21,1],[2,2]])
print('the matrix formed is',y)
z=np.matrix('12,13;14,15')
print(z)
print(z.shape)
z[0,1]
import math
for i in range(len(z)):
    for j in range(len(z)):
        print(math.sqrt(z[i,j]))

mat_emp=np.empty((2,2))
for i in range(len(z)):
    for j in range(len(z)):
        mat_emp=(math.sqrt(z[i,j]))
mat_emp
x=np.random.rand(50)
y=x+np.random.normal(50,.1,50)
import matplotlib as mpl       
import matplotlib.pyplot as plt
plt.style.use('seaborn-whitegrid')
x = np.random.rand(100)
y = np.random.rand(100)
get_ipython().magic('matplotlib inline')
plt.scatter(np.array(x),np.array(y))
plt.title('A random haphazard plot')
plt.xlabel('X')
plt.ylabel('Y')
# Creating a 4X4 matrix requires usage of Reshaping of Arrays
grid=np.arange(1,17,1).reshape((4,4))
grid

# Indexing Data

grid[1,3]
Auto=pd.read_csv('C:\Texas A&M Spring Semester\ISLR\Auto.csv')
Auto.shape
Auto.columns
plt.plot(Auto['horsepower'],Auto['mpg'])
# ##Summary Statistics for Auto dataset
Auto.describe()
Auto['mpg'].describe()

