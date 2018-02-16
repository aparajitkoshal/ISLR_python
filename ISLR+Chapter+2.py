
# coding: utf-8

# Chapter 2
# 
# 

# In[1]:


import numpy as np
import pandas as pd


# In[2]:


x=np.array([1,2,3,4])


# In[3]:


x


# In[4]:


len(x)


# In[5]:


del x


# In[6]:


np.random.randint(0,10,(3,3))


# In[7]:


np.eye(3)


# In[8]:


np.empty(3)


# Creating a 2X2 matrix,
# Don't forget the extra square bracket

# In[9]:


y=np.array([[12,3],[21,1],[2,2]])
print('the matrix formed is',y)


# In[10]:


z=np.matrix('12,13;14,15')
print(z)


# In[11]:


print(z.shape)
z[0,1]


# In[12]:


import math


# Taking a square root of a whole matrix is a bit of work in Python as compared to R

# In[13]:


for i in range(len(z)):
    for j in range(len(z)):
        print(math.sqrt(z[i,j]))


# Storing this into a matrix is difficult 

# In[14]:


mat_emp=np.empty((2,2))


# In[15]:


for i in range(len(z)):
    for j in range(len(z)):
        mat_emp=(math.sqrt(z[i,j]))


# In[16]:


mat_emp


# In[17]:


x=np.random.rand(50)


# In[18]:


y=x+np.random.normal(50,.1,50)


# In[19]:


##Correlation, a function to be developed


# Now to plot all the functions on Python, we will use Matplotlib

# In[20]:


import matplotlib as mpl       
import matplotlib.pyplot as plt


# In[21]:


plt.style.use('classic') 


# In[22]:


x = np.random.rand(100)
y = np.random.rand(100)


# In[23]:


get_ipython().magic('matplotlib inline')


# In[24]:


plt.plot(np.array(x),np.array(y))
plt.title('A random haphazard plot')
plt.xlabel('X')
plt.ylabel('Y')


# Creating a 4X4 matrix requires usage of Reshaping of Arrays

# In[25]:


grid=np.arange(1,17,1).reshape((4,4))
grid


# Indexing Data
# 

# In[26]:


grid[1,3]


# Loading the dataset

# In[27]:


Auto=pd.read_csv('C:\Texas A&M Spring Semester\ISLR\Auto.csv')


# In[28]:


Auto


# In[29]:


Auto.shape


# In[30]:


Auto.columns


# In[31]:


plt.plot(Auto['horsepower'],Auto['mpg'])


# ##Summary Statistics for Auto dataset

# In[33]:


Auto.describe()


# In[40]:


Auto['mpg'].describe()



# In[1]:


import os


# In[4]:


os.getcwd()

