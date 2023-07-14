#!/usr/bin/env python
# coding: utf-8

# # BUISSNESS PROBLEM

# In[8]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# # Load dataset
# 

# In[9]:


df=pd.read_csv("student_info.csv")


# In[10]:


df.shape


# In[7]:


df.head()


# In[11]:


df.tail()


# # Discover and visualize the data to gain insights

# In[9]:


df.info()


# In[10]:


df.describe()


# plt.scatter(x=df.study_hours, y=df.student_marks)
# plt.xlabel("Student Hours")
# plt.ylabel("Student marks")
# plt.title("Scatter Plot of Student Hours vs Student Marks")

# In[12]:


plt.scatter(x=df.study_hours, y=df.student_marks)
plt.xlabel("Student Hours") 
plt.ylabel("Student marks") 
plt.title("Scatter Plot of Student Hours vs Student Marks")


# # Prepare the data for Machine Learning alogorithms

# In[13]:


# Data Cleaning

df.isnull()


# In[14]:


df.isnull().sum()


# In[16]:


df.mean()


# In[17]:


df2=df.fillna(df.mean())


# In[13]:


df2.isnull().sum()


# In[18]:


df2.head()


# In[19]:


# Split Dataset   capital X means in machine learning is matrice and small y means vector

X=df2.drop("student_marks", axis= "columns")
y=df2.drop("study_hours", axis= "columns")
print("shape of X= ", X.shape)
print("shape of y= ", y.shape)


# In[21]:


# TRAIN THE DATA

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size =0.2, random_state=51)
print("shape of X_train=", X_train.shape)
print("shape of y_train=", y_train.shape)
print("shape of X_test=", X_test.shape)
print("shape of y_test=", y_test.shape)


# # Select the model and train it

# In[22]:


# y= mx+c  is equation for staight or linear line  (linear line observes by visuallisation of above scatter graph which shows straight line)

from sklearn.linear_model import LinearRegression
lr=LinearRegression()

lr.fit(X_train, y_train)


# In[23]:


lr.coef_


# In[23]:


lr.intercept_


# In[24]:


m= 3.93
c= 50.44
y= m* 4+c
y


# In[24]:


lr.predict([[4]])[0][0].round(2)

# TESTING

y_pred= lr.predict(X_test)
y_pred
# In[28]:


y_pred= lr.predict(X_test)
y_pred


# In[29]:


# join

pd.DataFrame(np.c_[X_test, y_test, y_pred], columns= ["study_hours", "student_marks_original", "student_marks_predicted"])


# # Fine-tune your model

# In[30]:


# FOR ACCURACY TEST

lr.score(X_test, y_test)


# In[31]:


# HOW WE CHECK THAT WHERE THE 5% LEFT ACCURACY WILL INVOVED


plt.scatter(X_train, y_train)


# In[6]:


plt.scatter(X_test, y_test)
plt.plot(X_train, lr.predict(X_train), color="r")

print("models is trained")
# # Present your solution

# # Save ML Model

# In[27]:


import joblib
joblib.dump(lr, "Student_marks_prediction_model.pkl")


# In[28]:


model=joblib.load("Student_marks_prediction_model.pkl")


# In[29]:


print(model.predict([[5]])[0][0])


# # Launch, Monitor and maintain your system

# In[ ]:





