
# coding: utf-8

# In[143]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[131]:


df_train = pd.read_csv("train.csv") ## Write down your file path inside inverted commas.


# In[132]:


df_train  = df_train[['Sex','Age','Survived']]


# In[133]:


x_train = df_train.iloc[:,[0,1]].values
y_train = df_train.iloc[:, 2].values


# In[134]:



from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = 'NaN', strategy = 'median', axis = 0)
imputer.fit(x_train[:, [1]])
x_train[:, [1]] = imputer.transform(x_train[:,[1]])


# In[135]:



from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_x = LabelEncoder()
x_train[:,0] = labelencoder_x.fit_transform(x_train[:, 0])




# In[136]:


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)


# In[137]:


df_test = pd.read_csv("test.csv")
df_test = df_test[['Sex','Age']]
x_test = df_test.iloc[:,[0,1]].values
imputer.fit(x_test[:, [1]])
x_test[:, [1]] = imputer.transform(x_test[:,[1]])
x_test[:,0] = labelencoder_x.fit_transform(x_test[:, 0])
x_test = sc.fit_transform(x_test)




# In[138]:


from sklearn.linear_model import LogisticRegression
classifier  = LogisticRegression(random_state = 0)
classifier.fit(x_train, y_train)


# In[162]:


y_pred = classifier.predict(x_test)


# In[163]:


y_pred


# In[156]:


len(y_pred)


# In[164]:


df_test = pd.read_csv("test.csv")
df_test['Predicted'] = y_pred


# In[165]:


df_test['Predicted'].value_counts()
