#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


train = pd.read_csv('titanic_train.csv')
train.isnull()


# In[3]:


sns.heatmap(train.isnull(),yticklabels=False,cbar=False,cmap='viridis')


# In[4]:


sns.countplot(x='Survived',data=train)


# In[5]:


sns.countplot(x='Survived',hue='Gender',data=train,palette='RdBu_r')


# In[6]:


sns.countplot(x='Survived',hue='Pclass',data=train,palette='rainbow')


# In[7]:


sns.distplot(train['Age'].dropna(),kde=False,color='darkred',bins=40)


# In[8]:


train['Age'].hist(bins=30,color='darkred',alpha=0.3)


# In[9]:


sns.countplot(x='SibSp',data=train)


# In[10]:


train['Fare'].hist(color='green',bins=40,figsize=(8,4))


# In[11]:


plt.figure(figsize=(12, 7))
sns.boxplot(x='Pclass',y='Age',data=train,palette='winter')


# In[12]:


def impute_age(cols):
    Age = cols[0]
    Pclass = cols[1]
    
    if pd.isnull(Age):

        if Pclass == 1:
            return 37

        elif Pclass == 2:
            return 29

        else:
            return 24

    else:
        return Age


# In[13]:


train['Age'] = train[['Age','Pclass']].apply(impute_age,axis=1)


# In[14]:


sns.heatmap(train.isnull(),yticklabels=False,cbar=False,cmap='viridis')


# In[15]:


train.drop('Cabin',axis=1,inplace=True)


# In[16]:


train.dropna(inplace=True)


# In[17]:


train.head()


# In[18]:


train.info()


# In[19]:


pd.get_dummies(train['Embarked'],drop_first=True).head()


# In[20]:


sex = pd.get_dummies(train['Gender'],drop_first=True)
embark = pd.get_dummies(train['Embarked'],drop_first=True)


# In[21]:


train.drop(['Gender','Embarked','Name','Ticket'],axis=1,inplace=True)


# In[22]:


train.head()


# In[23]:


train = pd.concat([train,sex,embark],axis=1)


# In[24]:


train.head()


# In[25]:


train.drop('Survived',axis=1).head()


# In[26]:


train['Survived'].head()


# In[28]:


from sklearn.model_selection import train_test_split


# In[29]:


X_train, X_test, y_train, y_test = train_test_split(train.drop('Survived',axis=1), 
                                                    train['Survived'], test_size=0.30, 
                                                    random_state=101)


# In[30]:


from sklearn.linear_model import LogisticRegression
logmodel = LogisticRegression()
logmodel.fit(X_train,y_train)


# In[31]:


predictions = logmodel.predict(X_test)


# In[32]:


from sklearn.metrics import confusion_matrix

accuracy=confusion_matrix(y_test,predictions)


# In[33]:


accuracy


# In[34]:


from sklearn.metrics import accuracy_score
accuracy


# In[35]:


predictions


# In[36]:


from sklearn.metrics import classification_report
print(classification_report(y_test,predictions))


# In[ ]:




