#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns


# In[2]:


ipl=pd.read_csv('matches.csv')


# In[3]:


ipl.head(11)


# In[4]:


ipl.tail(5)


# In[5]:


ipl.shape


# In[6]:


ipl['player_of_match'].value_counts()[0:6]


# In[7]:


ipl['player_of_match'].value_counts()[-5:]


# In[8]:


ipl['player_of_match'].value_counts()[0:5]


# In[9]:


plt.figure(figsize=(15,5))
plt.bar(list(ipl['player_of_match'].value_counts()[0:5].keys()),list(ipl['player_of_match'].value_counts()[0:5]))
plt.show()


# In[10]:


ipl['result'].value_counts()


# In[11]:


ipl['toss_winner'].value_counts()


# In[12]:


batting_first=ipl[ipl['win_by_runs']!=0]
batting_first.head()


# In[13]:


batting_first.head()


# In[14]:


plt.figure(figsize=(7,7))
plt.hist(batting_first['win_by_runs'])
plt.show()


# In[15]:


batting_first['winner'].value_counts()


# In[16]:


plt.figure(figsize=(7,7))
plt.bar(list(batting_first['winner'].value_counts()[0:3].keys()),list(batting_first['winner'].value_counts()[0:3]),color=["blue","yellow","orange"])
plt.show()


# In[17]:


plt.figure(figsize=(9,9))
plt.pie(list(batting_first['winner'].value_counts()),labels=list(batting_first['winner'].value_counts().keys()),autopct='%0.1f%%')
plt.show()


# In[20]:


batting_second=ipl[ipl['win_by_wickets']!=0]


# In[21]:


batting_second.head()


# In[22]:


plt.figure(figsize=(7,7))
plt.hist(batting_second['win_by_wickets'],bins=30)
plt.show()


# In[23]:


batting_second['winner'].value_counts()


# In[24]:


plt.figure(figsize=(7,7))
plt.bar(list(batting_second['winner'].value_counts()[0:3].keys()),list(batting_second['winner'].value_counts()[0:3]),color=["blue","green","orange"])
plt.show()


# In[25]:


plt.figure(figsize=(7,7))
plt.pie(list(batting_second['winner'].value_counts()),labels=list(batting_second['winner'].value_counts().keys()),autopct='%0.1f%%')
plt.show()


# In[26]:


ipl['season'].value_counts()


# In[27]:


ipl['city'].value_counts()


# In[28]:


import numpy as np
np.sum(ipl['toss_winner']==ipl['winner'])


# In[29]:


325/636


# In[ ]:




