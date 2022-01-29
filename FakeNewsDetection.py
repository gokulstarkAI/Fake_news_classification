#!/usr/bin/env python
# coding: utf-8

# ## necessary imports

# In[3]:


import numpy as np
import pandas as pd
import itertools
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score, confusion_matrix


# ## reading the data set into data frame and geting the shape of it.
# 
# 

# In[4]:


df = pd.read_csv('news.csv')
df.shape


# In[5]:


df.head()


# ## Getting labels from the data frame.

# In[6]:


labels = df.label
labels.head()


# ## spliting the data into training and testing sets.

# In[7]:


x_train,x_test,y_train,y_test = train_test_split(df['text'], labels, test_size=0.2, random_state=7)


# ## Fit and transform the vectorizer on the train set and transform the vectorizer on the test set.

# In[8]:


tfidf_vectorizer = TfidfVectorizer(stop_words='english',max_df=0.8)
tfidf_train = tfidf_vectorizer.fit_transform(x_train)
tfidf_test = tfidf_vectorizer.transform(x_test)


# ## Initialize PassiveAggressiveClassifier, fit tfidf_train and y_train

# In[9]:


pac = PassiveAggressiveClassifier(max_iter=50)
pac.fit(tfidf_train, y_train)


# ## predict

# In[10]:


y_pred = pac.predict(tfidf_test)
score = accuracy_score (y_test, y_pred)
print(f'Accuracy: {round(score*100,2)}%')


# In[11]:


confusion_matrix(y_test, y_pred, labels = ['FAKE','REAL'])


# ## so with this model, we have 590 true positive, 586 true negative, 48 false positive and 43 false negative.

# In[ ]:




