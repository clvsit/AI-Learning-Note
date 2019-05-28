#!/usr/bin/env python
# coding: utf-8

# In[1]:


from sklearn.feature_selection import RFE
from sklearn.feature_selection import RFECV
from sklearn.linear_model import Lasso
from sklearn.tree import DecisionTreeClassifier


# ### 引入数据集

# In[4]:


from sklearn.datasets import load_boston, load_iris


# ### 回归问题

# In[5]:


dataset_boston = load_boston()
data_boston = dataset_boston.data
target_boston = dataset_boston.target


# In[6]:


rfe = RFE(estimator=Lasso(), n_features_to_select=4)
rfe.fit(data_boston, target_boston)
rfe.support_


# ### 分类问题

# In[7]:


dataset_iris = load_iris()
data_iris = dataset_iris.data
target_iris = dataset_iris.target


# In[9]:


rfe = RFE(estimator=DecisionTreeClassifier(), n_features_to_select=2)
rfe.fit(data_iris, target_iris)
rfe.support_


# ### RFECV

# In[11]:


rfecv = RFECV(estimator=DecisionTreeClassifier())
rfecv.fit(data_iris, target_iris)
rfecv.support_

