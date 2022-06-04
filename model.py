#!/usr/bin/env python
# coding: utf-8

# In[12]:


import pickle
from sklearn import datasets
import joblib
import pandas as pd
import numpy as np


# In[4]:


iris = datasets.load_iris()


# In[24]:


X, y = iris.data, iris.target


# In[26]:


class Perceptron(object):
    def __init__(self, eta=0.01, n_iter=10):
        self.eta = eta
        self.n_iter = n_iter
    
    def fit(self, X, y):
        self.w_ = np.zeros(1 + X.shape[1])
        self.errors_ = []
        for _ in range(self.n_iter):
            errors = 0
            for xi, target in zip(X,y):
                update=self.eta*(target-self.predict(xi))
                self.w_[1:] += update *xi
                self.w_[0] += update
                errors += int(update != 0.0)
            self.errors_.append(errors)
        return self
    
    
    def net_input(self, X):
        return np.dot(X, self.w_[1:]) + self.w_[0]
    
    def predict(self, X):
        return np.where(self.net_input(X) >= 0, 1, -1)


# In[28]:


Classifier = Perceptron(eta = 0.1, n_iter =10)
Classifier.fit(X,y)


# In[29]:


saved_model = pickle.dumps(Classifier)


# In[30]:


joblib.dump(Classifier,'model.pkl')


# In[ ]:




