#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pickle
from flask import Flask
from flask import request
from flask import jsonify
import joblib
from sklearn import linear_model


# In[2]:


app = Flask(__name__)


# In[3]:


@app.route('/api/predict', methods=['GET'])
def get_prediction():
    sepal_length = float(request.args.get('sl'))
    petal_length = float(request.args.get('pl'))
    features = [sepal_length,
                petal_length]
    model = joblib.load('model.pkl')

    predicted_class = int(model.predict([features]))

    return jsonify(features=features, predicted_class=predicted_class)

if __name__ == '__main__':
    app.run(host='0.0.0.0')


# In[ ]:




