# -*- coding: utf-8 -*-
"""
Created on Sun Apr 11 10:27:57 2021

@author: JF
"""
# =============================================================================
# IMPORTS
# =============================================================================
import uvicorn
from fastapi import FastAPI
import pandas as pd
import numpy as np
import pickle

temp_scaler = pickle.load(open('tempScaler.pkl', 'rb'))
temp_PCA = pickle.load(open('tempPCA.pkl', 'rb'))
one_hot_encoder = pickle.load(open('OneHotEncoder.pkl', 'rb'))
model_gbm = pickle.load(open('bike_gbm.pkl', 'rb'))

# =============================================================================
# INPUT
# =============================================================================
json_obs = {'datetime':'2011-05-06 12:00:00',
            'season': 2,
            'holiday':0,
            'workingday':1,
            'weather':1,
            'temp':15.54,
            'atemp':18.89,
            'humidity':78,
            'windspeed':3.1}

# =============================================================================
# FUNCTIONS
# =============================================================================
def dataPipe(json_obs):
    """Converts a single json observation into a format that the models is
    expecting"""
    #convert to pandas DF
    df = pd.DataFrame(json_obs, index=[0])
    
    #dates
    df['datetime'] = pd.to_datetime(df['datetime'])
    df['hour'] = df['datetime'].dt.strftime('%H')
    df['month'] = df['datetime'].dt.strftime('%m')
    df['day'] = df['datetime'].dt.strftime('%a')
    df['year'] = df['datetime'].dt.strftime('%Y')
    df = df.drop('datetime', axis=1)
    
    #dummy variables
    toDummy = df.drop(['holiday', 'workingday','temp', 'atemp','humidity', 'windspeed'], axis =1)
    dum_values = one_hot_encoder.transform(toDummy)
    dum_columns = one_hot_encoder.get_feature_names(toDummy.columns)
    dum = pd.DataFrame(dum_values, columns = dum_columns)
    
    df = pd.concat((df[['holiday', 'workingday','temp', 'atemp','humidity', 'windspeed']],dum), axis=1)
    
    #temp Scale & PCA
    temp_scaled = temp_scaler.transform(df[['atemp', 'temp']])
    temp_PC = -temp_PCA.transform(temp_scaled)[:,0]
    
    df['tempPCA'] = temp_PC
    df = df.drop(['temp', 'atemp'], axis=1)
    
    return df

# =============================================================================
# PROCESS
# =============================================================================
df = dataPipe(json_obs)

preds = model_gbm.predict(df)