# -*- coding: utf-8 -*-
"""
Uses FastAPI to deploye the bike share model
"""
# =============================================================================
# IMPORTS
# =============================================================================
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
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
json_obs = {'datetime':'2012-05-06 12:00:00',
            'season': 2,
            'holiday':0,
            'workingday':0,
            'weather':1,
            'temp':28.22,
            'atemp':30.89,
            'humidity':35,
            'windspeed':10.2}

# =============================================================================
# #as BaseModel
# basey = conditions(datetime='2012-05-06 12:00:00',
#             season= 2,
#             holiday=0,
#             workingday=0,
#             weather=1,
#             temp=28.22,
#             atemp=30.89,
#             humidity=35,
#             windspeed=10.2)
#
# json_obs == basey.dict()
# =============================================================================

# =============================================================================
# FUNCTIONS & APP
# =============================================================================
class conditions(BaseModel):
    """Model input via API"""
    datetime: str
    season: int
    holiday: int
    workingday: int
    weather: int
    temp: float
    atemp: float
    humidity: int
    windspeed: float

def dataPipe(json_obs, temp_scaler, temp_PCA, one_hot_encoder):
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

app = FastAPI()

@app.get('/')
def home():
    return {'message': 'Hello, Please call predict with expected JSON input'}

@app.post('/predict')
def predict(data: conditions):
    data = data.dict()
    df = dataPipe(data, temp_scaler, temp_PCA, one_hot_encoder)
    pred = model_gbm.predict(df)[0]
    return {'prediction': pred}

# =============================================================================
# PROCESS
# =============================================================================
if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)
    
#uvicorn app:app --reload    