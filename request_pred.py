# -*- coding: utf-8 -*-
"""
Uses request lib whilst app is running to get a prediction
"""

import requests

req = requests.get('http://127.0.0.1:8000/')
print(req.text)
print()

json_obs = {'datetime':'2012-05-06 12:00:00',
            'season': 2,
            'holiday':0,
            'workingday':0,
            'weather':1,
            'temp':28.22,
            'atemp':30.89,
            'humidity':35,
            'windspeed':10.2}

req = requests.post('http://127.0.0.1:8000/predict', json=json_obs)
print(req.text)