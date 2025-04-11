# preprocessing.py

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

def load_data():
    data = pd.read_csv('https://storage.googleapis.com/download.tensorflow.org/data/creditcard.csv')
    data.pop('Time')
    scaler = StandardScaler().fit(data.Amount.values.reshape(-1,1))
    data['Amount'] = scaler.transform(data.Amount.values.reshape(-1,1))
    X = data.drop(['Class'], axis=1)
    y = data['Class']
    return X, y
