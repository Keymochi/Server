import sys
import numpy as np
from parse_rest.connection import register
from parse_rest.datatypes import Object
import key
from manager import KeyStrokeManager
import csv
from sklearn.externals import joblib


register(key.APP_ID, key.REST_API_KEY)


class DataChunk(Object):
    pass


data = DataChunk.Query.all().order_by('-createdAt').limit(1)

if len(sys.argv) > 1 and sys.argv[1] == '--all-model':
    modelPath = KeyStrokeManager.m_path + 'all.pkl'
    paramPath = KeyStrokeManager.m_path + 'all_params.csv'
else:
    modelPath = KeyStrokeManager.m_path + data[0].userId + '.pkl'
    paramPath = KeyStrokeManager.m_path + data[0].userId + '_params.csv'

keyStrokeManager = KeyStrokeManager()
normalizedFeatures = keyStrokeManager.parseTestFeatures(data, paramPath)
model = joblib.load(modelPath)
predictions = model.predict(normalizedFeatures)
results = [KeyStrokeManager.invEmotions[x] for x in predictions]

print(data[0].userId + ': ' + str(results) + '  ' + str(data[0].createdAt))
