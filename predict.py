import sys
import numpy as np
from parse_rest.connection import register
from parse_rest.datatypes import Object
import key
from manager import KeyStrokeManager
import csv
from sklearn.externals import joblib


register(key.APP_ID, key.REST_API_KEY)
invEmotions = {v: k for (k,v) in KeyStrokeManager.emotions.items()}


class DataChunk(Object):
    pass


def normalize(d):
    features = []
    fName = KeyStrokeManager.featureName
    with open(paramPath, 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        for idx, row in enumerate(reader):
            mean, std = float(row[0]), float(row[1])
            if std == 0:
                features.append(d[idx] - mean)
            else:
                features.append( (d[idx] - mean) / std )
    return features


data = DataChunk.Query.all().order_by('-createdAt').limit(1)

if len(sys.argv) > 1 and sys.argv[1] == 'model-all':
    modelPath = KeyStrokeManager.m_path + 'all.pkl'
    paramPath = KeyStrokeManager.m_path + 'all_params.csv'
else:
    modelPath = KeyStrokeManager.m_path + data[0].userId + '.pkl'
    paramPath = KeyStrokeManager.m_path + data[0].userId + '_params.csv'

pData = KeyStrokeManager.parseFeature(data, normalize=False)
nData = list(map( normalize, pData ))
model = joblib.load(modelPath)
results = list(map( lambda x: invEmotions[x], model.predict(nData) ))

print(data[0].userId + ': ' + str(results))
