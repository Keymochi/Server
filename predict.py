import numpy as np
from parse_rest.connection import register
from parse_rest.datatypes import Object
import key
from manager import KeyStrokeManager
import csv
from sklearn.externals import joblib


register(key.APP_ID, key.REST_API_KEY)
modelPath = KeyStrokeManager.m_path + 'model.pkl'
paramPath = KeyStrokeManager.m_path + 'params.csv'
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
data = KeyStrokeManager.parseFeature(data, normalize=False)
normalizedData = list(map( normalize, data ))
model = joblib.load(modelPath)
results = list(map( lambda x: invEmotions[x], model.predict(normalizedData) ))
print(results)
