import numpy as np
import csv
from sklearn.cross_validation import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.externals import joblib


class KeyStrokeManager():

    featureName = ['accelerationMagnitudes', 'totalNumberOfDeletions', \
                    'gyroMagnitudes', 'interTapDistances', \
                    'tapDurations', 'userId']
    emotions = {'Happy': 0, 'Neutral': 1, 'Calm': 1, 'Sad': 2, \
                'Angry': 3, 'Anxious': 4}
    uids = {'acsalu': 0, 'co273': 1, 'jean': 2}
    m_path = 'pkl/'


    def __init__(self, data):
        self.model = None
        self.features = KeyStrokeManager.parseFeature(data)
        self.labels = list(map(lambda x: KeyStrokeManager.emotions[x.emotion]\
                                , data))


    @classmethod
    def parseFeature(cls, data, normalize=True):
        [accMag, ttlNODel, gyro, intTapDist, tapDur, uid] = \
        [[getattr(d, feature) for d in data] for feature in cls.featureName]

        aveAccMag, stdAccMag = \
                [np.mean(a) for a in accMag], [np.std(a) for a in accMag]
        aveGyro, stdGyro = \
                [np.mean(g) for g in gyro], [np.std(g) for g in gyro]
        aveIntTapDist, stdIntTapDist = \
                [np.mean(i) for i in intTapDist], [np.std(i) for i in intTapDist]
        aveTapDur, stdTapDur = \
                [np.mean(t) for t in tapDur], [np.std(t) for t in tapDur]
        uidFea = list(map(lambda x: cls.uids[x], uid))

        if normalize:
            features = list(map(cls.normalize, \
                    [aveAccMag, stdAccMag, aveGyro, stdGyro, \
                    aveIntTapDist, stdIntTapDist, \
                    aveTapDur, stdTapDur, uidFea]))
        else:
            features = [aveAccMag, stdAccMag, aveGyro, stdGyro, \
                        aveIntTapDist, stdIntTapDist, \
                        aveTapDur, stdTapDur, uidFea]

        return np.array(features).T


    def normalize(feature):
        std = np.std(feature)
        mean = np.mean(feature)
        if std == 0:
            return feature - mean
        return (feature - mean) / std


    def saveParams(self):
        means = np.mean(self.features, axis=0)
        stds = np.std(self.features, axis=0)
        with open(KeyStrokeManager.m_path + 'params.csv', 'w') as csvfile:
            writer = csv.writer(csvfile, delimiter=',')
            for idx in range(len(self.features[0])):
                m, s = means[idx], stds[idx]
                writer.writerow([m, s])


    def logisticRegression(self):
        self.model = LogisticRegression()


    def svm(self):
        self.model = svm.SVC()


    def naiveBayes(self):
        self.model = GaussianNB()


    def randomForest(self):
        self.model = RandomForestClassifier()


    def crossValidScore(self):
        return cross_val_score(self.model, self.features, self.labels).mean()


    def saveModel(self):
        self.model.fit(self.features, self.labels)
        joblib.dump(self.model, KeyStrokeManager.m_path + 'model.pkl', protocol=2) 
