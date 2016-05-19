import numpy as np
import csv
from sklearn.cross_validation import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.externals import joblib
from sklearn.metrics import roc_curve, auc
from sklearn.cross_validation import train_test_split
from sklearn.cross_validation import StratifiedKFold
from scipy import interp
from  matplotlib import pyplot as plt


class KeyStrokeManager():

    featureName = ['accelerationMagnitudes', 'totalNumberOfDeletions', \
                    'gyroMagnitudes', 'interTapDistances', \
                    'tapDurations', 'symbol_punctuation', 'userId']
    emotions = {'Happy': 0, 'Calm': 1, 'Sad': 2, \
                'Angry': 3, 'Anxious': 4}
    invEmotions = {v: k for (k,v) in emotions.items()}
    uids = {'acsalu': 0, 'co273': 1, 'jean': 2}
    m_path = 'pkl/'


    def __init__(self, data):
        self.model = None
        self.raw, self.features = KeyStrokeManager.parseFeature(data)
        self.labels = list(map(lambda x: KeyStrokeManager.emotions[x.emotion]\
                                , data))
        self.n_classes = len(np.unique(self.labels))


    @classmethod
    def parseFeature(cls, data, train=True):
        [accMag, ttlNODel, gyro, intTapDist, tapDur, puncCount, uid] = \
        [[getattr(d, feature) for d in data] for feature in cls.featureName]

        aveAccMag, stdAccMag = \
                [np.mean(a) for a in accMag], [np.std(a) for a in accMag]
        aveGyro, stdGyro = \
                [np.mean(g) for g in gyro], [np.std(g) for g in gyro]
        aveIntTapDist, stdIntTapDist = \
                [np.mean(i) for i in intTapDist], [np.std(i) for i in intTapDist]
        aveTapDur, stdTapDur = \
                [np.mean(t) for t in tapDur], [np.std(t) for t in tapDur]
        avePunc, stdPunc = \
                [np.mean(p) for p in puncCount], [np.std(p) for p in puncCount]
        uidFea = list(map(lambda x: cls.uids[x], uid))


        raw = [aveAccMag, stdAccMag, aveGyro, stdGyro, \
                aveIntTapDist, stdIntTapDist, \
                aveTapDur, stdTapDur, avePunc, stdPunc, uidFea]

        if train:
            features = list(map(cls.normalize, \
                    [aveAccMag, stdAccMag, aveGyro, stdGyro, \
                    aveIntTapDist, stdIntTapDist, \
                    aveTapDur, stdTapDur, avePunc, stdPunc, uidFea]))

        if train:
            return (np.array(raw).T, np.array(features).T)
        else:
            return np.array(raw).T


    def normalize(feature):
        std = np.std(feature)
        mean = np.mean(feature)
        if std == 0:
            return feature - mean
        return (feature - mean) / std


    def saveParams(self, path):
        means = np.mean(self.raw, axis=0)
        stds = np.std(self.raw, axis=0)
        with open(path, 'w') as csvfile:
            writer = csv.writer(csvfile, delimiter=',')
            for idx in range(len(self.raw[0])):
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


    def plotROC(self):
        labels = np.array(self.labels)
        features = np.array(self.features)

        cv = StratifiedKFold(self.labels, n_folds=6, shuffle=True)
        mean_tpr = [0.0] * self.n_classes
        mean_fpr = [np.linspace(0, 1, 100)] * self.n_classes

        for k, (train, test) in enumerate(cv):
            X_train, X_test, y_train, y_test =\
                features[train], features[test], \
                labels[train], labels[test]
            y_score = self.model.fit(X_train, y_train).decision_function(X_test)

            # Compute ROC curve and ROC area for each class
            fpr = dict()
            tpr = dict()
            roc_auc = dict()
            for i in range(self.n_classes):
                y_binary = [True if y == i else False for y in y_test]
                fpr[i], tpr[i], _ = roc_curve(y_binary, y_score[:, i])
                mean_tpr[i] += interp(mean_fpr[i], fpr[i], tpr[i])
                mean_tpr[i][0] = 0.0
                roc_auc[i] = auc(fpr[i], tpr[i])

        mean_tpr = np.array(mean_tpr) / len(cv)
        mean_tpr[:,-1] = 1.0
        mean_auc = [auc(f,t) for (f,t) in zip(mean_fpr, mean_tpr)]
        plt.figure(figsize=(8,6))
        for i in range(self.n_classes):
            plt.plot(mean_fpr[i], mean_tpr[i], label = \
                'Mean ROC curve of class ' + KeyStrokeManager.invEmotions[i] \
                + ' (Mean AUC = {1:0.2f})'.format(i, mean_auc[i]))
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating characteristic')
        plt.legend(loc="lower right")
        plt.show()


    def saveModel(self, path):
        self.model.fit(self.features, self.labels)
        joblib.dump(self.model, path, protocol=2) 
