from abc import abstractmethod, ABC
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import roc_auc_score
import numpy as np
import pandas as pd

class Classifier(ABC):
    def __init__(self, dataset, target):
        self.dataset = dataset
        self.target = target

        self.fit(self.dataset, self.target)
    
    def evaluate(self, test_features, test_targets):
        predictions = np.array(self.predict(test_features))
        test_targets = np.array(test_targets.values)
        tp = np.sum((predictions == 1) & (test_targets == 1))
        fp = np.sum((predictions == 1) & (test_targets == 2))
        fn = np.sum((predictions == 2) & (test_targets == 1))
        tn = np.sum((predictions == 2) & (test_targets == 2))

        precision = tp / (tp + fp) if tp + fp != 0 else 0
        recall = tp / (tp + fn) if tp + fn != 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall != 0 else 0
        auc = roc_auc_score(test_targets, predictions)
        accuaracy = (tp + tn) / (tp + tn + fp + fn)
        return f1, precision, recall, auc, accuaracy

    @abstractmethod
    def _predict(self, features):
        pass

    @abstractmethod
    def _fit(self, features, targets):
        pass

    def fit(self, features, targets):
        self._fit(features, targets)

    def predict(self, features):
        return self._predict(features)

    def test(self, data, target, folds=5, repetitions=10):
        evaluations = 5
        scores = np.empty((repetitions, evaluations))
        for j in range(repetitions):
            folded_data = self.fold(data, k=folds)
            repetition_score_means = np.zeros(evaluations)

            for i in range(folds):
                train = pd.concat(folded_data[:i] + folded_data[i+1:])
                test = folded_data[i]

                self.fit(train.drop(target, axis=1), train[target])
                repetition_score_means += np.array(self.evaluate(test.drop(target, axis=1), test[target]))
            
            scores[j] = repetition_score_means / folds
        return scores
    
    @staticmethod
    def fold(data, k=5):
        data = data.sample(frac=1).reset_index(drop=True)
        folds = []
        for i in range(k):
            folds.append(data.iloc[i*len(data)//k:(i+1)*len(data)//k])
        return folds