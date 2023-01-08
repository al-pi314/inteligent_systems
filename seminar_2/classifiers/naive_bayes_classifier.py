from classifier import Classifier
from sklearn.naive_bayes import GaussianNB

class NaiveBayesClassifier(Classifier):
    def __init__(self, dataset, target, *args, **kwargs):
        self.classifier = GaussianNB(*args, **kwargs)
        super().__init__(dataset, target)

    def _predict(self, features):
        return self.classifier.predict(features)

    def _fit(self, features, targets):
        self.classifier.fit(features, targets)