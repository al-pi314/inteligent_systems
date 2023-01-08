from classifier import Classifier
from sklearn.linear_model import LogisticRegression

class LogisticRegressionClassifier(Classifier):
    def __init__(self, dataset, target, *args, **kwargs):
        self.classifier = LogisticRegression(*args, **kwargs)

        super().__init__(dataset, target)
        
    def _predict(self, features):
        return self.classifier.predict(features)

    def _fit(self, features, targets):
        self.classifier.fit(features, targets)