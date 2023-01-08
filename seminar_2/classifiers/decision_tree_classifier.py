from classifier import Classifier
from sklearn.tree import DecisionTreeClassifier as SkDecisionTreeClassifier

class DecisionTreeClassifier(Classifier):
    def __init__(self, dataset, target, *args, **kwargs):
        self.classifier = SkDecisionTreeClassifier(*args, **kwargs)

        super().__init__(dataset, target)
        
    def _predict(self, features):
        return self.classifier.predict(features)

    def _fit(self, features, targets):
        self.classifier.fit(features, targets)
