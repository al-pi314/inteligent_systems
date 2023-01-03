from classifier import Classifier
from random import choices


class RandomClassifier(Classifier):
    def __init__(self, dataset, target):
        super().__init__(dataset, target)
        
        self.options = self.target.unique()
    
    def _predict(self, samples):
        return choices(self.options, k=len(samples))

    def _fit(self, features, targets):
        pass