from classifier import Classifier
from random import choice


class RandomClassifier(Classifier):
    def __init__(self, dataset, target):
        super().__init__(dataset)
        
        self.options = self.dataset[target].unique()
    
    def classify(self, _):
        return choice(self.options)