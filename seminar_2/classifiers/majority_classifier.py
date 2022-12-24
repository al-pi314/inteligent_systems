from classifier import Classifier

class MajorityClassifier(Classifier):
    def __init__(self, dataset, target):
        super().__init__(dataset)
        
        self.options = self.dataset[target].value_counts()
        self.majority = self.options.index[0]

    def classify(self, _):
        return self.majority