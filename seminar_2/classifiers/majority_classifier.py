from classifier import Classifier

class MajorityClassifier(Classifier):
    def __init__(self, dataset, target):
        super().__init__(dataset, target)
        
        self.options = self.target.value_counts()
        self.majority = self.options.index[0]

    def _predict(self, samples):
        return [self.majority for _ in range(len(samples))]

    def _fit(self, _, targets):
        self.options = targets.value_counts()
        self.majority = self.options.index[0]