from classifier import Classifier
from sklearn.naive_bayes import GaussianNB
import pandas as pd

class NaiveBayesClassifier(Classifier):
    def __init__(self, dataset, target, discrete=False, N_bins=10, N_unique_values=100, fillna_method=None, dropna=False, outliers_method=None, polinomial=None, *args, **kwargs):
        super().__init__(dataset, target, discrete, N_bins, N_unique_values, fillna_method, dropna, outliers_method, polinomial)
        
        self.classifier = GaussianNB(*args, **kwargs)
        self.classifier.fit(self.dataset, self.target) 

    def _predict(self, features):
        return self.classifier.predict(features)

    def _fit(self, features, targets):
        print("Fitting...")
        self.classifier.fit(features, targets)