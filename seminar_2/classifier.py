from abc import abstractmethod, ABC
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import roc_auc_score
import numpy as np
import pandas as pd

class Classifier(ABC):
    def __init__(self, dataset, target, discrete=False, N_bins=10, N_unique_values=100, fillna_method=None, dropna=False, outliers_method=None, polinomial=None):
        self.discrete = discrete
        self.fillna_method = fillna_method
        self.dropna = dropna
        self.outliers_method = outliers_method

        if discrete:
            self.discretize_init(N_bins, N_unique_values)

        self.poly_transformer = None
        if polinomial:
            self.poly_transformer = PolynomialFeatures(polinomial, include_bias=True)
        
        self.dataset, self.target = self.preprocess(dataset, target)

    @abstractmethod
    def _predict(self, features):
        pass

    @abstractmethod
    def _fit(self, features, targets):
        pass

    def fit(self, dataset, target):
        self.dataset, self.target = self.preprocess(dataset, target)

        self._fit(self.dataset, self.target)

    def predict(self, features):
        features, _ = self.preprocess(features, None)

        return self._predict(features)

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


    def preprocess(self, data, target):
        if self.discrete:
            data = self.discretize(data)

        if self.fillna_method:
            data = self.fillna(data)

        if self.dropna:
            indicies = data.isna().any(axis=1)
            data = data[~indicies]
            if target is not None:
                target = target[~indicies]

        if self.outliers_method:
            data = self.replace_outliers(data)

        if self.poly_transformer:
            data = self.poly_transformer.fit_transform(data)

        return data, target

    def discretize_init(self, N_bins, N_unique_values):
        continuous_columns = [c for c in self.dataset.columns if len(self.dataset[c].unique()) > N_unique_values]
        self.bins_for_columns = {}
        for c in continuous_columns:
            self.bins_for_columns[c] = np.linspace(self.dataset[c].min(), self.dataset[c].max(), N_bins + 1)

    def discretize(self, data):
        for c, bins_index in self.bins_for_columns.items():
            data[c] = pd.cut(data[c], bins=bins_index, labels=[i for i in range(len(bins_index) - 1)])
            data[c] = data[c].astype(float)
        return data

    def fillna(self, data):
        data = data.fillna(self.fillna_method(data))
        return data

    def replace_outliers(self, data):
        stats = data.describe()

        iqr = stats.loc["75%"] - stats.loc["25%"]
        lower_bound = stats.loc["25%"] - (1.5 * iqr)
        upper_bound = stats.loc["75%"] + (1.5 * iqr)

        outliers = (data < lower_bound) | (data > upper_bound)
        return data.where(~outliers, data.mean(), axis=1, inplace=False)