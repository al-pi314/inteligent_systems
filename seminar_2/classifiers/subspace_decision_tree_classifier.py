from classifier import Classifier
from sklearn.tree import DecisionTreeClassifier as SkDecisionTreeClassifier
import pandas as pd

class SubspaceDecisionTreeClassifier(Classifier):
    def __init__(self, dataset, target, discrete=False, N_bins=10, N_unique_values=100, fillna_method=None, dropna=False, outliers_method=None, polinomial=None, *args, **kwargs):
        super().__init__(dataset, target, discrete, N_bins, N_unique_values, fillna_method, dropna, outliers_method, polinomial)
        
        self.classifier = SkDecisionTreeClassifier(*args, **kwargs)
        self.classifier.fit(self.dataset, self.target) 

    def _predict(self, features):
        return self.classifier.predict(features[self.most_important])

    def _fit(self, features, targets):
        self.classifier.fit(features, targets)

        self.most_important = self.select_most_important(features, targets)

        self.classifier.fit(features[self.most_important], targets)

    def feature_importance(self, features, targets):
        joined = pd.concat([features, targets], axis=1)

        FI = pd.DataFrame(columns=["Feature_Importance", "Correlation", "Feature_Name"])
        FI["Feature_Importance"] = self.classifier.feature_importances_
        FI["Correlation"] = abs(joined.corr()["Class"].drop(["Class"]).values)
        FI["Feature_Name"] = features.columns

        return FI, features.corr()

    def select_most_important(self, features, targets, N_best=10):
        FI, correlation = self.feature_importance(features, targets)

        most_important = FI.sort_values(by="Feature_Importance", ascending=False).head(N_best)

        for feature_name in most_important["Feature_Name"].values:
            most_important = SubspaceDecisionTreeClassifier.add_non_linearly_dependent_features(most_important, correlation, FI, feature_name)

        return most_important["Feature_Name"].values

    @staticmethod
    def add_non_linearly_dependent_features(most_important, correlation_matrix, FI, feature_name, threshold=1.05, correlation_th=0.85):
        feature_data = most_important[most_important["Feature_Name"] == feature_name].iloc[0]
        coeficient = feature_data["Correlation"] / feature_data["Feature_Importance"] if  feature_data["Feature_Importance"] != 0 else float("inf")
        if coeficient < threshold:
            strongest_correlation = correlation_matrix[feature_name].sort_values(ascending=False).head(2).iloc[1]
            correlated_features = correlation_matrix[feature_name][correlation_matrix[feature_name] > correlation_th * strongest_correlation].index

            for correlated_feature in correlated_features:
                if correlated_feature == feature_name:
                    continue
                if correlated_feature not in most_important["Feature_Name"].values:
                    most_important = pd.concat([most_important, FI[FI["Feature_Name"] == correlated_feature]], ignore_index=True)
                    SubspaceDecisionTreeClassifier.add_non_linearly_dependent_features(most_important, correlation_matrix, FI, correlated_feature, threshold)
        return most_important