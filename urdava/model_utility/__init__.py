import numpy as np
from abc import ABC, abstractmethod
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC


class ModelUtilityFunction(ABC):
    @abstractmethod
    def get_utility(self, coalition: tuple):
        pass


class ICoalitionalValue(ModelUtilityFunction):
    def __init__(self, dic):
        self.dic = dic

    def get_utility(self, coalition: tuple):
        coalition = tuple(sorted(coalition))
        if coalition not in self.dic:
            raise KeyError("Coalition does not exist in support set.")
        else:
            return self.dic[coalition]


class IClassificationModel(ModelUtilityFunction):
    def __init__(self, model, data_sources: dict, X_test: np.ndarray, y_test: np.ndarray):
        self.model = model
        self.data_sources = data_sources
        self.X_test = X_test
        self.y_test = y_test

    def get_utility(self, coalition: tuple):
        coalition = tuple(sorted(coalition))
        if len(coalition) == 0:
            return 0

        for i in coalition:
            if i not in self.data_sources:
                raise KeyError(f"Data source {i} does not exist in support set.")

        X_train = np.concatenate([self.data_sources.get(i)[0] for i in coalition])
        y_train = np.concatenate([self.data_sources.get(i)[1] for i in coalition])
        return self.model.fit(X_train, y_train).score(self.X_test, self.y_test)


model_knn = KNeighborsClassifier()
model_logistic_regression = LogisticRegression()
model_linear_svm = make_pipeline(StandardScaler(), LinearSVC())
model_gaussian_nb = GaussianNB()
