import numpy as np
from abc import ABC, abstractmethod
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.metrics import f1_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC


class ModelUtilityFunction(ABC):
    """Base class used to represent model utility function :math:`v: \\mathcal{P}(D) \\to \\mathbb{R}`."""

    @abstractmethod
    def get_utility(self, coalition: tuple):
        """Returns the utility of a given coalition.

        :param coalition: A tuple containing the indices of data sources in the coalition.
        :return: Utility of the coalition.
        """
        pass


class ICoalitionalValue(ModelUtilityFunction):
    """Stores every coalition and its utility."""
    def __init__(self, dic: dict):
        """Constructs an ``ICoalitionalValue`` model utility function.

        :param dic: A dictionary containing mappings between coalitions and their utilities.
        """
        self.dic = dic

    def get_utility(self, coalition: tuple):
        """Returns the utility of a given coalition.

        :param coalition: A tuple containing the indices of data sources in the coalition.
        :return: Utility of the coalition.
        """
        coalition = tuple(sorted(coalition))
        if coalition not in self.dic:
            raise KeyError("Coalition does not exist in support set.")
        else:
            return self.dic[coalition]

        
class ISymmetricClassificationModel(ModelUtilityFunction):
    """Assumes all data sources are identical. """
    def __init__(self, model, data_sources: dict, X_test: np.ndarray, y_test: np.ndarray):
        """Constructs an `ISymmetricClassificationModel`.

        :param model: A machine learning model from this module used for training.
        :param data_sources: A dictionary containing mappings between data source indices and their data `(X, y)`.
        :param X_test: Features of the testing (or validating) set.
        :param y_test: Labels of the testing (or validating) set.
        """
        self.model = model
        self.data_sources = data_sources
        self.X_test = X_test
        self.y_test = y_test
        self.dic = {0:0}
        size = 1
        for i in self.data_sources:
            if size == 1:
                X_train = self.data_sources[i][0]
                y_train = self.data_sources[i][1]
            else:
                X_train = np.concatenate([X_train, self.data_sources[i][0]])
                y_train = np.concatenate([y_train, self.data_sources[i][1]])
                
            self.dic[size] = model().fit(X_train, y_train).score(self.X_test, self.y_test)
            size += 1 
        print(self.dic)
        
    def get_utility(self, coalition: tuple):
        """Returns the utility of a given coalition.

        :param coalition: A tuple containing the indices of data sources in the coalition.
        :return: Utility of the coalition.
        """
        if len(coalition) not in self.dic:
            raise KeyError("Coalition size does not exist in support set.")
        else:
            return self.dic[len(coalition)]


class ISymmetricExampleModel(ModelUtilityFunction):
    """:meta private:"""
    def __init__(self, data_sources: dict):
        self.dic = {0: 0}
        for i in range(1, len(data_sources) + 1):
            self.dic[i] = 0 + i / len(data_sources)
        print(self.dic)

    def get_utility(self, coalition):
        if len(coalition) not in self.dic:
            raise KeyError("Coalition size does not exist in support set.")
        else:
            return self.dic[len(coalition)]


class IClassificationModel(ModelUtilityFunction):
    """Represents a model utility function based on a classification model and accuracy scores."""
    def __init__(self, model, data_sources: dict, X_test: np.ndarray, y_test: np.ndarray):
        """Constructs an ``IClassificationModel``.

        :param model: A machine learning model from this module used for training.
        :param data_sources: A dictionary containing mappings between data source indices and their data `(X, y)`.
        :param X_test: Features of the testing (or validating) set.
        :param y_test: Labels of the testing (or validating) set.
        """
        self.model = model
        self.data_sources = data_sources
        self.X_test = X_test
        self.y_test = y_test

    def get_utility(self, coalition: tuple):
        """Returns the utility of a given coalition.

        :param coalition: A tuple containing the indices of data sources in the coalition.
        :return: Utility of the coalition.
        """
        coalition = tuple(sorted(coalition))
        if len(coalition) == 0:
            return 0
        
        model = self.model()

        for i in coalition:
            if i not in self.data_sources:
                raise KeyError(f"Data source {i} does not exist in support set.")

        X_train = np.concatenate([self.data_sources.get(i)[0] for i in coalition])
        y_train = np.concatenate([self.data_sources.get(i)[1] for i in coalition])
        
        if np.all(y_train == y_train[0]):
            return DummyClassifier(strategy='constant', constant=y_train[0]).fit(X_train, y_train).score(self.X_test, self.y_test)
        
        return model.fit(X_train, y_train).score(self.X_test, self.y_test)


model_knn = lambda: KNeighborsClassifier()
"""Represents a :math:`k`-Nearest Neighbours classifier."""

model_logistic_regression = lambda: LogisticRegression()
"""Represents a Logistic Regression classifier."""

model_linear_svm = lambda: make_pipeline(StandardScaler(), LinearSVC(tol=1e-5, max_iter=100000))
"""Represents a linear Support Vector Machine classifier."""

model_gaussian_nb = lambda: GaussianNB()
"""Represents a Gaussian Naïve Bayes classifier."""

model_ridge_classifier = lambda: RidgeClassifier()
"""Represents a Ridge classifier."""
