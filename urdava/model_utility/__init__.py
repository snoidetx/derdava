import numpy as np
from abc import ABC, abstractmethod
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.metrics import f1_score
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
        

class ISymmetricExampleModel(ModelUtilityFunction):
    def __init__(self, data_sources: dict):
        self.dic = {0:0}
        for i in range(1, len(data_sources) + 1):
            self.dic[i] = 0 + i / len(data_sources)
        print(self.dic)
            
    def get_utility(self, coalition):
        if len(coalition) not in self.dic:
            raise KeyError("Coalition size does not exist in support set.")
        else:
            return self.dic[len(coalition)]

        
class ISymmetricClassificationModel(ModelUtilityFunction):
    def __init__(self, model, data_sources: dict, X_test: np.ndarray, y_test: np.ndarray):
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
        if len(coalition) not in self.dic:
            raise KeyError("Coalition size does not exist in support set.")
        else:
            return self.dic[len(coalition)]
        
        
class ISymmetricClassificationF1Model(ModelUtilityFunction):
    def __init__(self, model, data_sources: dict, X_test: np.ndarray, y_test: np.ndarray):
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
                
            y_pred = model().fit(X_train, y_train).predict(self.X_test)
            self.dic[size] = f1_score(self.y_test, y_pred)
            size += 1 
        print(self.dic)
        
    def get_utility(self, coalition: tuple):
        if len(coalition) not in self.dic:
            raise KeyError("Coalition size does not exist in support set.")
        else:
            return self.dic[len(coalition)]


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
        
        model = self.model()

        for i in coalition:
            if i not in self.data_sources:
                raise KeyError(f"Data source {i} does not exist in support set.")

        X_train = np.concatenate([self.data_sources.get(i)[0] for i in coalition])
        y_train = np.concatenate([self.data_sources.get(i)[1] for i in coalition])
        return model.fit(X_train, y_train).score(self.X_test, self.y_test)
    

class IClassificationF1Model(ModelUtilityFunction):
    def __init__(self, model, data_sources: dict, X_test: np.ndarray, y_test: np.ndarray):
        self.model = model
        self.data_sources = data_sources
        self.X_test = X_test
        self.y_test = y_test

    def get_utility(self, coalition: tuple):
        coalition = tuple(sorted(coalition))
        if len(coalition) == 0:
            return 0
        
        model = self.model()

        for i in coalition:
            if i not in self.data_sources:
                raise KeyError(f"Data source {i} does not exist in support set.")

        X_train = np.concatenate([self.data_sources.get(i)[0] for i in coalition])
        y_train = np.concatenate([self.data_sources.get(i)[1] for i in coalition])
        y_pred = model.fit(X_train, y_train).predict(self.X_test)
        return f1_score(self.y_test, y_pred)


model_knn = lambda: KNeighborsClassifier()
model_logistic_regression = lambda: LogisticRegression()
model_linear_svm = lambda: make_pipeline(StandardScaler(), LinearSVC(tol=1e-5, max_iter=100000))
model_gaussian_nb = lambda: GaussianNB()
model_ridge_classifier = lambda: RidgeClassifier()
