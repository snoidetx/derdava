import numpy as np
from abc import ABC, abstractmethod
from itertools import combinations


class CoalitionProbability(ABC):
    @abstractmethod
    def get_probability(self, coalition: tuple):
        pass


class IndependentCoalitionProbability(CoalitionProbability):
    def __init__(self, staying_probabilities):
        self.staying_probabilities = staying_probabilities

    def get_probability(self, coalition: tuple):
        for i in coalition:
            if i not in self.staying_probabilities:
                raise KeyError(f"Data source {i} does not have a staying probability specified.")

        prob = 1
        for i in self.staying_probabilities:
            if i in coalition:
                prob *= self.staying_probabilities[i]
            else:
                prob *= 1 - self.staying_probabilities[i]

        return prob


class RandomCoalitionProbability(CoalitionProbability):
    def __init__(self, support: tuple):
        support = tuple(sorted(tuple))
        n_data_sources = len(support)
        p = np.random.random(2 ** n_data_sources)
        p /= p.sum()
        self.support = support
        self.prob_distribution = {}
        t = 0
        for card in range(n_data_sources + 1):
            coalitions = combinations(support, card)
            for coalition in coalitions:
                self.prob_distribution[coalition] = p[t]
                t += 1

    def get_probability(self, coalition: tuple):
        for i in coalition:
            if i not in self.support:
                raise KeyError(f"Data source {i} does not have a staying probability specified.")

        return self.prob_distribution[coalition]
