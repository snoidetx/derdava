import numpy as np
from abc import ABC, abstractmethod
from itertools import combinations


class CoalitionProbability(ABC):
    """Represents the probability distribution that each coalition remains as the support set after data deletion."""
    @abstractmethod
    def get_probability(self, coalition: tuple):
        """Returns the staying probability of the given coalition.

        :param coalition: A tuple of integers representing the coalition to be queried.
        """

        pass

    @abstractmethod
    def simulate(self):
        """Randomly generates a coalition according to the joint probability."""
        pass


class IndependentCoalitionProbability(CoalitionProbability):
    """Each data source has independent staying probability."""

    def __init__(self, staying_probabilities):
        """Creates an ``IndependentCoalitionProbability``.

        :param staying_probabilities: A dictionary ``{ int: float }`` representing the independent
        staying probability of each data source.
        """

        self.staying_probabilities = staying_probabilities

    def get_probability(self, coalition: tuple):
        """Returns the staying probability of the given coalition.

        :param coalition: A tuple of integers representing the coalition to be queried.
        :return: Staying probability of the given coalition.
        """
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
    
    def simulate(self):
        """Randomly generates a coalition according to the joint probability.

        :return: The simulated coalition.
        """
        remaining_set = []
        for i in self.staying_probabilities:
            is_staying = np.random.random() < self.staying_probabilities[i]
            if is_staying:
                remaining_set.append(i)
        
        return tuple(remaining_set)


class RandomCoalitionProbability(CoalitionProbability):
    """Randomly creates the joint probability."""
    def __init__(self, support: tuple):
        """Creates an ``RandomCoalitionProbability``.

        :param support: A set containing the indices of data sources in the support set.
        """
        support = tuple(sorted(support))
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
        """Returns the staying probability of the given coalition.

        :param coalition: A tuple of integers representing the coalition to be queried.
        :return: Staying probability of the given coalition.
        """
        coalition = tuple(sorted(coalition))
        for i in coalition:
            if i not in self.support:
                raise KeyError(f"Data source {i} does not have a staying probability specified.")

        return self.prob_distribution[coalition]

    def simulate(self):
        """Randomly generates a coalition according to the joint probability.

        :return: The simulated coalition.
        """
        choice = np.random.choice(np.arange(len(self.prob_distribution)), p=list(self.prob_distribution.values()))
        return list(self.prob_distribution.keys())[choice]

