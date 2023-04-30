import numpy as np
from itertools import combinations
from math import comb
from urdava.data_valuation.semivalue_weight import get_weight
from urdava.model_utility import ModelUtilityFunction


class ValuableModel:
    def __init__(self, support: tuple, model_utility_function: ModelUtilityFunction):
        self.support = support
        self.model_utility_function = model_utility_function

    def loo(self):
        scores = {}
        coalition = list(self.support)
        post_utility = self.model_utility_function.get_utility(tuple(coalition))
        for i in self.support:
            coalition.remove(i)
            pre_utility = self.model_utility_function.get_utility(tuple(coalition))
            scores[i] = post_utility - pre_utility
            coalition.append(i)

        return scores

    def naive_prior(self, data_valuation_function="shapley", **kwargs):
        scores = {}
        for i in self.support:
            scores[i] = 0
            indices = list(self.support)
            indices.remove(i)
            for card in range(len(self.support)):
                coalitions = combinations(indices, card)
                for coalition in coalitions:
                    pre_utility = self.model_utility_function.get_utility(tuple(coalition))
                    coalition = coalition + (i,)
                    post_utility = self.model_utility_function.get_utility(tuple(coalition))
                    marginal_contribution = post_utility - pre_utility
                    scores[i] += marginal_contribution * \
                                 get_weight(len(self.support),
                                            len(coalition),
                                            data_valuation_function=data_valuation_function,
                                            **kwargs) / \
                                 comb(len(self.support) - 1, card)

        return scores

    def naive_urdava(self):
        data_source_values = {}
        for i in self.data_sources:
            all_coalitions = []
            data_source_values[i] = 0
            index_set = list(self.data_sources.keys())
            index_set.remove(i)
            for card in range(self.num_of_data_sources):
                coalitions = combinations(index_set, card)
                for coalition in coalitions:
                    all_coalitions.append(coalition)

            for S in all_coalitions:
                index_set = list(self.data_sources.keys())
                index_set.remove(i)
                for s in S:
                    index_set.remove(s)

                # marginal contribution
                pre_join_value = self.get_value(S)
                post_join_value = self.get_value(S + (i,))
                marginal_contribution = post_join_value - pre_join_value

                # weight
                weight = 0
                for card in range(len(index_set) + 1):
                    unincluded_coalitions = combinations(index_set, card)
                    for unincluded_coalition in unincluded_coalitions:
                        D_prime = tuple(sorted(S + (i,) + unincluded_coalition))
                        prob = prob_generator.get_prob(D_prime)
                        coeff = prob * \
                                get_weight(len(D_prime),
                                           len(S),
                                           data_valuation_function=prior_data_valuation_function,
                                           **kwargs) / \
                                comb(len(D_prime) - 1, len(S))
                        weight += coeff

                data_source_values[i] += weight * marginal_contribution

        return data_source_values

