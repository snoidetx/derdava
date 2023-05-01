import numpy as np
from itertools import combinations
from math import comb
from urdava.coalition_probability import CoalitionProbability
from urdava.data_valuation.semivalue_weight import get_weight
from urdava.model_utility import ModelUtilityFunction
from urdava.sampling import check_gelman_rubin, gelman_rubin, zot_sampling


class ValuableModel:
    def __init__(self, support: tuple, model_utility_function: ModelUtilityFunction):
        self.support = support
        self.model_utility_function = model_utility_function

    def valuate(self, data_valuation_function="dummy", **kwargs):
        kwargs["tolerance"] = kwargs.get("tolerance", 1.005)
        if data_valuation_function == "dummy":
            return {i: 0 for i in self.support}
        elif data_valuation_function == "loo":
            return self.loo()
        elif data_valuation_function == "shapley":
            return self.naive_prior(data_valuation_function="shapley")
        elif data_valuation_function == "banzhaf":
            return self.naive_prior(data_valuation_function="banzhaf")
        elif data_valuation_function == "beta":
            return self.naive_prior(data_valuation_function="beta",
                                    alpha=kwargs["alpha"], beta=kwargs["beta"])
        # naïve URDaVa
        elif data_valuation_function == "robust loo":
            return self.naive_urdava(kwargs["coalition_probability"], prior_data_valuation_function="loo")
        elif data_valuation_function == "robust shapley":
            return self.naive_urdava(kwargs["coalition_probability"], prior_data_valuation_function="shapley")
        elif data_valuation_function == "robust banzhaf":
            return self.naive_urdava(kwargs["coalition_probability"], prior_data_valuation_function="banzhaf")
        elif data_valuation_function == "robust beta":
            return self.naive_urdava(kwargs["coalition_probability"], prior_data_valuation_function="beta",
                                     alpha=kwargs["alpha"], beta=kwargs["beta"])
        # 012-MCMC URDaVa
        elif data_valuation_function == "012-mcmc robust loo":
            return self.zot_mcmc_urdv(kwargs["coalition_probability"], prior_data_valuation_function="loo",
                                      **kwargs)
        elif data_valuation_function == "012-mcmc robust shapley":
            return self.zot_mcmc_urdv(kwargs["coalition_probability"], prior_data_valuation_function="shapley",
                                      **kwargs)
        elif data_valuation_function == "012-mcmc robust banzhaf":
            return self.zot_mcmc_urdv(kwargs["coalition_probability"], prior_data_valuation_function="banzhaf",
                                      **kwargs)
        elif data_valuation_function == "012-mcmc robust beta":
            return self.zot_mcmc_urdv(kwargs["coalition_probability"], prior_data_valuation_function="beta",
                                      alpha=kwargs["alpha"], beta=kwargs["beta"],
                                      **kwargs)

        else:
            raise ValueError("Data valuation function does not exist or arguments are invalid.")

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
                                            card,
                                            data_valuation_function=data_valuation_function,
                                            **kwargs) / \
                                 comb(len(self.support) - 1, card)

        return scores

    def naive_urdava(self, coalition_probability: CoalitionProbability, prior_data_valuation_function="shapley", **kwargs):
        scores = {}
        for i in self.support:
            all_coalitions = []
            scores[i] = 0
            indices = list(self.support)
            indices.remove(i)
            for card in range(len(self.support)):
                coalitions = combinations(indices, card)
                for coalition in coalitions:
                    all_coalitions.append(coalition)

            for S in all_coalitions:
                indices = list(self.support)
                indices.remove(i)
                for s in S:
                    indices.remove(s)

                # marginal contribution
                pre_utility = self.model_utility_function.get_utility(tuple(S))
                post_utility = self.model_utility_function.get_utility(tuple(S) + (i,))
                marginal_contribution = post_utility - pre_utility

                # weight
                weight = 0
                for card in range(len(indices) + 1):
                    unincluded_coalitions = combinations(indices, card)
                    for unincluded_coalition in unincluded_coalitions:
                        D_prime = tuple(sorted(S + (i,) + unincluded_coalition))
                        prob = coalition_probability.get_probability(D_prime)
                        coeff = prob * \
                                get_weight(len(D_prime),
                                           len(S),
                                           data_valuation_function=prior_data_valuation_function,
                                           **kwargs) / \
                                comb(len(D_prime) - 1, len(S))
                        weight += coeff

                scores[i] += weight * marginal_contribution

        return scores

    def zot_mcmc_urdv(self, coalition_probability: CoalitionProbability,
                      prior_data_valuation_function="shapley", **kwargs):
        t = 0  # iterations
        statistics = {i: 0 for i in self.support}  # gelman-rubin statistic
        tol = kwargs['tolerance']
        max_iter = 4000
        m_chains = 10
        block_size = 50
        scores = {}
        samples = {}
        for i in self.support:
            scores[i] = 0
            samples[i] = []

        while (not check_gelman_rubin(statistics, tol)[0]) and t <= max_iter:
            if t % 50 == 0 and not kwargs.get('suppress_progress', False):
                print(f"====> Monte-Carlo Round {t} - Average convergence rate = "
                      f"{np.mean(np.array(list(statistics.values())))}")
                num_of_not_converged_data_sources = check_gelman_rubin(statistics, tol)[1]
                print(f"---------> Number of values that have not converged: "
                      f"{num_of_not_converged_data_sources}")

            for n_iter in range(block_size):  # number of new samples before checking convergence
                t += 1

                # do 012-sampling for each data source
                # remove d_i
                S = {i: [] for i in self.support}
                D_prime = {i: [] for i in self.support}
                for data_source in self.support:
                    outcome = zot_sampling()
                    for i in self.support:
                        if outcome == 0:
                            continue
                        elif outcome == 1:
                            if i == data_source:
                                continue
                            else:
                                D_prime[i].append(data_source)
                        else:
                            if i == data_source:
                                continue
                            else:
                                S[i].append(data_source)
                                D_prime[i].append(data_source)

                for i in self.support:
                    pre_utility = self.model_utility_function.get_utility(tuple(S[i]))
                    S[i].append(i)
                    post_utility = self.model_utility_function.get_utility(tuple(S[i]))
                    marginal_contribution = post_utility - pre_utility
                    D_prime[i].append(i)
                    term_prob = coalition_probability.get_probability(D_prime[i])
                    term_weight = get_weight(len(D_prime[i]),
                                             len(S[i]) - 1,
                                             data_valuation_function=prior_data_valuation_function,
                                             **kwargs)
                    term_comb = comb(len(D_prime[i]) - 1, len(S[i]) - 1)
                    scaled_marginal_contribution = (3 ** (len(self.support) - 1)) * \
                                                   term_prob * term_weight * marginal_contribution / term_comb
                    samples[i].append(scaled_marginal_contribution)
                    scores[i] = (scores[i] * (t - 1) + scaled_marginal_contribution) / t

            statistics = gelman_rubin(samples, m_chains)

        return scores

