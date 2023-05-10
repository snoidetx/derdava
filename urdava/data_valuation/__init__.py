import numpy as np
from itertools import combinations
from math import comb
from urdava.coalition_probability import CoalitionProbability
from urdava.data_valuation.semivalue_weight import get_weight
from urdava.model_utility import ModelUtilityFunction
from urdava.sampling import check_gelman_rubin, gelman_rubin, zot_sampling, cvar


class ValuableModel:
    def __init__(self, support: tuple, model_utility_function: ModelUtilityFunction):
        self.support = support
        self.model_utility_function = model_utility_function
        self.stored_utilities = {}

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
            return self.zot_mcmc_urdv(prior_data_valuation_function="loo",
                                      **kwargs)
        elif data_valuation_function == "012-mcmc robust shapley":
            return self.zot_mcmc_urdv(prior_data_valuation_function="shapley",
                                      **kwargs)
        elif data_valuation_function == "012-mcmc robust banzhaf":
            return self.zot_mcmc_urdv(prior_data_valuation_function="banzhaf",
                                      **kwargs)
        elif data_valuation_function == "012-mcmc robust beta":
            return self.zot_mcmc_urdv(prior_data_valuation_function="beta",
                                      **kwargs)
        # Risk-URDaVa
        # risk averse URDaVa
        elif data_valuation_function == "risk averse robust shapley":
            return self.naive_risk_averse_urdv(prior_data_valuation_function="shapley", **kwargs)
        elif data_valuation_function == "risk averse robust banzhaf":
            return self.naive_risk_averse_urdv(prior_data_valuation_function="banzhaf", **kwargs)
        elif data_valuation_function == "risk averse robust beta":
            return self.naive_risk_averse_urdv(prior_data_valuation_function="beta", **kwargs)
        # risk seeking URDaVa
        elif data_valuation_function == "risk seeking robust shapley":
            return self.naive_risk_seeking_urdv(prior_data_valuation_function="shapley", **kwargs)
        elif data_valuation_function == "risk seeking robust banzhaf":
            return self.naive_risk_seeking_urdv(prior_data_valuation_function="banzhaf", **kwargs)
        elif data_valuation_function == "risk seeking robust beta":
            return self.naive_risk_seeking_urdv(prior_data_valuation_function="beta", **kwargs)
        
        # partial prior
        elif data_valuation_function == "partial shapley":
            return self.partial_prior(data_valuation_function="shapley", **kwargs)
        elif data_valuation_function == "partial banzhaf":
            return self.partial_prior(data_valuation_function="banzhaf", **kwargs)
        elif data_valuation_function == "partial beta":
            return self.partial_prior(data_valuation_function="beta", **kwargs)

        else:
            raise ValueError("Data valuation function does not exist or arguments are invalid.")
            
    def get_utility(self, coalition: tuple):
        coalition = tuple(sorted(coalition))
        if coalition in self.stored_utilities:
            return self.stored_utilities[coalition]
        else:
            utility = self.model_utility_function.get_utility(coalition)
            self.stored_utilities[coalition] = utility
            return utility

    def loo(self):
        scores = {}
        coalition = list(self.support)
        post_utility = self.get_utility(tuple(coalition))
        for i in self.support:
            coalition.remove(i)
            pre_utility = self.get_utility(tuple(coalition))
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
                    pre_utility = self.get_utility(tuple(coalition))
                    coalition = coalition + (i,)
                    post_utility = self.get_utility(tuple(coalition))
                    marginal_contribution = post_utility - pre_utility
                    scores[i] += marginal_contribution * \
                                 get_weight(len(self.support),
                                            card,
                                            data_valuation_function=data_valuation_function,
                                            **kwargs) / \
                                 comb(len(self.support) - 1, card)

        return scores
    
    def partial_prior(self, data_valuation_function="shapley", **kwargs):
        scores = {}
        support = kwargs['partial_support']
        for i in self.support:          
            scores[i] = 0
            if i not in support:
                continue
            indices = list(support)
            indices.remove(i)
            for card in range(len(support)):
                coalitions = combinations(indices, card)
                for coalition in coalitions:
                    pre_utility = self.get_utility(tuple(coalition))
                    coalition = coalition + (i,)
                    post_utility = self.get_utility(tuple(coalition))
                    marginal_contribution = post_utility - pre_utility
                    scores[i] += marginal_contribution * \
                                 get_weight(len(support),
                                            card,
                                            data_valuation_function=data_valuation_function,
                                            **kwargs) / \
                                 comb(len(support) - 1, card)

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
            
            t = 0
            for S in all_coalitions:
                t += 1
                indices = list(self.support)
                indices.remove(i)
                for s in S:
                    indices.remove(s)

                # marginal contribution
                pre_utility = self.get_utility(tuple(S))
                post_utility = self.get_utility(tuple(S) + (i,))
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

    def zot_mcmc_urdv(self, prior_data_valuation_function="shapley", **kwargs):
        t = 0  # iterations
        statistics = {i: 0 for i in self.support}  # gelman-rubin statistic
        coalition_probability = kwargs['coalition_probability']
        tol = kwargs['tolerance']
        max_iter = 1000000
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
                    pre_utility = self.get_utility(tuple(S[i]))
                    S[i].append(i)
                    post_utility = self.get_utility(tuple(S[i]))
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


    def naive_risk_averse_urdv(self, prior_data_valuation_function="shapley", **kwargs):
        lower_tail = kwargs['lower_tail']
        coalition_probability = kwargs['coalition_probability']
        
        # generate the value and probability of each coalition
        coalition_probabilities = {}
        for card in range(len(self.support) + 1):
            coalitions = combinations(self.support, card)
            for coalition in coalitions:
                self.get_utility(coalition)
                coalition_probabilities[coalition] = coalition_probability.get_probability(coalition)

        # generate the CVaR value of each coalition
        coalition_cvar_values = {}
        for card in range(len(self.support) + 1):
            coalitions = combinations(self.support, card)
            for coalition in coalitions:
                values = {}
                for case in self.stored_utilities.keys():
                    actual_coalition = []
                    for i in coalition:
                        if i in case:
                            actual_coalition.append(i)

                    actual_value = self.get_utility(actual_coalition)
                    actual_prob = coalition_probabilities[case]
                    if actual_value not in values:
                        values[actual_value] = 0
                    values[actual_value] += actual_prob

                coalition_cvar_values[coalition] = cvar(values, lower_tail=lower_tail)

        scores = {}
        for i in self.support:
            scores[i] = 0
            index_set = list(self.support)
            index_set.remove(i)
            for card in range(len(self.support)):
                coalitions = combinations(index_set, card)
                for coalition in coalitions:
                    # calculate the risk averse expectation of the coalition
                    risk_averse_pre_join_value = coalition_cvar_values[coalition]
                    risk_averse_post_join_value = coalition_cvar_values[tuple(sorted(coalition + (i,)))]
                    marginal_contribution = risk_averse_post_join_value - risk_averse_pre_join_value
                    scores[i] += marginal_contribution * \
                                             get_weight(len(self.support),
                                                        len(coalition),
                                                        data_valuation_function=prior_data_valuation_function,
                                                        **kwargs) / \
                                             comb(len(self.support) - 1, card)

        return scores
    
    def naive_risk_seeking_urdv(self, prior_data_valuation_function="shapley", **kwargs):
        upper_tail = kwargs['upper_tail']
        coalition_probability = kwargs['coalition_probability']
        
        # generate the value and probability of each coalition
        coalition_probabilities = {}
        for card in range(len(self.support) + 1):
            coalitions = combinations(self.support, card)
            for coalition in coalitions:
                self.get_utility(coalition)
                coalition_probabilities[coalition] = coalition_probability.get_probability(coalition)

        # generate the CVaR value of each coalition
        coalition_cvar_values = {}
        for card in range(len(self.support) + 1):
            coalitions = combinations(self.support, card)
            for coalition in coalitions:
                values = {}
                for case in self.stored_utilities.keys():
                    actual_coalition = []
                    for i in coalition:
                        if i in case:
                            actual_coalition.append(i)

                    actual_value = self.get_utility(actual_coalition)
                    actual_prob = coalition_probabilities[case]
                    if actual_value not in values:
                        values[actual_value] = 0
                    values[actual_value] += actual_prob

                coalition_cvar_values[coalition] = cvar(values, lower_tail=1-upper_tail, reverse=True)

        scores = {}
        for i in self.support:
            scores[i] = 0
            index_set = list(self.support)
            index_set.remove(i)
            for card in range(len(self.support)):
                coalitions = combinations(index_set, card)
                for coalition in coalitions:
                    # calculate the risk averse expectation of the coalition
                    risk_averse_pre_join_value = coalition_cvar_values[coalition]
                    risk_averse_post_join_value = coalition_cvar_values[tuple(sorted(coalition + (i,)))]
                    marginal_contribution = risk_averse_post_join_value - risk_averse_pre_join_value
                    scores[i] += marginal_contribution * \
                                             get_weight(len(self.support),
                                                        len(coalition),
                                                        data_valuation_function=prior_data_valuation_function,
                                                        **kwargs) / \
                                             comb(len(self.support) - 1, card)

        return scores
