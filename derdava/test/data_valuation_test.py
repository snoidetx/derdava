from math import isclose
from derdava.coalition_probability import IndependentCoalitionProbability
from derdava.data_valuation import ValuableModel
from derdava.model_utility import ICoalitionalValue


def is_close(dic1, dic2):
    for i in dic1:
        if not isclose(dic1[i], dic2[i], rel_tol=0.1):
            return False

    return True


i_coalitional_value = ICoalitionalValue({
    tuple(): 0,
    (0,): 3,
    (1,): 2,
    (2,): 1,
    (0, 1): 6,
    (0, 2): 5,
    (1, 2): 4,
    (0, 1, 2): 7
})

support = (0, 1, 2)
valuable_model = ValuableModel(support, i_coalitional_value)

# naïve prior
dummy_value = valuable_model.valuate()
assert is_close(dummy_value, {0: 0, 1: 0, 2: 0}), str(dummy_value)
loo_value = valuable_model.valuate(data_valuation_function='loo')
assert is_close(loo_value, {0: 3, 1: 2, 2: 1}), str(loo_value)
shapley_value = valuable_model.valuate(data_valuation_function='shapley')
assert is_close(shapley_value, {0: 10/3, 1: 7/3, 2: 4/3}), str(shapley_value)
banzhaf_value = valuable_model.valuate(data_valuation_function='banzhaf')
assert is_close(banzhaf_value, {0: 3.5, 1: 2.5, 2: 1.5}), str(banzhaf_value)
beta_1_1_value = valuable_model.valuate(data_valuation_function='beta', alpha=1, beta=1)
assert is_close(beta_1_1_value, shapley_value), str(beta_1_1_value)

# naïve derdava
coalition_probability_independent_always = IndependentCoalitionProbability({0: 1, 1: 1, 2: 1})
coalition_probability_independent_half = IndependentCoalitionProbability({0: 0.5, 1: 0.5, 2: 0.5})
coalition_probability_independent_never = IndependentCoalitionProbability({0: 0, 1: 0, 2: 0})
derdava_loo_value_independent_always = \
    valuable_model.valuate(data_valuation_function='robust loo',
                           coalition_probability=coalition_probability_independent_always)
assert is_close(derdava_loo_value_independent_always, loo_value), derdava_loo_value_independent_always
derdava_loo_value_independent_half = \
    valuable_model.valuate(data_valuation_function='robust loo',
                           coalition_probability=coalition_probability_independent_half)
assert is_close(derdava_loo_value_independent_half, {0: 1.75, 1: 1.25, 2: 0.75}), derdava_loo_value_independent_half
derdava_loo_value_independent_never = \
    valuable_model.valuate(data_valuation_function='robust loo',
                           coalition_probability=coalition_probability_independent_never)
assert is_close(derdava_loo_value_independent_never, dummy_value), derdava_loo_value_independent_never

# 012-MCMC derdava
derdava_shapley_value_independent_half = \
    valuable_model.valuate(data_valuation_function='robust shapley',
                           coalition_probability=coalition_probability_independent_half)
zot_mcmc_derdava_shapley_value_independent_half = \
    valuable_model.valuate(data_valuation_function='012-mcmc robust shapley',
                           coalition_probability=coalition_probability_independent_half,
                           tolerance=1.0005)
assert is_close(derdava_shapley_value_independent_half, zot_mcmc_derdava_shapley_value_independent_half), \
    (derdava_shapley_value_independent_half, zot_mcmc_derdava_shapley_value_independent_half)


