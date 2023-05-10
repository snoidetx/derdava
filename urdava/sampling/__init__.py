import numpy as np


def zot_sampling():
    return np.random.randint(0, 3)


def gelman_rubin(samples: dict, m_chains: int):
    """
    Compute the Gelman-Rubin statistic using the given samples.
    Referenced from https://www.imperial.ac.uk/media/imperial-college/research-centres-and-groups/astrophysics/public/icic/data-analysis-workshop/2018/Convergence-Tests.pdf.

    :param samples: A dictionary containing samples generated for each data source.
    :param m_chains: Number of Markov chains to be run in parallel.
    :return: A dictionary containing Gelman-Rubin statistics of each data source.
    """

    n_data_sources = len(samples)
    first_data_source = list(samples.keys())[0]

    if len(samples[first_data_source]) // m_chains < 10:
        return {i: 0 for i in samples.keys()}

    statistics = {}

    for i in samples:
        sample = np.array(samples[i], dtype="float")
        chain_len = len(sample) // m_chains
        chains = np.reshape(sample, (m_chains, chain_len))
        chain_means = np.zeros(m_chains, dtype="float")
        for j in range(m_chains):
            chain_means[j] = np.mean(chains[j])

        between_chain_variance = chain_len * np.var(chain_means) * m_chains / (m_chains - 1)
        within_chain_variances = np.zeros(m_chains, dtype="float")
        for j in range(m_chains):
            within_chain_variances[j] = np.var(chains[j]) * chain_len / (chain_len - 1)

        mean_within_chain_variance = np.mean(within_chain_variances)
        statistics[i] = (((chain_len - 1) * mean_within_chain_variance + between_chain_variance) /
                          (chain_len * mean_within_chain_variance))

    return statistics


def check_gelman_rubin(statistics, tolerance):
    num_of_not_converged_data_sources = 0
    for i in statistics:
        if statistics[i] > tolerance or statistics[i] < 1 / tolerance:
            num_of_not_converged_data_sources += 1

    if num_of_not_converged_data_sources > 0:
        return False, num_of_not_converged_data_sources
    else:
        return True, 0

    
def cvar(values, lower_tail=0.6, reverse=False):
    """
    Arguments:
    values - A dictionary that contains {values:probability_of_the_value};
    lower_tail - A probability between 0 and 1 (both inclusive) that represents the tail;
    reverse - If set to True, calculate the upper tail.
    """
    all_values = sorted(list(values.keys()))
    if reverse:
        all_values.reverse()
        lower_tail = 1 - lower_tail

    total = 0
    prob = 0
    for value in all_values:
        value_prob = values[value]
        if prob + value_prob > lower_tail:
            total += value * (lower_tail - prob)
            break
        else:
            total += value * value_prob
            prob += value_prob

    return total / lower_tail