import numpy as np


def zot_sampling():
    """Returns the sampled state (one of :math:`\\{0, 1, 2\\}`) for each data source.

    :return: One of ``0``, ``1`` and ``2``.
    """
    return np.random.randint(0, 3)


def gelman_rubin(samples: dict, m_chains: int):
    """
    Computes the Gelman-Rubin statistic using the given samples.
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


def check_gelman_rubin(statistics: dict, tolerance: float):
    """Checks whether all Gelman-Rubin statistics have converged.

    :param statistics: A dictionary of Gelman-Rubin statistics for all data sources.
    :param tolerance: If one Gelman-Rubin statistic is between ``tolerance`` and ``1 / tolerance``.
    :return: A tuple containing two elements: (1) whether all Gelman-Rubin statistics have converged; (2) number of data sources that have not converged.
    """
    num_of_not_converged_data_sources = 0
    for i in statistics:
        if statistics[i] > tolerance or statistics[i] < 1 / tolerance:
            num_of_not_converged_data_sources += 1

    if num_of_not_converged_data_sources > 0:
        return False, num_of_not_converged_data_sources
    else:
        return True, 0

    
def cvar(values: dict, lower_tail: float=0.6, reverse: bool=False):
    """Returns the C-CVaR (Coalitional Conditional Value-at-Risk) value of a given discrete random variable (default C-CVaR\ :math:`^-`).

    :param values: A dictionary that contains mappings between values and probability of the random variable.
    :param lower_tail: The percentage of lower tail (:math:`\\alpha`) (default ``0.6``).
    :param reverse: Whether to compute C-CVaR\ :math:`^+` (upper tail) instead (default ``False``).
    :return: The computed C-CVaR value.
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
