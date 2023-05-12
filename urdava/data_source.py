import numpy as np


def generate_random_data_sources(X: np.ndarray, y: np.ndarray, num_of_data_sources: int=10):
    """Splits a given dataset to a specified number of data sources."""

    data_sources = {}
    n = len(X) // num_of_data_sources
    for i in range(num_of_data_sources):
        data_sources[i] = (X[n * i:n * i + n, :].copy(), y[n * i:n * i + n].copy())

    return data_sources


def add_classification_noise(y: np.ndarray, noise_level: float=0.2):
    """Add noises to the classification labels by randomly choosing one from the remaining label set.

    :param y: Labels of target dataset
    :param noise_level: Amount of noise to be added, defaults to ``0.2``
    :return: ``None``
    :raises ValueError: If `noise_level` is not in the range ``[0, 1]``
    """
    if not 0 <= noise_level <= 1:
        raise ValueError("Noise level must be between 0 and 1.")

    labels = set(y.tolist())
    has_noises = np.random.binomial(1, noise_level, len(y))
    for i in range(len(y)):
        if has_noises[i]:
            label_candidates = list(labels)
            label_candidates.remove(y[i])
            noisy_label = label_candidates[np.random.randint(len(label_candidates))]
            y[i] = noisy_label

    return y
