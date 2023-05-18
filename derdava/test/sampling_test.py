import numpy as np
from math import isclose
from derdava.sampling import zot_sampling


n_samples = 1000
outcomes = np.zeros(n_samples)
for i in range(n_samples):
    outcomes[i] = zot_sampling()

frac_zero = len(outcomes[outcomes == 0]) / n_samples
frac_one = len(outcomes[outcomes == 1]) / n_samples
frac_two = len(outcomes[outcomes == 2]) / n_samples
assert isclose(frac_zero, 1 / 3, abs_tol=0.1), frac_zero
assert isclose(frac_one, 1 / 3, abs_tol=0.1), frac_one
assert isclose(frac_two, 1 / 3, abs_tol=0.1), frac_two

