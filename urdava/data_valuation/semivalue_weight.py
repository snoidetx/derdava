from math import comb, gamma


def loo_weight(n, s):
    return 1 if s == n - 1 else 0


def shapley_weight(n, s):
    return 1 / n


def banzhaf_weight(n, s):
    return comb(n, s) / (2 ** n)


def beta_weight(n, s, alpha, beta):
    beta_fn = lambda a, b: gamma(a) * gamma(b) / gamma(a + b)
    return comb(n-1, s) * beta_fn(beta + s, alpha + n - 1 - s) / beta_fn(alpha, beta)


def get_weight(n: int, s: int, data_valuation_function: str="dummy", **kwargs):
    if data_valuation_function == "beta":
        alpha = kwargs["alpha"]
        beta = kwargs["beta"]
        return beta_weight(n, s, alpha, beta)
    elif data_valuation_function == "loo":
        return loo_weight(n, s)
    elif data_valuation_function == "shapley":
        return shapley_weight(n, s)
    elif data_valuation_function == "banzhaf":
        return banzhaf_weight(n, s)
    else:
        return 0
