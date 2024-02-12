import numpy as np
from scipy.optimize import root_scalar


def bootstrap(a, rng, n=1000):
    """Returns `n` bootstrapped mean estimates of `a`"""
    idx = rng.integers(0, len(a), (n, len(a)))
    return a[idx].mean(1)


def smooth(x, n=500, add_head=False, add_last=False):
    """Filters and subsamples signal"""
    N = x.shape[1]
    win = N // n
    filt = np.ones(win) / win
    ends = np.linspace(0, N, n + 1)[1:].astype(int)
    starts = np.r_[0, ends[:-1]]
    t = (starts + (ends) - 1) // 2
    y = np.apply_along_axis(lambda x: np.convolve(filt, x, 'valid'), 1, x)[:, starts]
    if add_head:
        t = np.r_[0, t]
        y = np.concatenate([x[:, 0, None], y], -1)
    if add_last:
        t = np.r_[t, x.shape[1] - 1]
        y = np.concatenate([y, x[:, -1, None]], -1)
    return t, y


def log_smooth(x, n=500, add_last=False):
    """Filters and subsamples signal with logarithmic spacing"""
    N = x.shape[1]

    # Find logarithm base
    beta_min = (N / n) ** (1 / (n - 1))
    beta_max = N ** (1 / (n - 1))

    def f(beta):
        return np.log(beta**n - 1) - np.log(beta - 1) - np.log(N)
    beta = root_scalar(f, bracket=[beta_min, beta_max]).root

    # Compute indices and smooth signal
    ends = np.ceil(np.cumsum(beta ** np.arange(n))).astype(int)
    starts = np.r_[0, ends[:-1]]
    t = (starts + (ends - 1)) // 2
    y = np.zeros((len(x), n))
    for i, (s, e) in enumerate(zip(starts, ends)):
        y[:, i] = x[:, s:e].mean(1)
    if add_last:
        t = np.r_[t, N - 1]
        y = np.concatenate([y, x[:, -1, None]], -1)
    return t, y
