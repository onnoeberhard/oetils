import jax
import numpy as np
from scipy.optimize import root_scalar
from tqdm.auto import tqdm


def bootstrap(a, rng, n=1000):
    """Returns `n` bootstrapped mean estimates of `a`"""
    idx = rng.integers(0, len(a), (n, len(a)))
    return a[idx].mean(1)


def bootstrap_cis(a, rng, p=0.05, n=1000):
    bs = np.stack([bootstrap(x, rng, n) for x in a])
    medians = np.median(bs, 1)
    lows = np.quantile(bs, p / 2, 1)
    highs = np.quantile(bs, 1 - p / 2, 1)
    errs = np.stack([medians - lows, highs - medians])
    return medians, errs, lows, highs


def smooth(x, n=500, add_head=False, add_last=False):
    """Filters and subsamples signal"""
    flat = x.ndim == 1
    if flat:
        x = x[None]
    N = x.shape[1]
    win = N // n
    filt = np.ones(win) / win
    ends = np.linspace(0, N, n + 1)[1:].astype(int)
    starts = np.r_[0, ends[:-1]]
    t = (starts + ends - 1) // 2
    y = np.apply_along_axis(lambda x: np.convolve(filt, x, 'valid'), 1, x)[:, starts]
    if add_head:
        t = np.r_[0, t]
        y = np.concatenate([x[:, 0, None], y], -1)
    if add_last:
        t = np.r_[t, x.shape[1] - 1]
        y = np.concatenate([y, x[:, -1, None]], -1)
    return t, y if not flat else y[0]


def nansmooth(x, n=500, add_head=False, add_last=False):
    """Filters and subsamples signal with missing data; n must divide signal length"""
    flat = x.ndim == 1
    if flat:
        x = x[None]
    N = x.shape[1]
    win = N // n
    ends = np.linspace(0, N, n + 1)[1:].astype(int)
    starts = np.r_[0, ends[:-1]]
    t = (starts + ends) // 2
    y = np.nanmean(x.reshape(len(x), win, -1, order='F'), 1)
    if add_head:
        t = np.r_[0, t]
        y = np.concatenate([x[:, 0, None], y], -1)
    if add_last:
        t = np.r_[t, x.shape[1] - 1]
        y = np.concatenate([y, x[:, -1, None]], -1)
    return t, y if not flat else y[0]


def log_smooth(x, n=500, add_last=False):
    """Filters and subsamples signal with logarithmic spacing"""
    flat = x.ndim == 1
    if flat:
        x = x[None]
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
    t = (starts + ends - 1) // 2
    y = np.zeros((len(x), n))
    for i, (s, e) in enumerate(zip(starts, ends)):
        y[:, i] = x[:, s:e].mean(1)
    if add_last:
        t = np.r_[t, N - 1]
        y = np.concatenate([y, x[:, -1, None]], -1)
    return t, y if not flat else y[0]


class JaxTqdm:
    def __init__(self, n, n_updates=100, **kwargs):
        self.n = n
        self.print_rate = max(1, n // n_updates)
        self.pbar = tqdm(total=self.n, **kwargs)

    def update(self):
        self.pbar.update(self.print_rate)

    def write(self, msg):
        self.pbar.write(msg)

    def loop(self, func):
        def f(i, val):
            jax.lax.cond((i + 1) % self.print_rate == 0,
                lambda: jax.debug.callback(self.update, ordered=True), lambda: None)
            return func(i, val)
        return f
