"""Value-at-Risk, Expected Shortfall and VaR backtests.

Losses are expressed as positive numbers: a 1-day 99% VaR of 0.025 means the
portfolio is expected to lose more than 2.5% on 1 day out of 100.
"""
import numpy as np
from scipy import stats


def var_historical(r, alpha=0.99):
    return -np.quantile(r, 1 - alpha)


def var_gaussian(r, alpha=0.99):
    return -(r.mean() + r.std(ddof=1) * stats.norm.ppf(1 - alpha))


def var_cornish_fisher(r, alpha=0.99):
    """Gaussian VaR corrected for skewness and excess kurtosis."""
    z = stats.norm.ppf(1 - alpha)
    s, k = stats.skew(r), stats.kurtosis(r)
    z_cf = (z + (z**2 - 1) * s / 6 + (z**3 - 3 * z) * k / 24 - (2 * z**3 - 5 * z) * s**2 / 36)
    return -(r.mean() + r.std(ddof=1) * z_cf)


def var_monte_carlo_t(r, alpha=0.99, n_sims=100_000, seed=0):
    """Student-t fitted to returns, VaR read from simulated draws."""
    df, loc, scale = stats.t.fit(r)
    sims = stats.t.rvs(df, loc, scale, size=n_sims, random_state=seed)
    return -np.quantile(sims, 1 - alpha)


def var_ewma(r, alpha=0.99, lam=0.94):
    """RiskMetrics: Gaussian VaR with an exponentially weighted volatility (lambda = 0.94),
    so the estimate reacts within days to a volatility spike."""
    w = lam ** np.arange(len(r))[::-1]
    sigma = np.sqrt(np.sum(w * r**2) / np.sum(w))
    return -sigma * stats.norm.ppf(1 - alpha)


def expected_shortfall(r, alpha=0.99):
    cut = np.quantile(r, 1 - alpha)
    return -r[r <= cut].mean()


VAR_METHODS = {
    "Historical": var_historical,
    "Gaussian": var_gaussian,
    "Cornish-Fisher": var_cornish_fisher,
    "Monte Carlo (Student-t)": var_monte_carlo_t,
    "EWMA (RiskMetrics)": var_ewma,
}


def kupiec_pof(violations, alpha=0.99):
    """Kupiec (1995) proportion-of-failures likelihood-ratio test.

    H0: the violation rate equals 1 - alpha. Returns (rate, LR statistic, p-value).
    """
    n, x = len(violations), int(np.sum(violations))
    p = 1 - alpha
    phat = x / n
    if x in (0, n):
        lr = -2 * (n * np.log(1 - p)) if x == 0 else -2 * (n * np.log(p))
    else:
        lr = -2 * ((n - x) * np.log(1 - p) + x * np.log(p)
                   - (n - x) * np.log(1 - phat) - x * np.log(phat))
    return phat, lr, 1 - stats.chi2.cdf(lr, df=1)


def christoffersen_independence(violations):
    """Christoffersen (1998) test that violations do not cluster in time."""
    v = np.asarray(violations, dtype=int)
    n00 = np.sum((v[:-1] == 0) & (v[1:] == 0))
    n01 = np.sum((v[:-1] == 0) & (v[1:] == 1))
    n10 = np.sum((v[:-1] == 1) & (v[1:] == 0))
    n11 = np.sum((v[:-1] == 1) & (v[1:] == 1))
    pi0 = n01 / max(n00 + n01, 1)
    pi1 = n11 / max(n10 + n11, 1)
    pi = (n01 + n11) / max(n00 + n01 + n10 + n11, 1)

    def ll(p, a, b):
        return (a * np.log(1 - p) if a and p < 1 else 0) + (b * np.log(p) if b and p > 0 else 0)

    lr = -2 * (ll(pi, n00 + n10, n01 + n11) - ll(pi0, n00, n01) - ll(pi1, n10, n11))
    return lr, 1 - stats.chi2.cdf(lr, df=1)


def max_drawdown(wealth):
    peak = np.maximum.accumulate(wealth)
    return (wealth / peak - 1).min()
