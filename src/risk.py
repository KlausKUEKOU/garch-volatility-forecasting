# Value-at-Risk (VaR), Expected Shortfall (ES) et tests de validation de la VaR.
#
# Les pertes sont des nombres positifs : une VaR 99 % à 1 jour de 0,025 veut dire
# qu'on s'attend à perdre plus de 2,5 % un jour sur cent.

import math

from scipy import stats


def mean(x):
    total = 0.0
    for v in x:
        total = total + v
    return total / len(x)


def std(x):
    """Écart-type (diviseur n - 1)."""
    m = mean(x)
    total = 0.0
    for v in x:
        total = total + (v - m) ** 2
    return math.sqrt(total / (len(x) - 1))


def quantile(x, q):
    """Quantile empirique : on trie, puis on interpole entre les deux valeurs voisines."""
    values = sorted(x)
    position = (len(values) - 1) * q
    below = int(math.floor(position))
    above = min(below + 1, len(values) - 1)
    weight = position - below
    return values[below] + weight * (values[above] - values[below])


def skewness(x):
    m = mean(x)
    m2, m3 = 0.0, 0.0
    for v in x:
        m2 = m2 + (v - m) ** 2
        m3 = m3 + (v - m) ** 3
    m2, m3 = m2 / len(x), m3 / len(x)
    return m3 / m2 ** 1.5


def excess_kurtosis(x):
    m = mean(x)
    m2, m4 = 0.0, 0.0
    for v in x:
        m2 = m2 + (v - m) ** 2
        m4 = m4 + (v - m) ** 4
    m2, m4 = m2 / len(x), m4 / len(x)
    return m4 / m2 ** 2 - 3


# 1) VaR historique : la perte au quantile 1 % des rendements passés
def var_historical(r, alpha=0.99):
    return -quantile(r, 1 - alpha)


# 2) VaR gaussienne : moyenne + écart-type x quantile de la loi normale (z = -2,33 à 99 %)
def var_gaussian(r, alpha=0.99):
    z = stats.norm.ppf(1 - alpha)
    return -(mean(r) + std(r) * z)


# 3) VaR de Cornish-Fisher : on corrige z avec l'asymétrie (s) et l'aplatissement (k)
def var_cornish_fisher(r, alpha=0.99):
    z = stats.norm.ppf(1 - alpha)
    s = skewness(r)
    k = excess_kurtosis(r)
    z_cf = z + (z ** 2 - 1) * s / 6 + (z ** 3 - 3 * z) * k / 24 - (2 * z ** 3 - 5 * z) * s ** 2 / 36
    return -(mean(r) + std(r) * z_cf)


# 4) VaR Monte Carlo : on ajuste une loi de Student, on simule, on lit le quantile 1 %
def var_monte_carlo_t(r, alpha=0.99, n_sims=100_000, seed=0):
    df, loc, scale = stats.t.fit(r)
    simulated = stats.t.rvs(df, loc, scale, size=n_sims, random_state=seed)
    return -quantile(simulated, 1 - alpha)


# ------------------------------------------------------------------------------
# 5) VaR EWMA (RiskMetrics)
# ALGORITHME
#   poids du jour t (en partant du plus récent) : lambda^0, lambda^1, lambda^2, ...  (lambda = 0,94)
#   variance <- somme(poids x rendement²) / somme(poids)
#   VaR <- -z x racine(variance)
# Les jours récents comptent beaucoup plus : la VaR réagit en quelques jours à une crise.
# ------------------------------------------------------------------------------
def var_ewma(r, alpha=0.99, lam=0.94):
    n = len(r)
    num, den = 0.0, 0.0
    for t in range(n):
        weight = lam ** (n - 1 - t)
        num = num + weight * r[t] ** 2
        den = den + weight
    sigma = math.sqrt(num / den)
    return -sigma * stats.norm.ppf(1 - alpha)


# Expected Shortfall : la perte moyenne les jours où l'on dépasse la VaR
def expected_shortfall(r, alpha=0.99):
    cut = quantile(r, 1 - alpha)
    tail = []
    for v in r:
        if v <= cut:
            tail.append(v)
    return -mean(tail)


VAR_METHODS = {
    "Historical": var_historical,
    "Gaussian": var_gaussian,
    "Cornish-Fisher": var_cornish_fisher,
    "Monte Carlo (Student-t)": var_monte_carlo_t,
    "EWMA (RiskMetrics)": var_ewma,
}


def chi2_1_pvalue(lr):
    """p-value d'un Khi-deux à 1 degré de liberté : P(X > lr) = 1 - erf(racine(lr / 2))."""
    return 1 - math.erf(math.sqrt(lr / 2))


# ------------------------------------------------------------------------------
# ALGORITHME TestDeKupiec (1995)
# H0 : la fréquence des dépassements de VaR vaut bien 1 % (pour une VaR à 99 %).
#   x <- nombre de jours où la perte dépasse la VaR ; n <- nombre de jours
#   LR <- -2 x [ log-vraisemblance avec p = 1 %  -  log-vraisemblance avec p = x / n ]
#   p-value <- P(Khi-deux(1) > LR)
# ------------------------------------------------------------------------------
def kupiec_pof(violations, alpha=0.99):
    n = len(violations)
    x = 0
    for v in violations:
        if v:
            x = x + 1
    p = 1 - alpha
    p_hat = x / n
    if x == 0:
        lr = -2 * (n * math.log(1 - p))
    elif x == n:
        lr = -2 * (n * math.log(p))
    else:
        loglik_h0 = (n - x) * math.log(1 - p) + x * math.log(p)
        loglik_h1 = (n - x) * math.log(1 - p_hat) + x * math.log(p_hat)
        lr = -2 * (loglik_h0 - loglik_h1)
    return p_hat, lr, chi2_1_pvalue(lr)


# ------------------------------------------------------------------------------
# ALGORITHME TestDIndependance (Christoffersen, 1998)
# H0 : un dépassement aujourd'hui ne rend pas plus probable un dépassement demain.
#   compter les transitions entre deux jours consécutifs : 0->0, 0->1, 1->0, 1->1
#   pi0 <- P(dépassement | pas de dépassement la veille)
#   pi1 <- P(dépassement | dépassement la veille)
#   LR compare le modèle "pi0 = pi1" au modèle "pi0 différent de pi1"
# ------------------------------------------------------------------------------
def christoffersen_independence(violations):
    n00, n01, n10, n11 = 0, 0, 0, 0
    for t in range(1, len(violations)):
        before, today = violations[t - 1], violations[t]
        if not before and not today:
            n00 = n00 + 1
        elif not before and today:
            n01 = n01 + 1
        elif before and not today:
            n10 = n10 + 1
        else:
            n11 = n11 + 1
    pi0 = n01 / max(n00 + n01, 1)
    pi1 = n11 / max(n10 + n11, 1)
    pi = (n01 + n11) / max(n00 + n01 + n10 + n11, 1)

    def loglik(prob, n_no, n_yes):
        total = 0.0
        if n_no > 0 and prob < 1:
            total = total + n_no * math.log(1 - prob)
        if n_yes > 0 and prob > 0:
            total = total + n_yes * math.log(prob)
        return total

    lr = -2 * (loglik(pi, n00 + n10, n01 + n11) - loglik(pi0, n00, n01) - loglik(pi1, n10, n11))
    return lr, chi2_1_pvalue(lr)


def max_drawdown(wealth):
    """Pire baisse depuis un sommet : on suit le plus haut atteint et l'écart à ce plus haut."""
    peak = wealth[0]
    worst = 0.0
    for value in wealth:
        if value > peak:
            peak = value
        drop = value / peak - 1
        if drop < worst:
            worst = drop
    return worst
