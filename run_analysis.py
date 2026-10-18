"""GARCH-family volatility models for the S&P 500: estimation, out-of-sample forecasting, VaR.

Usage: python run_analysis.py   (about 5 minutes)
"""
from pathlib import Path
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from arch import arch_model
from scipy import stats

from src.var_tests import christoffersen_independence, kupiec_pof

warnings.filterwarnings("ignore")
ROOT = Path(__file__).parent
plt.rcParams.update({"figure.dpi": 130, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.alpha": 0.3})

px = pd.read_csv(ROOT / "data/sp500_daily.csv", parse_dates=["Date"], index_col="Date")
r = 100 * np.log(px["Adj Close"]).diff().dropna()              # % log returns
# Parkinson (1980) range-based variance: a far less noisy volatility proxy than r^2
rv = (100 * np.log(px["High"] / px["Low"])) ** 2 / (4 * np.log(2))
rv = rv.loc[r.index]

SPECS = {
    "GARCH-N": dict(vol="GARCH", p=1, q=1, dist="normal"),
    "GARCH-t": dict(vol="GARCH", p=1, q=1, dist="t"),
    "GJR-GARCH-t": dict(vol="GARCH", p=1, o=1, q=1, dist="t"),
    "EGARCH-t": dict(vol="EGARCH", p=1, o=1, q=1, dist="t"),
}

# ---------------------------------------------------------------- 1. descriptive statistics
desc = pd.Series({
    "Observations": len(r), "Mean (% per day)": r.mean(), "Std (% per day)": r.std(),
    "Annualised vol (%)": r.std() * np.sqrt(252), "Skewness": stats.skew(r),
    "Excess kurtosis": stats.kurtosis(r), "Jarque-Bera p-value": stats.jarque_bera(r).pvalue,
    "Min (%)": r.min(), "Max (%)": r.max(),
})
from statsmodels.stats.diagnostic import acorr_ljungbox
desc["Ljung-Box(10) p, returns"] = acorr_ljungbox(r, lags=[10])["lb_pvalue"].iloc[0]
desc["Ljung-Box(10) p, squared returns"] = acorr_ljungbox(r**2, lags=[10])["lb_pvalue"].iloc[0]
desc.to_csv(ROOT / "results/descriptive_stats.csv", header=["value"], float_format="%.4f")

# ---------------------------------------------------------------- 2. full-sample estimation
rows, fits = [], {}
for name, spec in SPECS.items():
    res = arch_model(r, mean="Constant", **spec).fit(disp="off")
    fits[name] = res
    p = res.params
    alpha, beta = p.get("alpha[1]", np.nan), p.get("beta[1]", np.nan)
    gamma = p.get("gamma[1]", 0.0)
    if spec["vol"] == "EGARCH":
        persistence = beta
    else:
        persistence = alpha + beta + gamma / 2
    rows.append({
        "Model": name, "omega": p["omega"], "alpha": alpha, "gamma (leverage)": p.get("gamma[1]", np.nan),
        "beta": beta, "nu (t dof)": p.get("nu", np.nan), "Persistence": persistence,
        "Half-life (days)": np.log(0.5) / np.log(persistence),
        "Log-likelihood": res.loglikelihood, "AIC": res.aic, "BIC": res.bic,
    })
est = pd.DataFrame(rows).set_index("Model")
est.to_csv(ROOT / "results/estimation.csv", float_format="%.4f")

# standardised-residual diagnostics for the preferred model
best = fits["GJR-GARCH-t"]
z = best.std_resid.dropna()
diag = pd.Series({
    "Ljung-Box(10) p, z": acorr_ljungbox(z, lags=[10])["lb_pvalue"].iloc[0],
    "Ljung-Box(10) p, z^2": acorr_ljungbox(z**2, lags=[10])["lb_pvalue"].iloc[0],
    "Excess kurtosis z": stats.kurtosis(z),
})
diag.to_csv(ROOT / "results/gjr_residual_diagnostics.csv", header=["value"], float_format="%.4f")

fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
axes[0].plot(r, lw=0.5, color="#555")
axes[0].set_ylabel("Daily log return (%)")
axes[0].set_title("S&P 500, 1999 to 2018: volatility clusters")
ann = best.conditional_volatility * np.sqrt(252)
axes[1].plot(ann, color="#c0392b", lw=0.9, label="GJR-GARCH-t conditional volatility")
axes[1].plot(np.sqrt(rv.rolling(21).mean() * 252), color="#2471a3", lw=0.7, alpha=0.7,
             label="21-day Parkinson realised volatility")
axes[1].set_ylabel("Annualised volatility (%)")
axes[1].legend(fontsize=8)
fig.tight_layout()
fig.savefig(ROOT / "figures/returns_and_volatility.png")
plt.close(fig)

# news impact curves: next-day variance as a function of today's shock
eps = np.linspace(-6, 6, 241)
fig, ax = plt.subplots(figsize=(6.5, 4.2))
for name, c in zip(["GARCH-t", "GJR-GARCH-t", "EGARCH-t"], ["#7f7f7f", "#c0392b", "#2471a3"]):
    p = fits[name].params
    uncond = r.var()
    if name == "GARCH-t":
        h = p["omega"] + p["alpha[1]"] * eps**2 + p["beta[1]"] * uncond
    elif name == "GJR-GARCH-t":
        h = p["omega"] + (p["alpha[1]"] + p["gamma[1]"] * (eps < 0)) * eps**2 + p["beta[1]"] * uncond
    else:
        s = np.sqrt(uncond)
        zz = eps / s
        logh = p["omega"] + p["alpha[1]"] * (np.abs(zz) - np.sqrt(2 / np.pi)) + p["gamma[1]"] * zz \
            + p["beta[1]"] * np.log(uncond)
        h = np.exp(logh)
    ax.plot(eps, h, label=name, color=c, lw=1.8)
ax.set_xlabel("Today's return shock (%)")
ax.set_ylabel("Tomorrow's conditional variance")
ax.set_title("News impact curves: bad news raises volatility more")
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(ROOT / "figures/news_impact_curves.png")
plt.close(fig)

# ---------------------------------------------------------------- 3. out-of-sample forecasts
WINDOW, REFIT, START = 2000, 20, "2010-01-01"
start_i = r.index.get_loc(r.loc[START:].index[0])
fc_var = {k: [] for k in SPECS}
fc_q = {k: [] for k in SPECS}          # 1% quantile of tomorrow's return
dates = r.index[start_i:]
params = {}
for i, t in enumerate(range(start_i, len(r))):
    sample = r.iloc[t - WINDOW:t]
    for name, spec in SPECS.items():
        mdl = arch_model(sample, mean="Constant", **spec)
        if i % REFIT == 0 or name not in params:
            params[name] = mdl.fit(disp="off", show_warning=False).params
        f = mdl.fix(params[name]).forecast(horizon=1, reindex=False)
        mu = f.mean.iloc[-1, 0]
        var = f.variance.iloc[-1, 0]
        if spec["dist"] == "t":
            nu = params[name]["nu"]
            q = stats.t.ppf(0.01, nu) * np.sqrt((nu - 2) / nu)
        else:
            q = stats.norm.ppf(0.01)
        fc_var[name].append(var)
        fc_q[name].append(mu + np.sqrt(var) * q)

# RiskMetrics EWMA benchmark
lam = 0.94
ew = np.empty(len(r))
ew[0] = r.iloc[:250].var()
for t in range(1, len(r)):
    ew[t] = lam * ew[t - 1] + (1 - lam) * r.iloc[t - 1] ** 2
fc_var["EWMA"] = list(ew[start_i:])
fc_q["EWMA"] = list(stats.norm.ppf(0.01) * np.sqrt(ew[start_i:]))

fc = pd.DataFrame(fc_var, index=dates)
fc.assign(realised_parkinson=rv.loc[dates], return_pct=r.loc[dates]).to_csv(
    ROOT / "results/variance_forecasts.csv", float_format="%.6f")


def qlike(h, proxy):
    return proxy / h - np.log(proxy / h) - 1


def diebold_mariano(l1, l2, lag=5):
    """DM test on loss differential d = l1 - l2 with Newey-West variance; negative = model 1 better."""
    d = np.asarray(l1) - np.asarray(l2)
    T = len(d)
    dbar = d.mean()
    gamma0 = np.var(d)
    lrv = gamma0 + 2 * sum((1 - k / (lag + 1)) * np.cov(d[k:], d[:-k])[0, 1] for k in range(1, lag + 1))
    stat = dbar / np.sqrt(lrv / T)
    return stat, 2 * (1 - stats.norm.cdf(abs(stat)))


proxy = rv.loc[dates].clip(lower=1e-4)
eval_rows = []
for name in fc:
    l_q = qlike(fc[name], proxy)
    l_mse = (fc[name] - proxy) ** 2
    dm, p_dm = diebold_mariano(l_q, qlike(fc["GARCH-N"], proxy)) if name != "GARCH-N" else (np.nan, np.nan)
    ret = r.loc[dates]
    viol = (ret.values < np.array(fc_q[name]))
    rate, _, p_k = kupiec_pof(viol, 0.99)
    _, p_i = christoffersen_independence(viol)
    eval_rows.append({"Model": name, "QLIKE": l_q.mean(), "MSE": l_mse.mean(),
                      "DM stat vs GARCH-N (QLIKE)": dm, "DM p-value": p_dm,
                      "VaR 99% violations": int(viol.sum()), "Violation rate": rate,
                      "Kupiec p": p_k, "Independence p": p_i})
ev = pd.DataFrame(eval_rows).set_index("Model")
ev["Expected violations"] = round(len(dates) * 0.01, 1)
ev.to_csv(ROOT / "results/out_of_sample_evaluation.csv", float_format="%.4f")

fig, ax = plt.subplots(figsize=(10, 4.2))
ax.plot(dates, np.sqrt(proxy * 252), color="lightgrey", lw=0.6, label="Parkinson realised vol (daily)")
ax.plot(dates, np.sqrt(fc["GJR-GARCH-t"] * 252), color="#c0392b", lw=1, label="GJR-GARCH-t forecast")
ax.plot(dates, np.sqrt(fc["EWMA"] * 252), color="#2471a3", lw=0.8, alpha=0.8, label="EWMA forecast")
ax.set_ylabel("Annualised volatility (%)")
ax.set_title("One-day-ahead volatility forecasts, 2010 to 2018 (out of sample)")
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(ROOT / "figures/volatility_forecasts.png")
plt.close(fig)

fig, ax = plt.subplots(figsize=(10, 4.2))
ret = r.loc[dates]
ax.plot(dates, ret, color="lightgrey", lw=0.6, label="Daily return")
ax.plot(dates, fc_q["GJR-GARCH-t"], color="#c0392b", lw=0.9, label="99% VaR, GJR-GARCH-t")
ax.plot(dates, fc_q["GARCH-N"], color="#7f7f7f", lw=0.7, label="99% VaR, GARCH-N")
v = ret.values < np.array(fc_q["GJR-GARCH-t"])
ax.scatter(dates[v], ret[v], color="black", s=10, zorder=3, label="Violation (GJR-GARCH-t)")
ax.set_ylabel("%")
ax.set_title("99% one-day VaR backtest, 2010 to 2018")
ax.legend(fontsize=8, ncol=2)
fig.tight_layout()
fig.savefig(ROOT / "figures/var_backtest.png")
plt.close(fig)

pd.set_option("display.width", 200)
print(desc.round(4)); print(est.round(4)); print(diag.round(4)); print(ev.round(4))
