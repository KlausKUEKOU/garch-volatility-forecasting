# Modèles GARCH de la volatilité du S&P 500 : estimation, prévision hors échantillon, VaR.
#
# Lancement : python run_analysis.py   (environ 1 minute)

import math
from pathlib import Path
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from arch import arch_model
from scipy import stats

from src.risk import christoffersen_independence, excess_kurtosis, kupiec_pof, mean, skewness, std

warnings.filterwarnings("ignore")
ROOT = Path(__file__).parent
plt.rcParams.update({"figure.dpi": 130, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.alpha": 0.3})

# ==================================================================== DONNÉES
px = pd.read_csv(ROOT / "data/sp500_daily.csv", parse_dates=["Date"], index_col="Date")

# rendement logarithmique en % : r(t) = 100 x ln(P(t) / P(t-1))
close = px["Adj Close"].values
dates = px.index[1:]
r_list = []
for t in range(1, len(close)):
    r_list.append(100 * math.log(close[t] / close[t - 1]))
r = pd.Series(r_list, index=dates)

# variance de Parkinson (1980) : utilise le plus haut et le plus bas du jour.
# Beaucoup moins bruitée que le rendement au carré pour mesurer la "vraie" variance du jour.
high, low = px["High"].values, px["Low"].values
rv_list = []
for t in range(1, len(close)):
    rv_list.append((100 * math.log(high[t] / low[t])) ** 2 / (4 * math.log(2)))
rv = pd.Series(rv_list, index=dates)

SPECS = {
    "GARCH-N": dict(vol="GARCH", p=1, q=1, dist="normal"),
    "GARCH-t": dict(vol="GARCH", p=1, q=1, dist="t"),
    "GJR-GARCH-t": dict(vol="GARCH", p=1, o=1, q=1, dist="t"),
    "EGARCH-t": dict(vol="EGARCH", p=1, o=1, q=1, dist="t"),
}


# ------------------------------------------------------------------------------
# ALGORITHME LjungBox(x, h)
# H0 : pas d'autocorrélation jusqu'au retard h.
#   rho(k) <- corrélation entre x(t) et x(t-k)
#   Q <- n (n + 2) x somme pour k = 1..h de rho(k)² / (n - k)
#   p-value <- P(Khi-deux(h) > Q)
# ------------------------------------------------------------------------------
def autocorrelation(x, k):
    m = mean(x)
    num, den = 0.0, 0.0
    for t in range(len(x)):
        den = den + (x[t] - m) ** 2
        if t >= k:
            num = num + (x[t] - m) * (x[t - k] - m)
    return num / den


def ljung_box_pvalue(x, h=10):
    n = len(x)
    q = 0.0
    for k in range(1, h + 1):
        q = q + autocorrelation(x, k) ** 2 / (n - k)
    q = n * (n + 2) * q
    return stats.chi2.sf(q, h)


# ==================================================================== 1. FAITS STYLISÉS
x = list(r.values)
x_squared = []
for v in x:
    x_squared.append(v * v)
desc = pd.Series({
    "Observations": len(x),
    "Mean (% per day)": mean(x),
    "Std (% per day)": std(x),
    "Annualised vol (%)": std(x) * math.sqrt(252),
    "Skewness": skewness(x),
    "Excess kurtosis": excess_kurtosis(x),
    "Jarque-Bera p-value": stats.jarque_bera(x).pvalue,
    "Min (%)": min(x),
    "Max (%)": max(x),
    "Ljung-Box(10) p, returns": ljung_box_pvalue(x),
    "Ljung-Box(10) p, squared returns": ljung_box_pvalue(x_squared),
})
desc.to_csv(ROOT / "results/descriptive_stats.csv", header=["value"], float_format="%.4f")

# ==================================================================== 2. ESTIMATION SUR TOUT L'ÉCHANTILLON
# Les modèles (variance du jour = h(t)) :
#   GARCH(1,1)  : h(t) = omega + alpha x e(t-1)² + beta x h(t-1)
#   GJR-GARCH   : h(t) = omega + (alpha + gamma x [e(t-1) < 0]) x e(t-1)² + beta x h(t-1)
#                 gamma > 0 : une mauvaise nouvelle augmente plus la volatilité (effet de levier)
#   EGARCH      : ln h(t) = omega + alpha x (|z| - E|z|) + gamma x z + beta x ln h(t-1)
# Les paramètres sont estimés par maximum de vraisemblance avec la librairie arch.
rows = []
fits = {}
for name in SPECS:
    spec = SPECS[name]
    res = arch_model(r, mean="Constant", **spec).fit(disp="off")
    fits[name] = res
    p = res.params
    alpha = p.get("alpha[1]", np.nan)
    beta = p.get("beta[1]", np.nan)
    gamma = p.get("gamma[1]", 0.0)
    if spec["vol"] == "EGARCH":
        persistence = beta
    else:
        persistence = alpha + beta + gamma / 2      # la moitié des chocs sont négatifs
    rows.append({
        "Model": name, "omega": p["omega"], "alpha": alpha, "gamma (leverage)": p.get("gamma[1]", np.nan),
        "beta": beta, "nu (t dof)": p.get("nu", np.nan), "Persistence": persistence,
        "Half-life (days)": math.log(0.5) / math.log(persistence),   # jours pour qu'un choc perde 50 %
        "Log-likelihood": res.loglikelihood, "AIC": res.aic, "BIC": res.bic,
    })
estimation = pd.DataFrame(rows).set_index("Model")
estimation.to_csv(ROOT / "results/estimation.csv", float_format="%.4f")

# diagnostic : les résidus standardisés du GJR ne doivent plus montrer d'effet ARCH
best = fits["GJR-GARCH-t"]
z = list(best.std_resid.dropna().values)
z_squared = []
for v in z:
    z_squared.append(v * v)
diag = pd.Series({"Ljung-Box(10) p, z": ljung_box_pvalue(z),
                  "Ljung-Box(10) p, z^2": ljung_box_pvalue(z_squared),
                  "Excess kurtosis z": excess_kurtosis(z)})
diag.to_csv(ROOT / "results/gjr_residual_diagnostics.csv", header=["value"], float_format="%.4f")

fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
axes[0].plot(r, lw=0.5, color="#555")
axes[0].set_ylabel("Daily log return (%)")
axes[0].set_title("S&P 500, 1999 to 2018: volatility clusters")
axes[1].plot(best.conditional_volatility * math.sqrt(252), color="#c0392b", lw=0.9,
             label="GJR-GARCH-t conditional volatility")
axes[1].plot(np.sqrt(rv.rolling(21).mean() * 252), color="#2471a3", lw=0.7, alpha=0.7,
             label="21-day Parkinson realised volatility")
axes[1].set_ylabel("Annualised volatility (%)")
axes[1].legend(fontsize=8)
fig.tight_layout()
fig.savefig(ROOT / "figures/returns_and_volatility.png")
plt.close(fig)

# courbes d'impact des nouvelles : variance de demain en fonction du choc d'aujourd'hui
shocks = np.linspace(-6, 6, 241)
uncond = r.var()                                     # variance moyenne, prise comme h(t-1)
fig, ax = plt.subplots(figsize=(6.5, 4.2))
for name, color in [("GARCH-t", "#7f7f7f"), ("GJR-GARCH-t", "#c0392b"), ("EGARCH-t", "#2471a3")]:
    p = fits[name].params
    curve = []
    for e in shocks:
        if name == "GARCH-t":
            h = p["omega"] + p["alpha[1]"] * e ** 2 + p["beta[1]"] * uncond
        elif name == "GJR-GARCH-t":
            bad_news = 1 if e < 0 else 0
            h = p["omega"] + (p["alpha[1]"] + p["gamma[1]"] * bad_news) * e ** 2 + p["beta[1]"] * uncond
        else:
            zz = e / math.sqrt(uncond)
            log_h = (p["omega"] + p["alpha[1]"] * (abs(zz) - math.sqrt(2 / math.pi)) + p["gamma[1]"] * zz
                     + p["beta[1]"] * math.log(uncond))
            h = math.exp(log_h)
        curve.append(h)
    ax.plot(shocks, curve, label=name, color=color, lw=1.8)
ax.set_xlabel("Today's return shock (%)")
ax.set_ylabel("Tomorrow's conditional variance")
ax.set_title("News impact curves: bad news raises volatility more")
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(ROOT / "figures/news_impact_curves.png")
plt.close(fig)

# ==================================================================== 3. PRÉVISIONS HORS ÉCHANTILLON
# ALGORITHME PrevisionGlissante
# POUR chaque jour t à partir de 2010 FAIRE
#     échantillon <- les 2 000 jours avant t
#     tous les 20 jours : ré-estimer les paramètres de chaque modèle
#     prévoir la variance de t et le quantile 1 % du rendement de t (VaR 99 %)
# Comparaison : modèle EWMA de RiskMetrics, h(t) = 0,94 x h(t-1) + 0,06 x r(t-1)²
WINDOW, REFIT, START = 2000, 20, "2010-01-01"
start_i = r.index.get_loc(r.loc[START:].index[0])
test_dates = r.index[start_i:]

forecast_var = {}
forecast_q = {}
params = {}
for name in SPECS:
    forecast_var[name] = []
    forecast_q[name] = []

for i in range(len(test_dates)):
    t = start_i + i
    sample = r.iloc[t - WINDOW:t]
    for name in SPECS:
        spec = SPECS[name]
        model = arch_model(sample, mean="Constant", **spec)
        if i % REFIT == 0:
            params[name] = model.fit(disp="off", show_warning=False).params
        f = model.fix(params[name]).forecast(horizon=1, reindex=False)
        mu_t = f.mean.iloc[-1, 0]
        var_t = f.variance.iloc[-1, 0]
        if spec["dist"] == "t":
            nu = params[name]["nu"]
            q = stats.t.ppf(0.01, nu) * math.sqrt((nu - 2) / nu)   # Student réduite à variance 1
        else:
            q = stats.norm.ppf(0.01)
        forecast_var[name].append(var_t)
        forecast_q[name].append(mu_t + math.sqrt(var_t) * q)

lam = 0.94
h = [r.iloc[:250].var()]
for t in range(1, len(r)):
    h.append(lam * h[t - 1] + (1 - lam) * r.iloc[t - 1] ** 2)
forecast_var["EWMA"] = h[start_i:]
forecast_q["EWMA"] = []
for v in h[start_i:]:
    forecast_q["EWMA"].append(stats.norm.ppf(0.01) * math.sqrt(v))

table = pd.DataFrame(forecast_var, index=test_dates)
table["realised_parkinson"] = rv.loc[test_dates].values
table["return_pct"] = r.loc[test_dates].values
table.to_csv(ROOT / "results/variance_forecasts.csv", float_format="%.6f")


# Perte QLIKE : proxy / h - ln(proxy / h) - 1  (vaut 0 si la prévision est parfaite)
def qlike(h_list, proxy_list):
    out = []
    for k in range(len(h_list)):
        ratio = proxy_list[k] / h_list[k]
        out.append(ratio - math.log(ratio) - 1)
    return out


# ------------------------------------------------------------------------------
# ALGORITHME DieboldMariano
# d(t) <- perte du modèle 1 - perte du modèle 2
# stat <- moyenne(d) / racine(variance de long terme de d / T)   (Newey-West, 5 retards)
# stat < 0 : le modèle 1 se trompe moins ; |stat| > 1,96 : différence significative à 5 %
# ------------------------------------------------------------------------------
def diebold_mariano(loss_1, loss_2, lag=5):
    d = []
    for k in range(len(loss_1)):
        d.append(loss_1[k] - loss_2[k])
    T = len(d)
    d_mean = mean(d)
    long_run_var = 0.0
    for k in range(T):
        long_run_var = long_run_var + (d[k] - d_mean) ** 2
    long_run_var = long_run_var / T
    for k in range(1, lag + 1):
        cov_k = np.cov(d[k:], d[:-k])[0, 1]
        long_run_var = long_run_var + 2 * (1 - k / (lag + 1)) * cov_k
    stat = d_mean / math.sqrt(long_run_var / T)
    p_value = 2 * (1 - stats.norm.cdf(abs(stat)))
    return stat, p_value


proxy = []
for v in rv.loc[test_dates].values:
    proxy.append(max(v, 1e-4))                     # évite une division par 0 les jours sans écart
test_returns = list(r.loc[test_dates].values)
qlike_reference = qlike(forecast_var["GARCH-N"], proxy)

eval_rows = []
for name in forecast_var:
    losses = qlike(forecast_var[name], proxy)
    squared_errors = []
    for k in range(len(proxy)):
        squared_errors.append((forecast_var[name][k] - proxy[k]) ** 2)
    if name == "GARCH-N":
        dm, p_dm = np.nan, np.nan
    else:
        dm, p_dm = diebold_mariano(losses, qlike_reference)
    violations = []
    for k in range(len(test_returns)):
        violations.append(test_returns[k] < forecast_q[name][k])
    rate, lr, p_kupiec = kupiec_pof(violations, 0.99)
    lr_ind, p_ind = christoffersen_independence(violations)
    eval_rows.append({"Model": name, "QLIKE": mean(losses), "MSE": mean(squared_errors),
                      "DM stat vs GARCH-N (QLIKE)": dm, "DM p-value": p_dm,
                      "VaR 99% violations": sum(violations), "Violation rate": rate,
                      "Kupiec p": p_kupiec, "Independence p": p_ind,
                      "Expected violations": round(len(test_dates) * 0.01, 1)})
evaluation = pd.DataFrame(eval_rows).set_index("Model")
evaluation.to_csv(ROOT / "results/out_of_sample_evaluation.csv", float_format="%.4f")

fig, ax = plt.subplots(figsize=(10, 4.2))
ax.plot(test_dates, np.sqrt(np.array(proxy) * 252), color="lightgrey", lw=0.6, label="Parkinson realised vol (daily)")
ax.plot(test_dates, np.sqrt(np.array(forecast_var["GJR-GARCH-t"]) * 252), color="#c0392b", lw=1,
        label="GJR-GARCH-t forecast")
ax.plot(test_dates, np.sqrt(np.array(forecast_var["EWMA"]) * 252), color="#2471a3", lw=0.8, alpha=0.8,
        label="EWMA forecast")
ax.set_ylabel("Annualised volatility (%)")
ax.set_title("One-day-ahead volatility forecasts, 2010 to 2018 (out of sample)")
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(ROOT / "figures/volatility_forecasts.png")
plt.close(fig)

fig, ax = plt.subplots(figsize=(10, 4.2))
ax.plot(test_dates, test_returns, color="lightgrey", lw=0.6, label="Daily return")
ax.plot(test_dates, forecast_q["GJR-GARCH-t"], color="#c0392b", lw=0.9, label="99% VaR, GJR-GARCH-t")
ax.plot(test_dates, forecast_q["GARCH-N"], color="#7f7f7f", lw=0.7, label="99% VaR, GARCH-N")
hit_dates, hit_values = [], []
for k in range(len(test_returns)):
    if test_returns[k] < forecast_q["GJR-GARCH-t"][k]:
        hit_dates.append(test_dates[k])
        hit_values.append(test_returns[k])
ax.scatter(hit_dates, hit_values, color="black", s=10, zorder=3, label="Violation (GJR-GARCH-t)")
ax.set_ylabel("%")
ax.set_title("99% one-day VaR backtest, 2010 to 2018")
ax.legend(fontsize=8, ncol=2)
fig.tight_layout()
fig.savefig(ROOT / "figures/var_backtest.png")
plt.close(fig)

print(desc.round(4))
print(estimation.round(4))
print(evaluation.round(4))
