# Volatility Forecasting with GARCH Models

This project estimates four GARCH models on 20 years of daily S&P 500 returns (1999 to 2018) and
compares their forecasts of volatility and of Value-at-Risk on 2,260 days that the models have not
seen.

![Volatility](figures/returns_and_volatility.png)

## Objective

1. Which model forecasts daily volatility best?
2. Does it help to model the leverage effect, that is the fact that a fall in prices increases
   volatility more than a rise of the same size?

## Models

Let h(t) be the variance of day t and e(t) the return shock of day t.

| Model | Variance equation |
|---|---|
| GARCH(1,1) | h(t) = omega + alpha e(t-1)² + beta h(t-1) |
| GJR-GARCH(1,1) | h(t) = omega + (alpha + gamma if e(t-1) < 0) e(t-1)² + beta h(t-1) |
| EGARCH(1,1) | ln h(t) = omega + alpha (abs(z) - E abs(z)) + gamma z + beta ln h(t-1) |
| EWMA (RiskMetrics) | h(t) = 0.94 h(t-1) + 0.06 r(t-1)² |

GARCH is estimated with normal errors (GARCH-N) and Student errors (GARCH-t). GJR and EGARCH use
Student errors. Parameters are estimated by maximum likelihood with the `arch` library.

## Algorithm

```
FOR each day t from January 2010 to December 2018
    sample <- the 2,000 previous daily returns
    IF t is a multiple of 20 days THEN re-estimate the parameters of each model
    FOR each model
        h(t) <- forecast of the variance of day t
        VaR(t) <- mean + sqrt(h(t)) x 1% quantile of the model distribution
END FOR

proxy(t) <- Parkinson variance = (ln(High / Low))² / (4 ln 2)
QLIKE loss(t) <- proxy(t) / h(t) - ln(proxy(t) / h(t)) - 1
Diebold-Mariano test: is the average loss of a model different from that of GARCH-N?
Kupiec and Christoffersen tests on the VaR breaches
```

The Parkinson variance uses the daily high and low and is much less noisy than the squared return.
The QLIKE loss is the usual choice when the "true" variance is only measured with noise (Patton, 2011).

## Results

### Data

| Statistic | Value |
|---|---|
| Daily observations | 5,030 |
| Annualised volatility | 19.1% |
| Skewness | -0.20 |
| Excess kurtosis | 8.17 (Jarque-Bera p < 0.001) |
| Ljung-Box(10) on squared returns | p < 0.001 |

Returns have fat tails and their squares are autocorrelated: calm days follow calm days and
volatile days follow volatile days. This is what GARCH models describe.

### Estimation on the full sample

| Model | alpha | gamma | beta | Student dof | Persistence | Half-life | BIC |
|---|---|---|---|---|---|---|---|
| GARCH-N | 0.102 | | 0.885 | | 0.987 | 54 days | 13,917 |
| GARCH-t | 0.100 | | 0.900 | 6.5 | 0.9997 | very long | 13,712 |
| GJR-GARCH-t | 0.000 | 0.182 | 0.899 | 7.5 | 0.989 | 65 days | 13,548 |
| EGARCH-t | 0.129 | -0.154 | 0.982 | 7.3 | 0.982 | 39 days | 13,516 |

- Student errors improve the BIC by 205 points compared with normal errors. With 7 degrees of
  freedom, extreme days are much more frequent than under a normal distribution.
- In the GJR model alpha is 0 and gamma is 0.18: only negative shocks increase the next day's
  volatility. The negative gamma of the EGARCH model says the same thing. The two asymmetric models
  improve the BIC by more than 160 points.
- A volatility shock loses half of its effect after 40 to 65 trading days.
- After the GJR model, the squared standardised residuals are no longer autocorrelated
  (Ljung-Box p = 0.33).

![News impact](figures/news_impact_curves.png)

### Out-of-sample forecasts, 2010 to 2018

| Model | QLIKE | DM vs GARCH-N | DM p-value | 99% VaR breaches (22.6 expected) | Kupiec p | Independence p |
|---|---|---|---|---|---|---|
| GARCH-N | 0.555 | | | 50 | 0.000 | 0.125 |
| GARCH-t | 0.554 | -0.61 | 0.540 | 39 | 0.002 | 0.032 |
| GJR-GARCH-t | 0.520 | -4.88 | 0.000 | 35 | 0.016 | 0.119 |
| EGARCH-t | 0.530 | -2.83 | 0.005 | 35 | 0.016 | 0.119 |
| EWMA | 0.526 | -2.84 | 0.005 | 55 | 0.000 | 0.054 |

- GJR-GARCH-t has the lowest forecast error, and the difference with GARCH-N is significant
  (p < 0.001). The gain comes from the asymmetry: GARCH-t alone is not better than GARCH-N.
- Student tails reduce the number of VaR breaches from 50 to 35. The asymmetric models are the
  closest to 1%, but there are still a few too many breaches (Kupiec rejects at 5%, not at 1%).
  Most of them happen on sudden falls such as August 2011 or February 2018.
- EWMA forecasts volatility well, but its normal distribution gives too many VaR breaches (55).

![Forecasts](figures/volatility_forecasts.png)
![VaR](figures/var_backtest.png)

## Limits

The Parkinson variance ignores the overnight gap between the close and the next open, so it is
lower than the true daily variance. This changes the level of the MSE but not the QLIKE ranking.
Possible extensions: intraday data (HAR-RV model), skewed Student errors, multivariate DCC-GARCH.

## Run it

```bash
pip install -r requirements.txt
python run_analysis.py      # about 1 minute
```

```
data/sp500_daily.csv     S&P 500 daily prices, 1999 to 2018 (dataset shipped with the arch package)
run_analysis.py          statistics, estimation, rolling forecasts, VaR backtest, charts
src/risk.py              basic statistics, Kupiec and Christoffersen tests
results/                 tables (CSV)
figures/                 charts
```

Author: Klaus KUEKOU
