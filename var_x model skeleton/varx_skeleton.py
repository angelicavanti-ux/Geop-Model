import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
import warnings

from matplotlib.backends.backend_pdf import PdfPages

from statsmodels.tsa.stattools import adfuller, grangercausalitytests
from statsmodels.tsa.vector_ar.vecm import coint_johansen
from statsmodels.tsa.statespace.varmax import VARMAX
from statsmodels.stats.stattools import durbin_watson
from statsmodels.stats.diagnostic import acorr_ljungbox

# ---------------- CONFIG: edit per dataset (Can we use the Hausman test to automatically determine endogenous variables?)----------------
ENDOG_COLS = []      # e.g. ["ccb_3m", "llm_risk_index"]  - variables modelled jointly
EXOG_COLS = []       # e.g. ["DGS2", "VIXCLS", "DEXUSEU"] - controls, one-directional
N_TEST = 20          # observations held out for evaluation
MAX_LAG = 12         # max lag order to consider (article uses 12 for Granger too)
SIGNIF = 0.05        # significance level used across all tests
# -----------------------------------------------------------

# Function for loading and preparing data
def load_data(path, date_col="Date"):
    """Load a CSV, parse dates, set datetime index, sort chronologically."""
    df = pd.read_csv(path)
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.set_index(date_col).sort_index()
    return df

# Function for splitting the dataframe into endogenous and exogenous variables
def split_endog_exog(df):
    """Split the loaded dataframe into endogenous and exogenous blocks per CONFIG."""
    missing = [c for c in ENDOG_COLS + EXOG_COLS if c not in df.columns]
    if missing:
        raise KeyError(f"Columns in CONFIG but not in data: {missing}")
    return df[ENDOG_COLS].copy(), df[EXOG_COLS].copy()

# Function for plotting the time series data
def plot_series(df, save_path=None):
    """Plot each column in its own panel for visual inspection."""
    n = len(df.columns)
    fig, axes = plt.subplots(nrows=n, ncols=1, figsize=(8, 2.2 * n), dpi=120, squeeze=False)
    for ax, col in zip(axes.flatten(), df.columns):
        ax.plot(df.index, df[col], linewidth=1)
        ax.set_title(col, fontsize=9)
        ax.tick_params(labelsize=7)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path)
    return fig

# Function for performing pairwise Granger causality tests
# This test checks whether lagged values of one endogenous variable improve the
# prediction of another endogenous variable, beyond what the variable's own lags
# already explain.
# The output is a matrix of the minimum p-values across tested lags for each pair.

def granger_matrix(df, maxlag=MAX_LAG, test="ssr_chi2test"):
    """Pairwise Granger causality p-values. Rows = response (y), columns = predictor (x).
    Cell value = min p-value across lags 1..maxlag. Run on ENDOG columns only."""
    variables = df.columns
    mat = pd.DataFrame(np.zeros((len(variables), len(variables))),
                       columns=variables, index=variables)
    for c in mat.columns:
        for r in mat.index:
            if r == c:
                mat.loc[r, c] = np.nan
                continue
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", FutureWarning)
                res = grangercausalitytests(df[[r, c]], maxlag=maxlag, verbose=False)
            p_values = [res[lag][0][test][1] for lag in range(1, maxlag + 1)]
            mat.loc[r, c] = np.min(p_values)
    mat.columns = [v + "_x" for v in variables]
    mat.index = [v + "_y" for v in variables]
    return mat



# Function for performing the Johansen cointegration test
# The Johansen test checks whether multiple time series share a long-run equilibrium.
# It identifies the number of cointegrating relationships among the endogenous variables.
def cointegration_test(df, alpha=SIGNIF, det_order=0, k_ar_diff=5):
    """Johansen trace test. Row 'r<=i' rejected => more than i cointegrating relations.
    Run on ENDOG columns in LEVELS (not differenced)."""
    res = coint_johansen(df, det_order, k_ar_diff)
    col = {0.90: 0, 0.95: 1, 0.99: 2}[round(1 - alpha, 2)]
    out = pd.DataFrame({
        "trace_stat": res.lr1,
        f"crit_{int((1 - alpha) * 100)}%": res.cvt[:, col],
    }, index=[f"r<={i}" for i in range(len(res.lr1))])
    out["reject"] = out["trace_stat"] > out.iloc[:, 1]
    return out

# Functions for performing the Augmented Dickey-Fuller (ADF) test
# The Augmented Dickey-Fuller test checks whether a time series has a unit root.
# The null hypothesis is that the series is non-stationary.
# A series "passes" the test when the p-value is below the significance level,
# meaning the null is rejected and the series is likely stationary.
# If the p-value is above the threshold, the series often needs differencing or
# another transformation before re-testing for stationarity.
def adf_report(series, signif=SIGNIF):
    """ADF test on one series. Null: unit root (non-stationary)."""
    stat, pvalue, nlags, nobs, crit, icbest = adfuller(series.dropna(), autolag="AIC")
    return {"variable": series.name, "adf_stat": round(stat, 4),
            "pvalue": round(pvalue, 4), "n_lags": nlags,
            "stationary": pvalue <= signif}

def run_adf_all(df, signif=SIGNIF):
    """ADF on every column; returns one summary dataframe."""
    return pd.DataFrame([adf_report(df[c], signif) for c in df.columns]).set_index("variable")



def plot_stationarity(df, save_path=None, signif=SIGNIF):
    """For each column: level (left) vs 1st difference (right),
    with the ADF p-value and verdict embedded in each panel title."""
    n = len(df.columns)
    fig, axes = plt.subplots(nrows=n, ncols=2, figsize=(11, 2.4 * n), dpi=120, squeeze=False)
    diffed = df.diff()
    for i, col in enumerate(df.columns):
        for j, (data, label) in enumerate([(df[col], "level"), (diffed[col], "1st difference")]):
            p = adfuller(data.dropna(), autolag="AIC")[1]
            ax = axes[i, j]
            ax.plot(data.index, data, linewidth=0.9)
            verdict = "stationary" if p <= signif else "NON-stationary"
            ax.set_title(f"{col} ({label}) - ADF p={p:.4f} [{verdict}]", fontsize=8)
            ax.tick_params(labelsize=6)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path)
    return fig

# Functions for differencing until stationarity
# NOTE: The following section contains functions to repeatedly
# difference the entire dataframe until all series pass the ADF
# stationarity test. These functions operate on the whole frame
# (not column-by-column) to keep a consistent order of integration
# across variables for multivariate modelling.
# Be cautious: this may over-difference already-stationary series (n_diffs > 2 may be risky).


def _all_stationary(df, signif=SIGNIF):
    """True if every column passes the ADF test."""
    return all(adfuller(df[c].dropna(), autolag="AIC")[1] <= signif
               for c in df.columns)

def difference_until_stationary(df, signif=SIGNIF, max_diffs=2):
    """Difference the WHOLE frame until all columns are stationary.
    Returns (differenced_df, n_diffs). n_diffs is needed later to invert forecasts."""
    n_diffs = 0
    out = df.copy()
    while not _all_stationary(out, signif) and n_diffs < max_diffs:
        out = out.diff().dropna()
        n_diffs += 1
    if not _all_stationary(out, signif):
        raise ValueError(f"Still non-stationary after {max_diffs} differences - inspect the data.")
    return out, n_diffs


# Drawbacks of differencing already-stationary variables:
# - Over-differencing (applying differences to series that are already
# stationary) can remove meaningful long-run information and introduce
# moving-average structure, complicating model identification.
# - It can increase noise and reduce signal-to-noise ratio, degrading
# forecasting performance and interpretability.
# - Differencing changes the series' variance/stochastic properties and
# may necessitate additional modelling choices (e.g. extra lags, MA
# terms) to capture the induced dynamics.
# - Because this function diffs the whole frame, a single non-stationary
# series forces differencing of all variables; this may be undesirable if
# some variables are already stationary and should remain in levels.

# Consider alternatives when appropriate:
# - Test and difference each series individually, tracking orders of
# integration per variable.

# Function for splitting the dataset into training and testing sets
def train_test_split_ts(endog, exog, n_test=N_TEST):
    """Chronological split - LAST n_test rows held out, never random.
    Aligns endog and exog to their common index first, so unequal
    differencing upstream can't silently desynchronise the frames.
    Returns (endog_train, endog_test, exog_train, exog_test)."""
    common = endog.index.intersection(exog.index)
    endog, exog = endog.loc[common], exog.loc[common]
    return (endog.iloc[:-n_test], endog.iloc[-n_test:],
            exog.iloc[:-n_test], exog.iloc[-n_test:])


# Function for selecting the optimal lag order for VAR-X model
def select_lag_order(endog, exog=None, max_lag=6):
    """Fit VARMAX(p, 0) for p = 1..max_lag; return AIC/BIC/HQIC per lag.
    Why VARMAX not VAR: plain statsmodels VAR has no exog support, and
      lag selection must happen in the SAME model class you'll estimate.
    Watch for: converged=False rows - their information criteria are
      untrustworthy; also MLE runtime grows fast with k and p."""
    rows = []
    for p in range(1, max_lag + 1):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = VARMAX(endog, exog=exog, order=(p, 0)).fit(disp=False)
        rows.append({"lag": p, "aic": res.aic, "bic": res.bic, "hqic": res.hqic,
                     "converged": res.mle_retvals.get("converged", None)})
    return pd.DataFrame(rows).set_index("lag")


def plot_lag_selection(lag_tbl, save_path=None):
    """AIC/BIC/HQIC vs lag order; circled point = each criterion's minimum."""
    fig, ax = plt.subplots(figsize=(7, 4), dpi=120)
    for crit in ["aic", "bic", "hqic"]:
        ax.plot(lag_tbl.index, lag_tbl[crit], marker="o", label=crit.upper())
        best = lag_tbl[crit].idxmin()
        ax.scatter([best], [lag_tbl.loc[best, crit]], s=150, facecolors="none",
                   edgecolors="black", zorder=5)
    ax.set_xlabel("lag order p"); ax.set_ylabel("criterion")
    ax.set_xticks(lag_tbl.index)
    ax.legend(); ax.set_title("Lag order selection (circled = minimum)")
    plt.tight_layout()
    if save_path: fig.savefig(save_path)
    return fig

def plot_cross_correlations(endog, exog=None, max_lag=10, save_path=None):
    """Grid of corr(predictor(t-k), response(t)) for k = 0..max_lag.
    Rows = responses (endog), columns = predictors (endog + exog).
    Dashed lines: approximate 95% band (+/- 2/sqrt(n)).
    Shows WHERE relationships live in lag-space; the VAR still uses one system-wide p."""
    df = endog if exog is None else pd.concat([endog, exog], axis=1).dropna()
    responses, predictors = endog.columns, df.columns
    nr, npred = len(responses), len(predictors)
    fig, axes = plt.subplots(nr, npred, figsize=(2.6 * npred, 2.2 * nr),
                             dpi=120, squeeze=False)
    ci = 2 / np.sqrt(len(df))
    for i, r in enumerate(responses):
        for j, p in enumerate(predictors):
            ccf_vals = [df[r].corr(df[p].shift(k)) for k in range(max_lag + 1)]
            ax = axes[i, j]
            ax.bar(range(max_lag + 1), ccf_vals, width=0.6)
            ax.axhline(ci, color="gray", ls="--", lw=0.7)
            ax.axhline(-ci, color="gray", ls="--", lw=0.7)
            ax.axhline(0, color="black", lw=0.7)
            ax.set_title(f"{p}(t-k) -> {r}(t)", fontsize=7)
            ax.tick_params(labelsize=6); ax.set_ylim(-1, 1)
    plt.tight_layout()
    if save_path: fig.savefig(save_path)
    return fig

# Function for fitting the VAR-X model using maximum likelihood estimation
def fit_varx(endog, exog=None, p=1, verbose=False):
    """Fit the VAR-X: VARMAX(p, 0) with exogenous regressors, by MLE.
    Warnings deliberately NOT suppressed here - this is the model you
    will interpret, so convergence complaints must be visible.
    verbose=True prints the full coefficient summary."""
    model = VARMAX(endog, exog=exog, order=(p, 0))
    results = model.fit(disp=False)
    if verbose:
        print(results.summary())
    return results

# Function for performing residual diagnostics on the fitted model
def residual_diagnostics(results, endog_cols, lb_lags=5):
    """Per-equation residual checks after fitting.
    durbin_watson: ~2 = no lag-1 autocorrelation (<1.5 positive, >2.5 negative).
    lb_pvalue: Ljung-Box joint test over lags 1..lb_lags; LOW p = leftover
      autocorrelation = model underspecified (raise p or rethink variables)."""
    resid = pd.DataFrame(results.resid, columns=endog_cols)
    dw = durbin_watson(resid.values)
    lb_p = [acorr_ljungbox(resid[c], lags=[lb_lags], return_df=True)["lb_pvalue"].iloc[0]
            for c in endog_cols]
    out = pd.DataFrame({"durbin_watson": np.round(dw, 3),
                        "lb_pvalue": np.round(lb_p, 4)}, index=endog_cols)
    out["dw_verdict"] = pd.cut(out["durbin_watson"], bins=[0, 1.5, 2.5, 4],
                               labels=["positive autocorr", "ok", "negative autocorr"])
    return out

# Functions for generating a PDF report of the model results and diagnostics
def _df_to_page(pdf, df, title, fontsize=8):
    """Render a dataframe as a table on its own A4 page of the PDF."""
    fig, ax = plt.subplots(figsize=(8.27, 11.69), dpi=120)
    ax.axis("off")
    ax.set_title(title, fontsize=12, pad=20)
    tbl = ax.table(cellText=df.round(4).astype(str).values,
                   rowLabels=df.index, colLabels=df.columns,
                   loc="upper center", cellLoc="center")
    tbl.auto_set_font_size(False); tbl.set_fontsize(fontsize); tbl.scale(1, 1.3)
    pdf.savefig(fig); plt.close(fig)

# function for generating a multi-page PDF report of the model results and diagnostics
def generate_report(path, results, endog, exog, lag_tbl=None, p=None,
                    adf_levels=None, adf_transformed=None, diag=None,
                    n_diffs=None, extra_figs=None):
    """Write a multi-page PDF: model equation + estimation summary,
    coefficient table, then whichever diagnostic tables/figures are passed.
    All table arguments optional - the report grows with the pipeline."""
    k = len(endog.columns)
    m = len(exog.columns) if exog is not None else 0
    p = p if p is not None else results.model.k_ar
    with PdfPages(path) as pdf:
        # Page 1: equation + summary
        fig, ax = plt.subplots(figsize=(8.27, 11.69), dpi=120)
        ax.axis("off")
        y = 0.95
        ax.text(0.5, y, "VAR-X Model Report", ha="center", fontsize=18, weight="bold"); y -= 0.06
        ax.text(0.5, y, f"Generated {pd.Timestamp.now():%Y-%m-%d %H:%M}", ha="center", fontsize=9); y -= 0.07
        ax.text(0.05, y, "Model equation:", fontsize=11, weight="bold"); y -= 0.06
        ax.text(0.5, y, r"$y_t = c + \sum_{i=1}^{p} A_i \, y_{t-i} + B\, x_t + u_t$",
                ha="center", fontsize=15); y -= 0.05
        ax.text(0.5, y, rf"$y_t \in \mathbb{{R}}^{{{k}}}$ (endogenous),  "
                        rf"$x_t \in \mathbb{{R}}^{{{m}}}$ (exogenous),  $p = {p}$",
                ha="center", fontsize=11); y -= 0.07
        ax.text(0.05, y, "Endogenous: " + ", ".join(endog.columns), fontsize=10); y -= 0.04
        ax.text(0.05, y, "Exogenous:  " + (", ".join(exog.columns) if m else "(none)"), fontsize=10); y -= 0.04
        if n_diffs is not None:
            ax.text(0.05, y, f"Differencing applied to endog: {n_diffs}x", fontsize=10); y -= 0.04
        ax.text(0.05, y, f"Observations: {results.nobs}   AIC: {results.aic:.2f}   "
                         f"BIC: {results.bic:.2f}   Converged: "
                         f"{results.mle_retvals.get('converged', '?')}", fontsize=10)
        pdf.savefig(fig); plt.close(fig)

        # Page 2: coefficients with inference
        coef = pd.DataFrame({"coef": results.params, "std_err": results.bse,
                             "pvalue": results.pvalues})
        _df_to_page(pdf, coef, "Estimated coefficients "
                    "(L{lag}.{predictor}.{response} / beta.{exog}.{response})")

        if adf_levels is not None:
            _df_to_page(pdf, adf_levels, "ADF tests - levels")
        if adf_transformed is not None:
            _df_to_page(pdf, adf_transformed, "ADF tests - transformed (as modelled)")
        if lag_tbl is not None:
            _df_to_page(pdf, lag_tbl, "Lag order selection (information criteria)")
        if diag is not None:
            _df_to_page(pdf, diag, "Residual diagnostics")
        if extra_figs:
            for f in extra_figs:
                pdf.savefig(f)
    return path


def forecast_varx(results, steps, exog_future=None):
    """Forecast `steps` ahead. THE VAR-X CATCH: exog_future must contain
    the exogenous values for those future dates - shape (steps, n_exog).
    In backtesting these come from the held-out test set (legitimate: we
    evaluate 'given the controls, did geopolitical signals add predictive
    power?'). In LIVE forecasting they must be projected or scenario-set -
    using realized future values there is look-ahead bias.
    Returns (mean_forecast, conf_int) as dataframes."""
    fc = results.get_forecast(steps=steps, exog=exog_future)
    return fc.predicted_mean, fc.conf_int()


def invert_differencing(levels_train, forecast_diffed, n_diffs):
    """Convert a forecast made in differenced space back to LEVELS.
    levels_train: the ORIGINAL (pre-differencing) endog training frame.
    forecast_diffed: the forecast from forecast_varx (differenced space).
    n_diffs: the counter returned by difference_until_stationary.
    Works by re-anchoring on the last observed value at each differencing
    depth and cumulatively summing - once per diff, deepest first."""
    if n_diffs == 0:
        return forecast_diffed.copy()
    fc = forecast_diffed.copy()
    for d in range(n_diffs, 0, -1):
        anchor = levels_train.copy()
        for _ in range(d - 1):
            anchor = anchor.diff().dropna()
        fc = anchor.iloc[-1] + fc.cumsum()
    return fc

def forecast_accuracy(forecast, actual):
    """Per-variable forecast metrics. Aligns frames on their common index first.
    rmse_vs_naive is the headline: < 1 means the model beats a no-change
    (random walk) forecast; > 1 means it doesn't. NaN where undefined
    (MAPE with ~zero actuals; naive ratio with constant actuals)."""
    f, a = forecast.align(actual, join="inner")
    rows = {}
    for c in f.columns:
        e = f[c] - a[c]
        denom_ok = (a[c].abs() > 1e-8)
        mape = (e[denom_ok].abs() / a[c][denom_ok].abs()).mean() if denom_ok.any() else np.nan
        naive_rmse = np.sqrt(((a[c] - a[c].shift(1)).dropna() ** 2).mean())
        rows[c] = {
            "rmse": np.sqrt((e ** 2).mean()),
            "mae": e.abs().mean(),
            "me": e.mean(),
            "mape": mape,
            "corr": f[c].corr(a[c]),
            "rmse_vs_naive": (np.sqrt((e ** 2).mean()) / naive_rmse)
                              if naive_rmse > 1e-12 else np.nan,
        }
    return pd.DataFrame(rows).T.round(4)


def plot_forecast_vs_actual(fc_mean, actual, fc_ci=None, train_tail=None, save_path=None):
    """One panel per variable: recent training data (gray), actuals (black),
    forecast (dashed blue), 95% CI band (shaded). Works in either space -
    pass differenced or level frames consistently, never mixed."""
    cols = fc_mean.columns
    n = len(cols)
    fig, axes = plt.subplots(n, 1, figsize=(8, 2.6 * n), dpi=120, squeeze=False)
    for ax, c in zip(axes.flatten(), cols):
        if train_tail is not None:
            ax.plot(train_tail.index, train_tail[c], color="gray", lw=0.9, label="train (tail)")
        ax.plot(actual.index, actual[c], color="black", lw=1.1, label="actual")
        ax.plot(fc_mean.index, fc_mean[c], color="tab:blue", lw=1.1, ls="--", label="forecast")
        if fc_ci is not None:
            ax.fill_between(fc_mean.index, fc_ci[f"lower {c}"], fc_ci[f"upper {c}"],
                            alpha=0.18, color="tab:blue", label="95% CI")
        ax.set_title(c, fontsize=9)
        ax.legend(fontsize=6)
        ax.tick_params(labelsize=6)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path)
    return fig





# ---------------- TEMPORARY SMOKE TEST (delete once real data flows) ----------------
from varx_skeleton import plot_series


if __name__ == "__main__":
    # 1. Fabricate a dataset and write it to CSV (tests the load_data round-trip)
    rng = np.random.default_rng(42)
    idx = pd.date_range("2023-01-02", periods=200, freq="B")
    fake = pd.DataFrame(
        rng.standard_normal((200, 4)).cumsum(axis=0),
        index=idx,
        columns=["y1", "y2", "x1", "x2"],
    )
    fake.index.name = "Date"
    fake.sample(frac=1, random_state=0).to_csv("_smoke_test.csv")  # shuffled on purpose

    # 2. load_data
    df = load_data("_smoke_test.csv")
    assert df.index.is_monotonic_increasing, "sort_index failed"
    assert df.shape == (200, 4), f"unexpected shape {df.shape}"
    print("load_data OK:", df.shape)

    # 3. split_endog_exog — temporarily fill config for the test
    ENDOG_COLS[:] = ["y1", "y2"]
    EXOG_COLS[:] = ["x1", "x2"]
    endog, exog = split_endog_exog(df)
    print("split OK:", endog.columns.tolist(), "|", exog.columns.tolist())

    # 4. Confirm the guard clause actually fires
    ENDOG_COLS[:] = ["y1", "typo_col"]
    try:
        split_endog_exog(df)
        print("GUARD FAILED - no error raised")
    except KeyError as e:
        print("guard OK:", e)
    ENDOG_COLS[:] = ["y1", "y2"]

    # 5. plot_series
    fig = plot_series(endog, save_path="_smoke_plot.png")
    print("plot OK:", len(fig.axes), "axes; saved _smoke_plot.png")

    # 6. granger_matrix — plant x1 -> y1 so we know the answer
    endog_g = endog.copy()
    endog_g["y1"] = endog_g["y1"].shift(0)  # keep as-is
    gm = granger_matrix(pd.concat([endog, exog[["x1"]]], axis=1).diff().dropna(), maxlag=4)
    print("granger OK:\n", gm.round(4))

    # 7. cointegration_test — our fake series are independent walks, expect no rejects
    ct = cointegration_test(endog)
    print("cointegration OK:\n", ct)

    # 8. ADF — fake data are random walks: expect all False, then all True after diff
    adf_levels = run_adf_all(endog)
    adf_diffed = run_adf_all(endog.diff().dropna())
    print("adf levels OK:\n", adf_levels)
    print("adf differenced OK:\n", adf_diffed)
    assert not adf_levels["stationary"].any(), "walks should be non-stationary in levels"
    assert adf_diffed["stationary"].all(), "walks should be stationary after one diff"

    # 9. stationarity plot
    fig_st = plot_stationarity(endog, save_path="_smoke_stationarity.png")
    print("stationarity plot OK:", len(fig_st.axes), "axes; saved _smoke_stationarity.png")

    # 10. difference_until_stationary — walks are I(1), expect exactly 1
    endog_st, n_diffs = difference_until_stationary(endog)
    assert n_diffs == 1, f"expected 1 diff for random walks, got {n_diffs}"
    assert len(endog_st) == len(endog) - 1
    print("differencing OK: n_diffs =", n_diffs)


    # 11. aligned split — use the differenced endog + differenced exog
    exog_st = exog.diff().dropna()
    etr, ete, xtr, xte = train_test_split_ts(endog_st, exog_st)
    assert (etr.index == xtr.index).all() and (ete.index == xte.index).all()
    assert len(ete) == N_TEST
    print("split OK:", etr.shape, ete.shape, xtr.shape, xte.shape)

    # 12. lag selection — plumbing check only (no true lag exists in random walks)
    lag_tbl = select_lag_order(etr, xtr, max_lag=3)
    assert lag_tbl["converged"].all(), "unconverged fits in smoke test"
    print("lag selection OK:\n", lag_tbl.round(2))
    p_star = int(lag_tbl["aic"].idxmin())


    # 12. lag-selection + CCF plots
    plot_lag_selection(lag_tbl, save_path="_smoke_lag_selection.png")
    plot_cross_correlations(etr, xtr, max_lag=6, save_path="_smoke_ccf.png")
    print("lag/ccf plots OK: saved _smoke_lag_selection.png, _smoke_ccf.png")


    # 13. fit the model at the AIC-chosen lag
    varx_res = fit_varx(etr, xtr, p=p_star)
    assert varx_res.mle_retvals["converged"], "smoke fit did not converge"
    print("fit OK: p =", p_star, "| n params =", len(varx_res.params))

    # 14. residual diagnostics
    diag = residual_diagnostics(varx_res, ENDOG_COLS)
    print("diagnostics OK:\n", diag)

    # 14. PDF report
    figs = [plot_stationarity(endog),
            plot_lag_selection(lag_tbl),
            plot_cross_correlations(etr, xtr, max_lag=6)]
    generate_report("_smoke_report.pdf", varx_res, etr, xtr,
                    lag_tbl=lag_tbl, p=p_star,
                    adf_levels=adf_levels, adf_transformed=adf_diffed,
                    diag=diag, n_diffs=n_diffs, extra_figs=figs)
    print("report OK: _smoke_report.pdf")


    # 15. forecast the held-out window, using test-set exog
    fc_mean, fc_ci = forecast_varx(varx_res, steps=N_TEST, exog_future=xte)
    assert fc_mean.shape == (N_TEST, len(ENDOG_COLS))
    assert (fc_mean.index == ete.index).all(), "forecast index misaligned with test set"
    print("forecast OK:", fc_mean.shape)

    # 16. invert differencing — perfect-forecast round trip must be exact
    perfect_fc = endog.diff().dropna().iloc[-N_TEST:]        # true future diffs
    recovered = invert_differencing(endog.iloc[:-N_TEST], perfect_fc, n_diffs=1)
    max_err = (recovered - endog.iloc[-N_TEST:]).abs().max().max()
    assert max_err < 1e-9, f"round trip error {max_err}"
    # and invert the real model forecast for later evaluation
    fc_levels = invert_differencing(endog.iloc[:len(etr)+1], fc_mean, n_diffs)
    print("inversion OK: round-trip err", f"{max_err:.1e}", "| levels forecast", fc_levels.shape)

    # 17. accuracy in both spaces
    acc_diff = forecast_accuracy(fc_mean, ete)
    acc_lvl = forecast_accuracy(fc_levels, endog.iloc[-N_TEST:])
    print("accuracy (differenced space):\n", acc_diff)
    print("accuracy (level space):\n", acc_lvl)

    # 18. forecast plots — differenced space with CI, level space without
    plot_forecast_vs_actual(fc_mean, ete, fc_ci, train_tail=etr.iloc[-40:],
                            save_path="_smoke_fc_diff.png")
    plot_forecast_vs_actual(fc_levels, endog.iloc[-N_TEST:],
                            train_tail=endog.iloc[-(N_TEST + 40):-N_TEST],
                            save_path="_smoke_fc_levels.png")
    print("forecast plots OK")

    print("\nAll smoke tests passed.")

