import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import OPTICS
from statsmodels.regression.linear_model import OLS
from statsmodels.tools.tools import add_constant
from arch.unitroot.cointegration import engle_granger
from hurst import compute_Hc

import plotly.graph_objects as go
from plotly.subplots import make_subplots

# -----------------------------
# Step 0: Universe (S&P 100)
# -----------------------------
sp100_tickers = [
    "AAPL","MSFT","AMZN","META","GOOGL","GOOG","TSLA","NVDA","BRK-B","JPM",
    "JNJ","V","PG","UNH","HD","MA","XOM","BAC","PFE","CVX","ABBV","KO","PEP",
    "LLY","AVGO","COST","MRK","TMO","DIS","CSCO","ABT","WMT","ACN","DHR","MCD",
    "NKE","NEE","TXN","LIN","PM","VZ","HON","ORCL","AMGN","UNP","BMY","RTX",
    "INTC","MS","GS","QCOM","IBM","LOW","CAT","GE","AMD","AMAT","SPGI","MDT",
    "SCHW","BKNG","LMT","DE","CVS","ISRG","C","BLK","AXP","NOW","T","SYK","ADBE",
    "GILD","MDLZ","DUK","REGN","MO","VRTX","SO","PNC","CB","CI","PYPL","ADI",
    "CSX","ZTS","MMC","BDX","EL","MU","NSC","APD","ADP","EQIX","ICE","SHW",
    "GM","CL","FIS","HUM","CME","ECL","USB","FDX","PSA","PLD","ROP","AON"
]

end_date = datetime.today()
start_date = end_date - timedelta(days=3*365)

print("Downloading price data...")
prices = yf.download(sp100_tickers, start=start_date, end=end_date, interval="1d")["Close"]
prices = prices.dropna(axis=1, how="any")
daily_returns = prices.pct_change().dropna()

# -----------------------------
# Step 1: PCA (3D coords)
# -----------------------------
standardized_returns = StandardScaler().fit_transform(daily_returns)
pca = PCA(n_components=3, random_state=0).fit(standardized_returns)

factor_exposures = pd.DataFrame(
    data=pca.components_.T,
    index=daily_returns.columns,
    columns=["component_0","component_1","component_2"]
)

# -----------------------------
# Step 2: OPTICS clustering
# -----------------------------
clustering = OPTICS(min_samples=5).fit(factor_exposures)
factor_exposures["cluster"] = clustering.labels_

# -----------------------------
# Step 3: Pair tests (relaxed)
# -----------------------------
from itertools import combinations
pairs_detected = []
test_results = {"cointegration":0,"hurst":0,"half_life":0,"mean_crossings":0}
pairs_tested = 0

for cluster_id in set(clustering.labels_):
    if cluster_id == -1:  # skip noise
        continue
    cluster_syms = factor_exposures[factor_exposures["cluster"]==cluster_id].index

    for sym_a, sym_b in combinations(cluster_syms, 2):
        pairs_tested += 1
        # Engle-Granger test
        try:
            test1 = engle_granger(prices[sym_a], prices[sym_b])
            test2 = engle_granger(prices[sym_b], prices[sym_a])
            candidate = min([test1,test2], key=lambda x:x.stat)
        except Exception:
            continue
        if getattr(candidate, "pvalue", 1.0) > 0.05:  # relaxed
            test_results["cointegration"] += 1
            continue

        # Spread
        spread = (
            prices[candidate.cointegrating_vector.index[0]] +
            candidate.cointegrating_vector.values[1] *
            prices[candidate.cointegrating_vector.index[1]]
        ).dropna()

        # Hurst
        try:
            H, _, _ = compute_Hc(spread, kind="price", simplified=False)
        except Exception:
            continue
        if H >= 0.55:  # relaxed
            test_results["hurst"] += 1
            continue

        # Half-life
        lagged = np.roll(spread.values, 1)
        lagged[0] = 0
        delta = spread.values - lagged
        delta[0] = 0
        model = OLS(delta, add_constant(lagged)).fit()
        beta = model.params[1]
        half_life = -np.log(2)/beta if beta != 0 else np.inf
        if not (1 < half_life < 500):  # relaxed
            test_results["half_life"] += 1
            continue

        # Mean crossings
        mean_val = spread.mean()
        crossings = ((spread > mean_val) & (spread.shift(1) <= mean_val)) | \
                    ((spread < mean_val) & (spread.shift(1) >= mean_val))
        if crossings.sum() < 24:  # relaxed
            test_results["mean_crossings"] += 1
            continue

        pairs_detected.append({
            "sym_a": sym_a, "sym_b": sym_b,
            "spread": spread, "hurst": H, "half_life": half_life,
            "cluster": cluster_id
        })

print(f"Pairs tested: {pairs_tested}")
print(f"Pairs detected: {len(pairs_detected)}")

# -----------------------------
# Step 4: Visualization
# -----------------------------
# 3D cluster plot
fig = go.Figure()
for cluster_id, group in factor_exposures.groupby("cluster"):
    fig.add_trace(go.Scatter3d(
        x=group["component_0"], y=group["component_1"], z=group["component_2"],
        mode="markers", name=f"Cluster {cluster_id}" if cluster_id>=0 else "Noise",
        text=[f"{sym}" for sym in group.index],
        marker=dict(size=4)
    ))
fig.update_layout(title="3D OPTICS Clusters of S&P 100",
                  scene=dict(xaxis_title="PC1", yaxis_title="PC2", zaxis_title="PC3"))
fig.show()

# Spread plots for pairs
if pairs_detected:
    rows = len(pairs_detected)
    fig2 = make_subplots(rows=rows, cols=2,
                         subplot_titles=["Prices","Normalized Spread"],
                         specs=[[{"secondary_y":True},{}] for _ in range(rows)])
    for i, pair in enumerate(pairs_detected, start=1):
        s_a, s_b = pair["sym_a"], pair["sym_b"]
        spread = pair["spread"]

        # Prices
        fig2.add_trace(go.Scatter(x=prices.index, y=prices[s_a], name=s_a, line=dict(color="blue")), row=i, col=1)
        fig2.add_trace(go.Scatter(x=prices.index, y=prices[s_b], name=s_b, line=dict(color="red")), row=i, col=1, secondary_y=True)

        # Spread normalized
        norm_spread = (spread - spread.mean())/spread.std()
        fig2.add_trace(go.Scatter(x=spread.index, y=norm_spread, name="Spread", line=dict(color="green")), row=i, col=2)

    fig2.update_layout(title="Detected Pairs (Prices & Spread)", height=300*rows, showlegend=False)
    fig2.show()
