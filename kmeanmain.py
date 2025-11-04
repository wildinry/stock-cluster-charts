import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from itertools import combinations
import warnings
import os # Added for potential database cleanup

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from statsmodels.regression.linear_model import OLS
from statsmodels.tools.tools import add_constant
from arch.unitroot.cointegration import engle_granger
from hurst import compute_Hc

import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# --- CONFIGURATION ---
N_CLUSTERS = 10
PCA_COMPONENTS = 3

# -----------------------------
# Step 0: Universe and Data Fetching (Attempting to handle errors)
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

# --- ENVIRONMENT FIX: Remove potentially corrupted yfinance cache ---
# The OperationalError ('unable to open database file') suggests a corrupted yfinance cache file.
# We attempt to delete it before running the download.
try:
    # yfinance often uses a file called 'yfinance.cache' or similar in the temp directory.
    # We will specifically target the default 'yfinance.sqlite' if it's causing the issue.
    # Note: This is an educated guess based on the error and yfinance's behavior.
    cache_path = os.path.expanduser('~/.cache/yfinance.sqlite')
    if os.path.exists(cache_path):
        os.remove(cache_path)
        print("Cleaned up potentially corrupted yfinance cache.")
except Exception as e:
    print(f"Could not clean up cache: {e}")
# --- END ENVIRONMENT FIX ---

print("Downloading price data...")
# We use a try/except block to catch connection errors and continue with the data we did get.
try:
    prices = yf.download(sp100_tickers, start=start_date, end=end_date, interval="1d")["Close"]
    prices = prices.dropna(axis=1, how="any")
    daily_returns = prices.pct_change().dropna()
except Exception as e:
    print(f"Initial download failed. Please check network/DNS settings. Error: {e}")
    # Handle the case where 'prices' might not be defined
    if 'prices' not in locals():
        raise SystemExit("Fatal: Cannot proceed without data.")

print(f"Successfully retrieved data for {len(prices.columns)} tickers.")

# -----------------------------
# Step 1: Feature Engineering (PCA & Hurst)
# -----------------------------

# PCA on standardized returns (3D features)
standardized_returns = StandardScaler().fit_transform(daily_returns)
pca = PCA(n_components=PCA_COMPONENTS, random_state=0).fit(standardized_returns)

factor_exposures = pd.DataFrame(
    data=pca.components_.T,
    index=daily_returns.columns,
    columns=[f"PC{i+1}" for i in range(PCA_COMPONENTS)]
)
print(f"PCA Variance Explained: {pca.explained_variance_ratio_.sum():.2f}")

# Calculate Hurst Exponent for each stock's returns (1D feature)
print("Calculating Hurst Exponents for individual stocks...")
def calculate_hurst(series):
    # compute_Hc on returns ('change') is often more appropriate for persistence
    H, c, data = compute_Hc(series, kind='change', simplified=True)
    return H

hurst_features = daily_returns.apply(calculate_hurst, axis=0)
factor_exposures["Hurst"] = hurst_features

# Combine and standardize features for clustering
clustering_features = StandardScaler().fit_transform(factor_exposures)
clustering_features_df = pd.DataFrame(clustering_features, index=factor_exposures.index)


# -----------------------------
# Step 2: K-Means Clustering
# -----------------------------
print(f"Running K-Means clustering with K={N_CLUSTERS}...")
kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=42, n_init='auto').fit(clustering_features)
factor_exposures["cluster"] = kmeans.labels_


# -----------------------------
# Step 3: Pair tests (using Cointegration, Hurst on Spread, Half-Life)
# -----------------------------
pairs_detected = []
pairs_tested = 0

print("\nStarting pairs detection...")

for cluster_id in sorted(factor_exposures["cluster"].unique()):
    cluster_syms = factor_exposures[factor_exposures["cluster"]==cluster_id].index

    if len(cluster_syms) < 2:
        continue

    for sym_a, sym_b in combinations(cluster_syms, 2):
        pairs_tested += 1

        # 1. Engle-Granger Cointegration Test
        try:
            test1 = engle_granger(prices[sym_a], prices[sym_b])
            test2 = engle_granger(prices[sym_b], prices[sym_a])
            candidate = min([test1,test2], key=lambda x:x.stat)
        except Exception:
            continue

        if getattr(candidate, "pvalue", 1.0) > 0.05:
            continue

        # Calculate Spread: Spread = Price_A + Beta * Price_B
        spread = (
            prices[candidate.cointegrating_vector.index[0]] +
            candidate.cointegrating_vector.values[1] *
            prices[candidate.cointegrating_vector.index[1]]
        ).dropna()

        # 2. Hurst Exponent of the Spread
        try:
            H, _, _ = compute_Hc(spread, kind="price", simplified=False)
        except Exception:
            continue
        if H >= 0.5:  # Require mean-reversion (H < 0.5) for pairs trading
            continue

        # 3. Half-life (Rate of mean-reversion)
        lagged = np.roll(spread.values, 1)
        lagged[0] = 0
        delta = spread.values - lagged
        delta[0] = 0

        model = OLS(delta, add_constant(lagged)).fit()
        beta = model.params[1]

        # Half-life = -ln(2) / beta
        half_life = -np.log(2)/beta if beta != 0 and beta < 0 else np.inf
        if not (10 < half_life < 252*2):
            continue

        pairs_detected.append({
            "sym_a": sym_a, "sym_b": sym_b,
            "spread": spread, "hurst": H, "half_life": half_life,
            "cluster": cluster_id
        })

print(f"Total pairs tested: {pairs_tested}")
print(f"Mean-reverting pairs detected: {len(pairs_detected)}")

# -----------------------------
# Step 4: Visualization
# -----------------------------

# 3D cluster plot (using PC1, PC2, PC3)
fig = go.Figure()
for cluster_id, group in factor_exposures.groupby("cluster"):
    fig.add_trace(go.Scatter3d(
        x=group["PC1"], y=group["PC2"], z=group["PC3"],
        mode="markers", name=f"Cluster {cluster_id}",
        text=[f"{sym} (H={group.loc[sym, 'Hurst']:.2f})" for sym in group.index],
        marker=dict(size=4)
    ))
fig.update_layout(title="3D K-Means Clusters (PC1, PC2, PC3)",
                  scene=dict(xaxis_title="PC1", yaxis_title="PC2", zaxis_title="PC3"))
fig.show()

# Spread plots for pairs (using Z-Score for signal visualization)
if pairs_detected:
    rows = len(pairs_detected)
    fig2 = make_subplots(rows=rows, cols=2,
                         subplot_titles=["Prices (Normalized)", "Spread Z-Score (Signal)"],
                         specs=[[{"secondary_y":True},{}] for _ in range(rows)])

    for i, pair in enumerate(pairs_detected, start=1):
        s_a, s_b = pair["sym_a"], pair["sym_b"]
        spread = pair["spread"]

        # 1. Prices (Normalized)
        norm_a = (prices[s_a] - prices[s_a].mean())/prices[s_a].std()
        norm_b = (prices[s_b] - prices[s_b].mean())/prices[s_b].std()
        fig2.add_trace(go.Scatter(x=prices.index, y=norm_a, name=s_a, line=dict(color="blue")), row=i, col=1)
        fig2.add_trace(go.Scatter(x=prices.index, y=norm_b, name=s_b, line=dict(color="red")), row=i, col=1)

        # 2. Spread Z-Score (Signal)
        z_score_spread = (spread - spread.mean())/spread.std()
        hline_up = 2.0
        hline_down = -2.0
        hline_mean = 0

        fig2.add_trace(go.Scatter(x=spread.index, y=z_score_spread, name=f"{s_a}/{s_b} Spread Z-Score", line=dict(color="green")), row=i, col=2)
        fig2.add_hline(y=hline_up, line_dash="dash", line_color="red", row=i, col=2)
        fig2.add_hline(y=hline_down, line_dash="dash", line_color="red", row=i, col=2)
        fig2.add_hline(y=hline_mean, line_dash="solid", line_color="gray", row=i, col=2)

        fig2.update_yaxes(title_text="Z-Score", row=i, col=2, range=[-3, 3])

        # FIX: Replace fig2.update_annotations with fig2.add_annotation
        # Add the Hurst and Half-life stats to the top right of the Z-Score plot
        fig2.add_annotation(
            text=f'H={pair["hurst"]:.2f}, HL={pair["half_life"]:.0f}d',
            xref=f"x{2*i}", yref=f"y{2*i}", # x/y axes for the 2nd column of the current row
            x=spread.index[-1], y=z_score_spread.iloc[-1], # Position the text at the last data point
            showarrow=False,
            row=i, col=2,
            bgcolor="rgba(255,255,255,0.7)"
        )


    fig2.update_layout(title="Detected Mean-Reverting Pairs (Z-Score Signal)", height=300*rows, showlegend=False)
    fig2.show()