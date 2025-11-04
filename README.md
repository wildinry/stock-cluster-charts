📈 Quant Pairs Trading EngineThis repository hosts an advanced quantitative finance application designed to detect stable, mean-reverting stock pairs ideal for Pairs Trading strategies.The analysis pipeline leverages Principal Component Analysis (PCA) to model risk factors, the Hurst Exponent to quantify market memory, and Cointegration (Engle-Granger) to validate long-term price relationships.The core distinction between the two included scripts is the clustering approach used to intelligently group related assets: OPTICS (Density-Based) and K-Means (Centroid-Based).🛠️ Setup and Installation1. Clone the RepositoryBashgit clone [REPO_URL]
cd stock-cluster-charts
2. File StructureEnsure your Python scripts are named as follows:File NameClustering Algorithmmain.pyOPTICSkmeans_clustering.pyK-Means3. Create and Activate a Virtual EnvironmentUsing a virtual environment is highly recommended to manage project dependencies.Bash# Create the environment
python3 -m venv venv

# Activate (macOS/Linux)
source venv/bin/activate

# Activate (Windows)
# .\venv\Scripts\activate
4. Install DependenciesInstall all required quantitative and visualization libraries in a single command:Bashpip install yfinance pandas numpy scikit-learn statsmodels arch hurst plotly
🚀 Usage and ExecutionBoth scripts analyze the S&P 100 universe, fetching three years of historical price data. The output for both is an interactive 3D cluster plot and plots of the detected mean-reverting pairs' Z-Scores.1. K-Means ClusteringThis script uses K-Means, clustering assets based on PCA components + individual stock Hurst Exponents. This approach explicitly groups stocks with similar factor exposure and market memory characteristics.Bashpython kmeans_clustering.py
2. OPTICS ClusteringThis script uses the OPTICS algorithm, which is density-based and can find arbitrarily shaped clusters without requiring a fixed number of clusters (K).Bashpython main.py
💡 Trading InterpretationThe final output is the Spread Z-Score chart, which serves as the direct trading signal for mean-reversion pairs.IndicatorValueTrading ActionZ-Score$> +2.0$ (Overbought)SELL the Spread (Short the Expensive Stock / Long the Cheap Stock).Z-Score$< -2.0$ (Oversold)BUY the Spread (Long the Expensive Stock / Short the Cheap Stock).Z-Score$\approx 0$ (Mean)EXIT the trade to realize profit.Spread Hurst$< 0.5$CONFIRMATION: Indicates the spread is statistically mean-reverting.Half-Life$10-60$ daysVALIDATION: Indicates a desirable mean-reversion speed for swing trading.
