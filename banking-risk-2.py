import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import timedelta
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.metrics import classification_report

# ---------------------------
# 1. Data Simulation Function
# ---------------------------
def simulate_client_data(n_clients=50, start_date='2021-01-01', end_date='2023-12-31', seed=42):
    """
    Simulates daily data for n_clients including loan utilization, operating deposits,
    and a credit risk rating. The simulation adds trend, seasonality and noise.
    """
    np.random.seed(seed)
    dates = pd.date_range(start_date, end_date, freq='D')
    n_days = len(dates)
    
    # Create an empty dataframe list
    data_list = []
    for client in range(1, n_clients+1):
        # Base values
        base_loan = np.random.uniform(50, 200)    # base loan utilization in million $
        base_dep = np.random.uniform(100, 500)      # base operating deposit in million $
        
        # Simulate a linear trend over the year plus seasonality
        trend_loan = np.linspace(0, np.random.uniform(-20, 20), n_days)
        trend_dep = np.linspace(0, np.random.uniform(-30, 30), n_days)
        seasonal_loan = 10 * np.sin(np.linspace(0, 3*np.pi, n_days) + np.random.uniform(0, 2*np.pi))
        seasonal_dep = 15 * np.cos(np.linspace(0, 3*np.pi, n_days) + np.random.uniform(0, 2*np.pi))
        
        # Random noise
        noise_loan = np.random.normal(0, 5, n_days)
        noise_dep = np.random.normal(0, 5, n_days)
        
        loan_utilization = base_loan + trend_loan + seasonal_loan + noise_loan
        operating_deposits = base_dep + trend_dep + seasonal_dep + noise_dep
        
        # Simulate credit risk rating (categorical: 1, 1-, 1+, …, 10) 
        # Here we simulate as a continuous risk score that we later bucket.
        risk_score = np.clip(np.random.normal(3, 1, n_days) + 0.01*np.arange(n_days), 1, 10)
        # For simplicity, we create buckets
        def bucket_rating(score):
            if score < 2:
                return '1'
            elif score < 2.5:
                return '1-'
            elif score < 3:
                return '1+'
            elif score < 4:
                return '2'
            elif score < 5:
                return '2-'
            elif score < 6:
                return '2+'
            elif score < 7:
                return '3'
            elif score < 8:
                return '3-'
            elif score < 9:
                return '3+'
            else:
                return '4'
        ratings = [bucket_rating(s) for s in risk_score]
        
        df_client = pd.DataFrame({
            'date': dates,
            'client_id': client,
            'loan_utilization': loan_utilization,
            'operating_deposits': operating_deposits,
            'credit_rating': ratings
        })
        data_list.append(df_client)
        
    data = pd.concat(data_list, ignore_index=True)
    return data

# ---------------------------
# 2. KPI Calculation Functions
# ---------------------------
def compute_rolling_correlation(df, window=30):
    """
    Computes rolling correlation between loan utilization and operating deposits
    for each client over a specified window (in days).
    """
    df = df.sort_values(['client_id', 'date'])
    df['rolling_corr'] = df.groupby('client_id').apply(
        lambda x: x[['loan_utilization','operating_deposits']].rolling(window, min_periods=5).corr().unstack().iloc[:,1]
    ).reset_index(level=0, drop=True)
    return df

def compute_trend_slopes(df, window=30):
    """
    For each client, compute the rolling trend (slope) for both metrics over a window.
    Returns additional columns for loan slope and deposit slope.
    """
    from scipy.stats import linregress
    def rolling_slope(series, window):
        slopes = series.rolling(window).apply(
            lambda y: linregress(np.arange(len(y)), y)[0] if len(y.dropna())==window else np.nan, raw=False
        )
        return slopes
    
    df = df.sort_values(['client_id', 'date'])
    df['loan_slope'] = df.groupby('client_id')['loan_utilization'].apply(lambda x: rolling_slope(x, window))
    df['dep_slope'] = df.groupby('client_id')['operating_deposits'].apply(lambda x: rolling_slope(x, window))
    return df

# ---------------------------
# 3. Early Warning Signal Function
# ---------------------------
def flag_early_warning(df, recent_days=90, loan_increase_threshold=0.05, deposit_decrease_threshold=0.02):
    """
    Flag clients where recent trend in loan utilization and operating deposits indicate risk.
    For each client, if in the past 'recent_days' the average loan growth is above a threshold
    while deposit growth is below a (possibly negative) threshold, flag as risky.
    """
    flag_list = []
    for client, group in df.groupby('client_id'):
        recent = group.sort_values('date').tail(recent_days)
        # Compute percentage change over the period
        loan_change = (recent['loan_utilization'].iloc[-1] - recent['loan_utilization'].iloc[0]) / recent['loan_utilization'].iloc[0]
        dep_change = (recent['operating_deposits'].iloc[-1] - recent['operating_deposits'].iloc[0]) / recent['operating_deposits'].iloc[0]
        # Flag if loan increases by more than threshold and deposit increases less than deposit_decrease_threshold (or negative)
        warning = (loan_change > loan_increase_threshold) and (dep_change < deposit_decrease_threshold)
        flag_list.append({'client_id': client, 'loan_change': loan_change, 'dep_change': dep_change, 'early_warning': warning})
    warning_df = pd.DataFrame(flag_list)
    return warning_df

# ---------------------------
# 4. Clustering & Cohort Analysis Functions
# ---------------------------
def extract_features_for_clustering(df, window=30):
    """
    For each client, compute aggregated features over the most recent window period.
    Features include: mean loan utilization, mean operating deposits, rolling correlation,
    and trend slopes. Returns a dataframe with one row per client.
    """
    feature_list = []
    for client, group in df.groupby('client_id'):
        recent = group.sort_values('date').tail(window)
        features = {
            'client_id': client,
            'mean_loan': recent['loan_utilization'].mean(),
            'std_loan': recent['loan_utilization'].std(),
            'mean_dep': recent['operating_deposits'].mean(),
            'std_dep': recent['operating_deposits'].std(),
            'mean_corr': recent['rolling_corr'].mean(),
            'loan_slope': recent['loan_slope'].mean(),
            'dep_slope': recent['dep_slope'].mean()
        }
        feature_list.append(features)
    features_df = pd.DataFrame(feature_list).fillna(0)
    return features_df

def perform_clustering(features_df, n_clusters=4):
    """
    Standardize features and perform KMeans clustering. Returns cluster assignments.
    """
    feature_cols = ['mean_loan', 'std_loan', 'mean_dep', 'std_dep', 'mean_corr', 'loan_slope', 'dep_slope']
    scaler = StandardScaler()
    X = scaler.fit_transform(features_df[feature_cols])
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    features_df['cluster'] = kmeans.fit_predict(X)
    return features_df, kmeans

def time_varying_clustering(df, window=60, n_clusters=4):
    """
    Performs clustering in a rolling window manner for each time step (or each window's end)
    to track how client cluster memberships change over time.
    Returns a dataframe with date, client_id and cluster assignment.
    """
    cluster_records = []
    dates = sorted(df['date'].unique())
    # Choose rolling windows (for example, every 30 days)
    step = 30
    for i in range(0, len(dates)-window, step):
        window_start = dates[i]
        window_end = dates[i+window-1]
        sub_df = df[(df['date'] >= window_start) & (df['date'] <= window_end)]
        feat = extract_features_for_clustering(sub_df, window=window)
        if feat.empty:
            continue
        feat, _ = perform_clustering(feat, n_clusters=n_clusters)
        feat['window_end'] = window_end
        cluster_records.append(feat[['client_id', 'cluster', 'window_end']])
    clusters_over_time = pd.concat(cluster_records, ignore_index=True)
    return clusters_over_time

# ---------------------------
# 5. Forecasting Risk Downgrades / Defaults
# ---------------------------
def simulate_target_variable(df, warning_df):
    """
    Create a target variable for each client (e.g., rating downgrade in next 3 months or default)
    For simulation, we assume clients with early warning signals are more likely to get downgraded.
    Returns a dataframe with client_id and target binary flag.
    """
    # Merge with warning signals
    target = warning_df.copy()
    # For simulation, add some randomness: if early warning flagged, 60% chance of downgrade;
    # if not, 10% chance.
    target['downgrade'] = target['early_warning'].apply(lambda x: np.random.choice([1, 0], p=[0.6, 0.4]) if x 
                                                         else np.random.choice([1, 0], p=[0.1, 0.9]))
    return target[['client_id', 'downgrade']]

def forecast_risk(features_df, target_df):
    """
    Builds a logistic regression model to predict risk downgrades.
    Uses the aggregated features as predictors.
    """
    # Merge features and target
    df_model = features_df.merge(target_df, on='client_id')
    X = df_model[['mean_loan', 'std_loan', 'mean_dep', 'std_dep', 'mean_corr', 'loan_slope', 'dep_slope']]
    y = df_model['downgrade']
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    model = LogisticRegression(random_state=42)
    model.fit(X_scaled, y)
    
    y_pred = model.predict(X_scaled)
    print("Classification Report for Risk Forecast:")
    print(classification_report(y, y_pred))
    
    # For each client, also output predicted probability
    df_model['downgrade_prob'] = model.predict_proba(X_scaled)[:, 1]
    return model, df_model

# ---------------------------
# 6. Visualization Functions
# ---------------------------
def plot_time_series_for_client(df, client_id):
    """
    Plots the time series for a given client including loan utilization and operating deposits.
    """
    client_df = df[df['client_id'] == client_id].sort_values('date')
    plt.figure(figsize=(14,5))
    plt.plot(client_df['date'], client_df['loan_utilization'], label='Loan Utilization')
    plt.plot(client_df['date'], client_df['operating_deposits'], label='Operating Deposits')
    plt.title(f'Client {client_id} - Loan Utilization & Operating Deposits')
    plt.xlabel('Date')
    plt.ylabel('Value')
    plt.legend()
    plt.show()

def plot_cluster_evolution(clusters_over_time):
    """
    Visualizes the evolution of clusters over time by showing the count of clients per cluster.
    """
    cluster_counts = clusters_over_time.groupby(['window_end','cluster']).size().reset_index(name='count')
    plt.figure(figsize=(12,6))
    sns.lineplot(data=cluster_counts, x='window_end', y='count', hue='cluster', marker="o")
    plt.title('Time-Varying Cluster Evolution')
    plt.xlabel('Window End Date')
    plt.ylabel('Number of Clients')
    plt.show()

def plot_forecast_results(df_model):
    """
    Plots the distribution of predicted downgrade probabilities.
    """
    plt.figure(figsize=(10,6))
    sns.histplot(df_model['downgrade_prob'], bins=20, kde=True)
    plt.title('Distribution of Predicted Downgrade Probabilities')
    plt.xlabel('Downgrade Probability')
    plt.ylabel('Frequency')
    plt.show()

# ---------------------------
# 7. Main Analysis Pipeline
# ---------------------------
def run_analysis_pipeline():
    # Step 1: Generate synthetic data
    data = simulate_client_data()
    print("Data simulation complete. Data shape:", data.shape)
    
    # Step 2: Compute rolling KPIs
    data = compute_rolling_correlation(data, window=30)
    data = compute_trend_slopes(data, window=30)
    print("KPI calculations complete.")
    
    # Step 3: Flag early warning signals based on recent 90 days behavior
    warnings = flag_early_warning(data, recent_days=90)
    print("Early warning signal computation complete.")
    
    # Step 4: Extract features and perform clustering for a cohort analysis
    features = extract_features_for_clustering(data, window=30)
    features, kmeans_model = perform_clustering(features, n_clusters=4)
    print("Static clustering complete. Cluster counts:")
    print(features['cluster'].value_counts())
    
    # Perform time-varying clustering to see evolution
    clusters_over_time = time_varying_clustering(data, window=60, n_clusters=4)
    print("Time-varying clustering complete.")
    
    # Step 5: Simulate target variable and forecast risk downgrades / defaults
    target = simulate_target_variable(data, warnings)
    model, forecast_df = forecast_risk(features, target)
    
    # Step 6: Visualization
    # Plot time series for a sample client
    sample_client = 1
    plot_time_series_for_client(data, sample_client)
    
    # Plot cluster evolution over time
    plot_cluster_evolution(clusters_over_time)
    
    # Plot forecast result distribution
    plot_forecast_results(forecast_df)
    
    return {
        "data": data,
        "warnings": warnings,
        "features": features,
        "clusters_over_time": clusters_over_time,
        "forecast_df": forecast_df,
        "risk_model": model
    }

# ---------------------------
# Execute the Analysis Pipeline
# ---------------------------
if __name__ == '__main__':
    results = run_analysis_pipeline()
