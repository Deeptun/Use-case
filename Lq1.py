import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dtw import dtw
from statsmodels.tsa.seasonal import STL
from tslearn.clustering import TimeSeriesKMeans
from tslearn.utils import to_time_series_dataset
from ruptures import Binseg
from scipy import fft, signal
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.decomposition import PCA
from joblib import Parallel, delayed
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Configuration
np.random.seed(42)
N_CLIENTS = 4000
START_DATE = pd.Timestamp('2018-01-01')
END_DATE = pd.Timestamp('2022-12-31')
N_JOBS = 8  # Set based on available CPU cores

# 1. Enhanced Synthetic Data Generator with Regime Changes
def generate_client_data(client_id):
    dates = pd.date_range(START_DATE, END_DATE, freq='D')
    n_days = len(dates)
    
    # Base deposits with regime changes
    regime_changes = np.sort(np.random.choice(
        np.arange(n_days), 
        size=np.random.poisson(3),
        replace=False
    ))
    regimes = np.zeros(n_days, dtype=int)
    current_regime = 0
    for rc in regime_changes:
        regimes[rc:] = current_regime
        current_regime += 1
    
    deposits = np.zeros(n_days)
    for r in np.unique(regimes):
        mask = regimes == r
        deposits[mask] = np.random.lognormal(
            mean=10 + r*0.5, 
            sigma=0.3 + r*0.1,
            size=mask.sum()
        )
    
    # Add correlated economic cycle
    economic_cycle = 0.2 * signal.sawtooth(2 * np.pi * 1/365 * dates.dayofyear, width=0.5)
    
    # Multi-frequency seasonality
    seasonality = (
        0.4 * np.sin(2 * np.pi * dates.dayofyear/365) +
        0.3 * np.cos(4 * np.pi * dates.dayofyear/365) +
        0.2 * np.sin(8 * np.pi * dates.dayofyear/365)
    
    # Combine components
    deposits = deposits * (1 + 0.1*seasonality + 0.05*economic_cycle) * np.random.lognormal(0, 0.05, n_days)
    
    # Loan utilization with hysteresis
    loans = np.zeros(n_days)
    utilization = 0.4
    for t in range(1, n_days):
        deposit_change = (deposits[t] - deposits[t-1])/deposits[t-1]
        utilization += 0.1 * deposit_change - 0.05 * utilization + np.random.normal(0, 0.01)
        utilization = np.clip(utilization, 0.1, 0.8)
        loans[t] = utilization * deposits[t]
    
    return pd.DataFrame({
        'client_id': client_id,
        'date': dates,
        'operating_deposits': np.abs(deposits),
        'loan_utilization': loans,
        'regime': regimes
    })

# Parallel data generation
print("Generating synthetic data...")
client_dfs = Parallel(n_jobs=N_JOBS)(
    delayed(generate_client_data)(i) for i in range(N_CLIENTS))
df = pd.concat(client_dfs)

# 2. CPU-Optimized Feature Engineering
def calculate_ts_features(client_df):
    deposits = client_df['operating_deposits'].values
    loans = client_df['loan_utilization'].values
    
    # STL Decomposition
    stl = STL(client_df['operating_deposits'], period=365, robust=True).fit()
    
    # FFT using SciPy
    fft_vals = np.abs(fft.fft(deposits - np.mean(deposits))[1:len(deposits)//2]
    dominant_freq = np.argmax(fft_vals)/len(deposits)
    
    # Change point detection
    algo = Binseg(model='l2').fit(deposits)
    change_points = len(algo.predict(n_bkps=3))
    
    # Cross-correlation
    ccf = signal.correlate(deposits - np.mean(deposits),
                          loans - np.mean(loans),
                          mode='full')
    max_ccf_lag = np.argmax(ccf) - len(deposits) + 1
    
    # Liquidity risk features
    liquidity_gap = deposits - loans
    crisis_days = np.sum(liquidity_gap < 0)/len(deposits)
    
    return {
        'client_id': client_df['client_id'].iloc[0],
        'trend_strength': max(0, 1 - (np.var(stl.resid)/np.var(stl.trend + stl.resid))),
        'seasonality_strength': max(0, 1 - (np.var(stl.resid)/np.var(stl.seasonal + stl.resid))),
        'change_point_freq': change_points/len(deposits),
        'dominant_freq': dominant_freq,
        'max_ccf_lag': max_ccf_lag,
        'crisis_days': crisis_days,
        'avg_liquidity_gap': np.mean(liquidity_gap),
        'volatility': np.std(deposits)
    }

print("Calculating features...")
feature_list = Parallel(n_jobs=N_JOBS)(
    delayed(calculate_ts_features)(df[df['client_id'] == cid]) 
    for cid in df['client_id'].unique())
features = pd.DataFrame(feature_list)

# 3. CPU-optimized DTW Clustering
def cpu_dtw_clustering(features, n_clusters=5):
    # Prepare time series data
    series_data = df.groupby('client_id')['operating_deposits'].apply(np.array)
    dataset = to_time_series_dataset(series_data.tolist())
    
    # CPU-optimized DTW clustering
    model = TimeSeriesKMeans(n_clusters=n_clusters, 
                            metric="dtw",
                            max_iter=10,
                            random_state=42,
                            n_init=3,
                            n_jobs=N_JOBS)
    
    # Dimensionality reduction
    scaler = StandardScaler()
    pca = PCA(n_components=0.95)
    reduced_data = pca.fit_transform(scaler.fit_transform(features))
    
    # Clustering
    clusters = model.fit_predict(reduced_data)
    return clusters

print("Performing clustering...")
features['cluster'] = cpu_dtw_clustering(features.drop('client_id', axis=1))

# 4. Risk Scoring and Early Warning System (unchanged)
def calculate_risk_scores(features):
    risk_factors = features[[
        'crisis_days', 'change_point_freq', 
        'volatility', 'avg_liquidity_gap'
    ]].copy()
    
    # Normalize and weight factors
    scaler = StandardScaler()
    risk_factors = pd.DataFrame(
        scaler.fit_transform(risk_factors),
        columns=risk_factors.columns
    )
    
    # Custom weights
    weights = np.array([0.4, 0.3, 0.2, 0.1])
    risk_factors['risk_score'] = risk_factors.dot(weights)
    
    # Early warning flags
    risk_factors['early_warning'] = risk_factors['risk_score'] > 1.5
    return risk_factors

features = features.merge(calculate_risk_scores(features), 
                        left_index=True, right_index=True)

# 5. Interactive Visualizations (unchanged)
def create_dashboard(features, df):
    fig = make_subplots(
        rows=2, cols=2,
        specs=[[{"type": "scatter3d"}, {"type": "heatmap"}],
               [{"type": "scatter"}, {"type": "scatter"}]],
        subplot_titles=(
            "3D Cluster Visualization",
            "Risk Score Distribution",
            "Cluster Evolution",
            "Early Warning Signals"
        )
    )

    # 3D Cluster Plot
    fig.add_trace(
        go.Scatter3d(
            x=features['trend_strength'],
            y=features['volatility'],
            z=features['avg_liquidity_gap'],
            mode='markers',
            marker=dict(
                color=features['cluster'],
                size=features['risk_score']*10,
                opacity=0.7
            ),
            text=features['client_id']
        ),
        row=1, col=1
    )

    # Risk Score Heatmap
    risk_heatmap = features.pivot_table(
        index='cluster',
        columns=pd.qcut(features['risk_score'], 5),
        values='client_id',
        aggfunc='count'
    )
    fig.add_trace(
        go.Heatmap(
            z=risk_heatmap.values,
            x=risk_heatmap.columns.astype(str),
            y=risk_heatmap.index.astype(str),
            colorscale='Viridis'
        ),
        row=1, col=2
    )

    # Cluster Evolution
    cluster_ts = df.merge(features[['client_id', 'cluster']], on='client_id')
    cluster_ts = cluster_ts.groupby(['date', 'cluster'])['operating_deposits'].mean().reset_index()
    
    for cluster in sorted(cluster_ts['cluster'].unique()):
        cluster_data = cluster_ts[cluster_ts['cluster'] == cluster]
        fig.add_trace(
            go.Scatter(
                x=cluster_data['date'],
                y=cluster_data['operating_deposits'],
                name=f'Cluster {cluster}',
                mode='lines'
            ),
            row=2, col=1
        )

    # Early Warning Signals
    warnings = features[features['early_warning']].groupby('cluster').size()
    fig.add_trace(
        go.Bar(
            x=warnings.index,
            y=warnings.values,
            marker_color='crimson'
        ),
        row=2, col=2
    )

    fig.update_layout(
        height=1200,
        width=1600,
        title_text="Liquidity Risk Analytics Dashboard",
        showlegend=False
    )
    fig.show()

print("Generating dashboard...")
create_dashboard(features, df)

# 6. Automated Reporting (unchanged)
def generate_report(features):
    report = {
        "high_risk_clusters": features[features['risk_score'] > 1.5]['cluster'].value_counts().to_dict(),
        "cluster_characteristics": features.groupby('cluster').mean().to_dict(),
        "early_warnings": features['early_warning'].sum(),
        "volatility_distribution": features['volatility'].describe().to_dict()
    }
    
    print("\nLiquidity Risk Report:")
    print(f"High risk clients: {report['early_warnings']}")
    print(f"Most risky cluster: {max(report['high_risk_clusters'], key=report['high_risk_clusters'].get)}")
    print(f"Average volatility: {report['volatility_distribution']['mean']:.2f}")
    
    return report

final_report = generate_report(features)


**************************************************************

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import STL
from tslearn.clustering import TimeSeriesKMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from ruptures import Binseg
from scipy import fft

# Enhanced Configuration
N_CLIENTS = 4000
START_DATE = pd.Timestamp('2018-01-01')
END_DATE = pd.Timestamp('2022-12-31')

# 1. Improved Synthetic Data Generator with More Realistic Patterns
def generate_client_data_v2(client_id):
    dates = pd.date_range(START_DATE, END_DATE, freq='D')
    n_days = len(dates)
    
    # Base pattern with regime changes
    regimes = np.cumsum(np.random.choice([0,1], size=n_days, p=[0.995, 0.005]))
    base = np.zeros(n_days)
    for r in np.unique(regimes):
        mask = regimes == r
        base[mask] = np.random.lognormal(mean=10 + r*2, sigma=0.5)
    
    # Enhanced seasonality with multiple frequencies
    seasonality = (
        0.5 * np.sin(dates.dayofyear/365 * 2*np.pi) +
        0.3 * np.cos(dates.dayofyear/365 * 4*np.pi) +
        0.2 * np.sin(dates.dayofyear/365 * 8*np.pi)
    
    # Add economic cycles (3-year period)
    economic_cycle = 0.3 * np.sin(dates.dayofyear/(365*3) * 2*np.pi)
    
    # Combine components
    deposits = base * (1 + 0.1*seasonality + 0.05*economic_cycle)
    deposits *= np.random.lognormal(0, 0.05, n_days)
    
    # Loan utilization with hysteresis effect
    loans = np.zeros(n_days)
    loans[0] = deposits[0] * 0.2
    for t in range(1, n_days):
        loans[t] = 0.8*loans[t-1] + 0.2*(deposits[t] * np.random.uniform(0.1, 0.7))
    
    return pd.DataFrame({
        'client_id': client_id,
        'date': dates,
        'operating_deposits': np.abs(deposits),
        'loan_utilization': np.clip(loans, 0, None)
    })

# Generate all clients data
df = pd.concat([generate_client_data_v2(i) for i in range(N_CLIENTS)])

# 2. Advanced Feature Engineering Pipeline
def create_ts_features(client_series):
    """Create comprehensive time-series features for a single client"""
    features = {}
    series = client_series['operating_deposits']
    loans = client_series['loan_utilization']
    
    # STL Decomposition
    stl = STL(series, period=365, robust=True).fit()
    features.update({
        'stl_trend_strength': max(0, 1 - (np.var(stl.resid)/np.var(stl.trend + stl.resid))),
        'stl_seasonality_strength': max(0, 1 - (np.var(stl.resid)/np.var(stl.seasonal + stl.resid))),
        'trend_slope': np.polyfit(np.arange(len(series)), stl.trend, 1)[0]
    })
    
    # Change Point Detection
    algo = Binseg(model='l2').fit(series.values)
    change_points = algo.predict(n_bkps=3)
    features['change_point_freq'] = len(change_points)/len(series)
    
    # Frequency Domain Features
    fft_vals = np.abs(fft.fft(series - series.mean())[1:len(series)//2])
    features['dominant_freq'] = np.argmax(fft_vals)/len(series)
    
    # Cross-correlation features
    ccf = np.correlate(series - series.mean(), loans - loans.mean(), mode='full')
    features['max_ccf_lag'] = np.argmax(ccf) - len(series) + 1
    
    # Liquidity Risk Features
    features['liquidity_gap_30d'] = series.rolling(30).mean() - loans.rolling(30).mean()
    
    return pd.Series(features)

# 3. Efficient Feature Calculation with Sliding Windows
def calculate_features(df, window_size=90, step_size=30):
    features = []
    dates = pd.date_range(df['date'].min() + pd.Timedelta(days=window_size), 
                         df['date'].max(), freq=f'{step_size}D')
    
    for end_date in dates:
        window_df = df[(df['date'] > end_date - pd.Timedelta(days=window_size)) &
                      (df['date'] <= end_date)]
        
        window_features = window_df.groupby('client_id').apply(create_ts_features)
        window_features['period'] = end_date
        features.append(window_features)
    
    return pd.concat(features).reset_index()

features_df = calculate_features(df)

# 4. Dynamic Time Warping Clustering
def dtw_clustering(features, n_clusters=5):
    # Align time series using DTW
    scaler = StandardScaler()
    pca = PCA(n_components=0.95)
    
    pipeline = make_pipeline(scaler, pca)
    reduced_features = pipeline.fit_transform(features)
    
    # Time-series aware clustering
    model = TimeSeriesKMeans(n_clusters=n_clusters, 
                            metric="dtw", 
                            max_iter=10,
                            random_state=42)
    clusters = model.fit_predict(reduced_features)
    return clusters

# 5. Temporal Consistency Analysis
def analyze_cluster_transitions(cluster_assignments):
    transitions = cluster_assignments.pivot_table(
        index='client_id', columns='period', values='cluster'
    ).ffill(axis=1).bfill(axis=1)
    
    # Calculate transition probabilities
    transition_matrix = pd.crosstab(
        transitions.iloc[:, :-1].values.flatten(),
        transitions.iloc[:, 1:].values.flatten(),
        normalize='index'
    )
    
    # Plot transition heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(transition_matrix, annot=True, cmap='YlGnBu', fmt='.1%')
    plt.title('Cluster Transition Probabilities')
    plt.show()
    
    return transition_matrix

# 6. Enhanced Visualization with Plotly
import plotly.express as px

def plot_cluster_evolution(features_df):
    cluster_counts = features_df.groupby(['period', 'cluster']).size().reset_index(name='count')
    
    fig = px.line(cluster_counts, x='period', y='count', color='cluster',
                  title='Cluster Population Evolution',
                  labels={'count': 'Number of Clients', 'period': 'Date'},
                  template='plotly_dark')
    
    fig.update_layout(hovermode='x unified')
    fig.show()

# 7. Liquidity Early Warning System
def calculate_risk_scores(features_df):
    risk_features = features_df.groupby('client_id').agg({
        'stl_trend_strength': 'mean',
        'change_point_freq': 'mean',
        'dominant_freq': 'std',
        'liquidity_gap_30d': lambda x: (x < 0).mean()
    })
    
    scaler = StandardScaler()
    risk_scores = pd.DataFrame(
        scaler.fit_transform(risk_features),
        columns=risk_features.columns,
        index=risk_features.index
    )
    
    risk_scores['overall_risk'] = risk_scores.mean(axis=1)
    return risk_scores

# Execute Pipeline
if __name__ == "__main__":
    # Feature Engineering
    features_df = calculate_features(df)
    
    # Clustering
    features_df['cluster'] = dtw_clustering(features_df.drop(['client_id', 'period'], axis=1))
    
    # Analysis
    transition_matrix = analyze_cluster_transitions(features_df)
    risk_scores = calculate_risk_scores(features_df)
    plot_cluster_evolution(features_df)
    
    # Generate 3D Cluster Visualization
    fig = px.scatter_3d(
        features_df.sample(1000),
        x='stl_trend_strength',
        y='change_point_freq',
        z='liquidity_gap_30d',
        color='cluster',
        size='dominant_freq',
        hover_data=['client_id'],
        title='3D Cluster Visualization'
    )
    fig.show()


*****************************************
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from dateutil.relativedelta import relativedelta

# Configuration
np.random.seed(42)
N_CLIENTS = 4000
START_DATE = '2020-01-01'
END_DATE = '2022-12-31'
SEGMENTS = ['Retail', 'Consumer']
CLIENT_SEGMENT_PROBS = [0.6, 0.4]

# 1. Generate Realistic Synthetic Data
def generate_client_data(client_id, segment):
    dates = pd.date_range(START_DATE, END_DATE, freq='D')
    n_days = len(dates)
    
    # Base patterns
    base_deposit = np.random.lognormal(mean=10, sigma=0.5 if segment=='Retail' else 0.3)
    base_loan = base_deposit * np.random.uniform(0.2, 0.8)
    
    # Trend component
    trend = np.linspace(0, np.random.normal(0.1, 0.05), n_days)
    
    # Seasonality
    seasonality = (np.sin(dates.dayofyear / 365 * 2 * np.pi) * 
                  np.random.normal(1, 0.2) + 
                  np.cos(dates.dayofyear / 365 * 4 * np.pi) * 
                  np.random.normal(0.5, 0.1))
    
    # Events/Shocks
    shocks = np.zeros(n_days)
    shock_days = np.random.choice(n_days, size=np.random.poisson(3), replace=False)
    shocks[shock_days] = np.random.normal(-0.3, 0.1, len(shock_days))
    
    # Combine components
    deposits = (base_deposit * (1 + trend + 0.2*seasonality + shocks) * 
               np.random.lognormal(0, 0.1, n_days))
    loans = (base_loan * (1 + trend*1.2 + 0.1*seasonality + shocks*0.8) * 
            np.random.lognormal(0, 0.08, n_days))
    
    return pd.DataFrame({
        'client_id': client_id,
        'date': dates,
        'operating_deposits': np.abs(deposits),
        'loan_utilization': np.clip(loans, 0, None),
        'segment': segment
    })

# Generate all clients data
client_segments = np.random.choice(SEGMENTS, N_CLIENTS, p=CLIENT_SEGMENT_PROBS)
df = pd.concat([generate_client_data(i, seg) for i, seg in enumerate(client_segments)])

# 2. Calculate KPIs
def calculate_kpis(df):
    df = df.sort_values(['client_id', 'date'])
    
    # Loan-to-Deposit Ratio (LDR)
    df['ldr'] = df['loan_utilization'] / (df['operating_deposits'] + 1e-6)
    
    # Rolling metrics
    window = 30
    roll = df.groupby('client_id', group_keys=False).rolling(window, min_periods=15)
    
    df['rolling_ldr_mean'] = roll['ldr'].mean().reset_index(level=0, drop=True)
    df['rolling_deposit_vol'] = roll['operating_deposits'].std().reset_index(level=0, drop=True)
    df['deposit_trend'] = roll['operating_deposits'].apply(
        lambda x: np.polyfit(np.arange(len(x)), x, 1)[0]).reset_index(level=0, drop=True)
    
    return df

df = calculate_kpis(df)

# 3. Cohort Analysis with Time-Varying Clusters
def temporal_clustering(df, freq='M'):
    results = []
    dates = pd.date_range(df['date'].min(), df['date'].max(), freq=freq)
    
    for i in range(1, len(dates)):
        start_date = dates[i-1]
        end_date = dates[i]
        
        # Get window data
        window_df = df[(df['date'] > start_date) & (df['date'] <= end_date)]
        features = window_df.groupby('client_id').agg({
            'rolling_ldr_mean': 'mean',
            'rolling_deposit_vol': 'mean',
            'deposit_trend': 'mean',
            'segment': 'first'
        }).dropna()
        
        # Normalize features
        scaler = StandardScaler()
        X = scaler.fit_transform(features[['rolling_ldr_mean', 
                                         'rolling_deposit_vol',
                                         'deposit_trend']])
        
        # Optimal clusters using silhouette score
        best_score = -1
        best_k = 2
        for k in range(2, 5):
            kmeans = KMeans(n_clusters=k, random_state=42).fit(X)
            score = silhouette_score(X, kmeans.labels_)
            if score > best_score:
                best_score = score
                best_k = k
                
        # Final clustering
        kmeans = KMeans(n_clusters=best_k, random_state=42).fit(X)
        features['cluster'] = kmeans.labels_
        features['period'] = end_date.strftime('%Y-%m')
        results.append(features.reset_index())
    
    return pd.concat(results)

cluster_results = temporal_clustering(df)

# 4. Visualization
plt.figure(figsize=(15, 10))

# KPI Trends
plt.subplot(2, 2, 1)
sns.lineplot(data=df.groupby(['date', 'segment'])['ldr'].mean().reset_index(),
             x='date', y='ldr', hue='segment')
plt.title('Loan-to-Deposit Ratio Trend by Segment')
plt.xticks(rotation=45)

# Cluster Composition
plt.subplot(2, 2, 2)
cluster_dist = cluster_results.groupby(['period', 'cluster', 'segment']).size().unstack()
cluster_dist.plot(kind='area', stacked=True, colormap='tab20')
plt.title('Cluster Composition Over Time')
plt.xlabel('Period')

# Cluster Characteristics
plt.subplot(2, 2, 3)
latest = cluster_results[cluster_results['period'] == '2022-12']
sns.scatterplot(data=latest, x='rolling_ldr_mean', y='rolling_deposit_vol',
                hue='cluster', style='segment', palette='viridis', s=100)
plt.title('Cluster Characteristics (Latest Period)')

# Cluster Transitions Heatmap
plt.subplot(2, 2, 4)
transitions = cluster_results.pivot_table(index='client_id', columns='period', 
                                        values='cluster', aggfunc='first')
transitions = transitions.fillna(method='ffill', axis=1).dropna()
trans_counts = transitions.apply(lambda x: x.value_counts(normalize=True))
sns.heatmap(trans_counts.T, cmap='YlGnBu', annot=True, fmt='.0%')
plt.title('Cluster Distribution Over Time')

plt.tight_layout()
plt.show()

# 5. Liquidity Risk Analysis
def liquidity_risk_analysis(cluster_results):
    risk_profile = cluster_results.groupby('cluster').agg({
        'rolling_ldr_mean': 'mean',
        'rolling_deposit_vol': 'mean',
        'deposit_trend': 'mean'
    }).sort_values('rolling_ldr_mean', ascending=False)
    
    risk_profile['risk_score'] = (risk_profile['rolling_ldr_mean'].rank(ascending=False) +
                                 risk_profile['rolling_deposit_vol'].rank(ascending=False) +
                                 risk_profile['deposit_trend'].rank(ascending=True))
    
    return risk_profile

print("Liquidity Risk Profile:")
print(liquidity_risk_analysis(cluster_results))
