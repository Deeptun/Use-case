"""
Robust Financial Behavior Analysis Pipeline with Advanced Features
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from scipy.stats import pearsonr, linregress
from datetime import datetime, timedelta
import warnings
from tqdm import tqdm
import logging

# Configuration
CONFIG = {
    'data': {
        'n_companies': 500,
        'start_date': '2018-01-01',
        'end_date': '2021-12-31',
        'valid_threshold': 0.8,
        'min_non_zero': 0.1
    },
    'segments': {
        'quantiles': [0.2, 0.4, 0.6, 0.8],
        'labels': ['very_low', 'low', 'medium', 'high', 'very_high']
    },
    'correlation': {
        'min_samples': 30,
        'threshold': 0.6,
        'window_size': 90  # days for rolling correlation
    },
    'risk': {
        'trend_windows': [30, 90, 180],  # days
        'change_thresholds': {
            'sharp': 0.5,
            'moderate': 0.3,
            'gradual': 0.1
        },
        'risk_rules': {
            'util_increase_deposit_decrease': 0.3,
            'diverging_trends': 0.5,
            'accelerating_drawdown': 1.2
        }
    },
    'clustering': {
        'n_clusters': 5,
        'random_state': 42,
        'feature_components': ['trend', 'seasonality', 'volatility', 'correlation']
    }
}

# Configure logging
logging.basicConfig(level=logging.INFO)
warnings.filterwarnings('ignore')

# --------------------------
# Data Generation & Cleaning
# --------------------------

def generate_company_ts(date_range, company_id):
    """Generate realistic time series with trends and seasonality"""
    n_days = len(date_range)
    base_date = pd.to_datetime('2018-01-01')
    
    # Base patterns
    time_index = (date_range - base_date).days.values
    seasonal = np.sin(2 * np.pi * time_index / 365) * np.random.normal(0.5, 0.2)
    trend = np.linspace(0, np.random.normal(0, 0.1), n_days)
    noise = np.random.normal(0, 0.1, n_days)
    
    # Deposit pattern
    deposit = np.exp(seasonal + trend + noise) * np.random.lognormal(4, 0.5)
    
    # Loan patterns
    loan_utilization = np.clip(0.3 + 0.4 * np.tanh(trend * 2), 0, 0.95)
    total_loan = np.random.lognormal(10, 0.5) * (1 + 0.1 * seasonal)
    used_loan = total_loan * loan_utilization
    unused_loan = total_loan - used_loan
    
    # Add missing values and zeros
    mask = np.random.choice([True, False], size=n_days, p=[0.1, 0.9])
    deposit[mask] = np.nan if np.random.rand() > 0.3 else 0
    
    return pd.DataFrame({
        'company_id': company_id,
        'date': date_range,
        'deposit': deposit,
        'used_loan': used_loan,
        'unused_loan': unused_loan
    })

def generate_data():
    """Generate synthetic dataset with realistic patterns"""
    dates = pd.date_range(CONFIG['data']['start_date'], 
                         CONFIG['data']['end_date'])
    company_ids = [f'CMP{str(i).zfill(4)}' for i in range(CONFIG['data']['n_companies'])]
    
    dfs = []
    for cmp in tqdm(company_ids, desc='Generating data'):
        df = generate_company_ts(dates, cmp)
        dfs.append(df)
    
    return pd.concat(dfs).reset_index(drop=True)

def clean_data(df):
    """Robust data cleaning with detailed validation"""
    def valid_ratio(s):
        return ((s.notna() & np.isfinite(s) & (s > CONFIG['data']['min_non_zero'])).mean())
    
    valid_stats = df.groupby('company_id').agg({
        'deposit': valid_ratio,
        'used_loan': valid_ratio,
        'unused_loan': valid_ratio
    })
    
    valid_mask = (valid_stats >= CONFIG['data']['valid_threshold']).all(axis=1)
    valid_companies = valid_stats[valid_mask].index
    
    logging.info(f"Data cleaning: Retained {len(valid_companies)}/{len(valid_stats)} companies")
    return df[df.company_id.isin(valid_companies)].copy()

# --------------------------
# Feature Engineering
# --------------------------

def calculate_features(df):
    """Calculate financial metrics with error handling"""
    df = df.copy()
    
    with np.errstate(divide='ignore', invalid='ignore'):
        df['loan_utilization'] = df['used_loan'] / (df['used_loan'] + df['unused_loan'].replace(0, np.nan))
        df['deposit_loan_ratio'] = df['deposit'] / (df['used_loan'] + 1e-6)
    
    # Rolling features
    df.sort_values('date', inplace=True)
    for window in [7, 30, 90]:
        df[f'util_ma_{window}'] = df.groupby('company_id')['loan_utilization']\
                                   .transform(lambda x: x.rolling(window, min_periods=5).mean())
        df[f'deposit_ma_{window}'] = df.groupby('company_id')['deposit']\
                                      .transform(lambda x: x.rolling(window, min_periods=5).mean())
    
    return df

# --------------------------
# Segmentation & Correlation
# --------------------------

def create_segments(df):
    """Dynamic quantile-based segmentation"""
    company_stats = df.groupby('company_id').agg({
        'loan_utilization': 'median',
        'used_loan': 'median',
        'deposit': 'median',
        'deposit_loan_ratio': 'median'
    }).reset_index()
    
    for col in ['loan_utilization', 'used_loan', 'deposit', 'deposit_loan_ratio']:
        quantiles = company_stats[col].quantile(CONFIG['segments']['quantiles']).values
        bins = [-np.inf] + quantiles.tolist() + [np.inf]
        company_stats[f'{col}_segment'] = pd.cut(company_stats[col], bins=bins, 
                                                labels=CONFIG['segments']['labels'])
    
    return company_stats

def calculate_correlations(df):
    """Rolling window correlation analysis"""
    corr_results = []
    
    for cmp, group in tqdm(df.groupby('company_id'), desc='Calculating correlations'):
        group = group.sort_values('date').dropna(subset=['loan_utilization', 'deposit'])
        if len(group) < CONFIG['correlation']['min_samples']:
            continue
            
        # Rolling correlation
        rolling_corr = group['loan_utilization'].rolling(
            window=CONFIG['correlation']['window_size'],
            min_periods=30
        ).corr(group['deposit'])
        
        # Significance testing
        max_corr = rolling_corr.max()
        min_corr = rolling_corr.min()
        
        if max_corr > CONFIG['correlation']['threshold']:
            corr_results.append((cmp, 'positive', max_corr))
        if min_corr < -CONFIG['correlation']['threshold']:
            corr_results.append((cmp, 'negative', min_corr))
    
    return pd.DataFrame(corr_results, columns=['company_id', 'correlation_type', 'strength'])

# --------------------------
# Risk Analysis
# --------------------------

def analyze_trends(series, windows):
    """Advanced trend analysis with multiple windows"""
    results = {}
    x = np.arange(len(series))
    
    for w in windows:
        if len(series) < w:
            continue
            
        # Slopes
        slope = linregress(x[-w:], series[-w:])[0]
        
        # Percentage changes
        pct_change = (series.iloc[-1] - series.iloc[-w]) / series.iloc[-w]
        
        # Volatility
        volatility = series[-w:].std()
        
        results[f'trend_{w}'] = slope
        results[f'change_{w}'] = pct_change
        results[f'volatility_{w}'] = volatility
    
    return pd.Series(results)

def detect_risks(df):
    """Comprehensive risk detection system"""
    risk_records = []
    
    for cmp, group in tqdm(df.groupby('company_id'), desc='Risk analysis'):
        group = group.sort_values('date').dropna(subset=['loan_utilization', 'deposit'])
        if len(group) < 30:
            continue
            
        # Calculate trends
        util_trends = analyze_trends(group['loan_utilization'], CONFIG['risk']['trend_windows'])
        deposit_trends = analyze_trends(group['deposit'], CONFIG['risk']['trend_windows'])
        
        # Risk rules
        flags = []
        
        # Rule 1: Diverging trends
        for w in CONFIG['risk']['trend_windows']:
            if f'trend_{w}' in util_trends and f'trend_{w}' in deposit_trends:
                trend_ratio = util_trends[f'trend_{w}'] / (deposit_trends[f'trend_{w}'] + 1e-6)
                if abs(trend_ratio) > CONFIG['risk']['risk_rules']['diverging_trends']:
                    flags.append(f'diverging_trends_{w}d')
        
        # Rule 2: Accelerating drawdown
        for w in [30, 90]:
            if f'change_{w}' in deposit_trends and f'change_{w}' in util_trends:
                if (deposit_trends[f'change_{w}'] < -0.3 and 
                    util_trends[f'change_{w}'] > 0.2):
                    flags.append(f'accelerating_drawdown_{w}d')
        
        # Rule 3: Volatility spikes
        if 'volatility_30' in deposit_trends and 'volatility_90' in deposit_trends:
            if deposit_trends['volatility_30'] > 1.5 * deposit_trends['volatility_90']:
                flags.append('deposit_volatility_spike')
        
        if flags:
            risk_records.append({
                'company_id': cmp,
                'risk_flags': '|'.join(flags),
                'last_date': group['date'].max(),
                'util_trends': util_trends.to_dict(),
                'deposit_trends': deposit_trends.to_dict()
            })
    
    return pd.DataFrame(risk_records)

# --------------------------
# Clustering & Visualization
# --------------------------

def create_cluster_features(df):
    """Feature engineering for clustering"""
    features = []
    
    for cmp, group in tqdm(df.groupby('company_id'), desc='Creating features'):
        group = group.sort_values('date').dropna()
        
        # Time series features
        features.append({
            'company_id': cmp,
            'util_trend': linregress(np.arange(len(group)), group['loan_utilization'])[0],
            'deposit_trend': linregress(np.arange(len(group)), group['deposit'])[0],
            'util_seasonality': group.groupby(group['date'].dt.month)['loan_utilization'].mean().std(),
            'deposit_volatility': group['deposit'].pct_change().std(),
            'correlation': group['loan_utilization'].corr(group['deposit'])
        })
    
    return pd.DataFrame(features).set_index('company_id')

def perform_clustering(feature_df):
    """Robust clustering pipeline"""
    # Preprocessing
    imputer = SimpleImputer(strategy='median')
    scaler = RobustScaler()
    processed = scaler.fit_transform(imputer.fit_transform(feature_df))
    
    # Dimensionality reduction
    pca = PCA(n_components=0.95)
    reduced = pca.fit_transform(processed)
    
    # Clustering
    kmeans = KMeans(n_clusters=CONFIG['clustering']['n_clusters'], 
                   random_state=CONFIG['clustering']['random_state'])
    clusters = kmeans.fit_predict(reduced)
    
    # Results
    result_df = feature_df.copy()
    result_df['cluster'] = clusters
    result_df['pca1'] = reduced[:,0]
    result_df['pca2'] = reduced[:,1]
    
    return result_df

def visualize_clusters(cluster_df):
    """Interactive cluster visualization"""
    plt.figure(figsize=(12, 8))
    sns.scatterplot(
        x='pca1',
        y='pca2',
        hue='cluster',
        data=cluster_df,
        palette='viridis',
        s=100,
        alpha=0.7
    )
    plt.title('Client Clusters in PCA Space')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.show()

# --------------------------
# Main Pipeline
# --------------------------

def main():
    """Main analysis pipeline"""
    # Data pipeline
    logging.info("Starting data generation...")
    df = generate_data()
    df_clean = clean_data(df)
    df_features = calculate_features(df_clean)
    
    # Segmentation
    segments = create_segments(df_features)
    
    # Correlation analysis
    correlations = calculate_correlations(df_features)
    
    # Risk analysis
    risks = detect_risks(df_features)
    
    # Clustering
    cluster_features = create_cluster_features(df_features)
    cluster_results = perform_clustering(cluster_features)
    visualize_clusters(cluster_results)
    
    # Cohort analysis
    merged_data = df_features.merge(
        cluster_results.reset_index()[['company_id', 'cluster']],
        on='company_id'
    )
    
    cohort = merged_data.groupby([
        pd.Grouper(key='date', freq='Q'),
        'cluster'
    ]).size().unstack().fillna(0)
    
    cohort.plot(kind='area', stacked=True, figsize=(14, 8))
    plt.title('Cluster Distribution Over Time')
    plt.ylabel('Number of Companies')
    plt.show()

if __name__ == '__main__':
    main()
