import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.seasonal import seasonal_decompose
import plotly.graph_objects as go
from datetime import datetime, timedelta

# Set random seed for reproducibility
np.random.seed(42)

# Step 1: Generate realistic sample data with seasonality
def generate_sample_data():
    dates = pd.date_range(start='2021-01-01', end='2024-12-31', freq='D')
    company_ids = [f'C{str(i).zfill(3)}' for i in range(1, 101)]
    
    data = []
    for company in company_ids:
        base_deposit = np.random.uniform(100000, 1000000)
        base_loan = np.random.uniform(50000, 500000)
        for i, date in enumerate(dates):
            seasonality = 1 + 0.2 * np.sin(2 * np.pi * i / 365)  # Yearly seasonality
            deposit = base_deposit * seasonality * np.random.uniform(0.5, 1.5) if np.random.random() > 0.2 else 0
            used_loan = base_loan * seasonality * np.random.uniform(0.5, 1.5) if np.random.random() > 0.1 else 0
            unused_loan = base_loan * 0.4 * np.random.uniform(0, 1) if np.random.random() > 0.1 else 0
            data.append([company, date, deposit, used_loan, unused_loan])
    
    df = pd.DataFrame(data, columns=['company_id', 'date', 'deposit', 'used_loan', 'unused_loan'])
    return df

# Step 2: Filter companies with >80% non-zero, non-NaN, non-inf values
def filter_valid_companies(df):
    def check_validity(group):
        total_days = len(group)
        threshold = total_days * 0.8
        valid_deposit = ((group['deposit'] > 0) & ~group['deposit'].isna() & ~np.isinf(group['deposit'])).sum()
        valid_used = ((group['used_loan'] > 0) & ~group['used_loan'].isna() & ~np.isinf(group['used_loan'])).sum()
        valid_unused = ((group['unused_loan'] > 0) & ~group['unused_loan'].isna() & ~np.isinf(group['unused_loan'])).sum()
        return (valid_deposit >= threshold) and (valid_used >= threshold) and (valid_unused >= threshold)
    
    return df.groupby('company_id').filter(check_validity)

# Step 3: Feature engineering
def engineer_features(df):
    df['loan_utilization'] = df['used_loan'] / (df['used_loan'] + df['unused_loan'] + 1e-6)  # Avoid division by zero
    df['total_loan'] = df['used_loan'] + df['unused_loan']
    
    # Lagged features and volatility
    for col in ['deposit', 'loan_utilization', 'used_loan']:
        df[f'{col}_lag7'] = df.groupby('company_id')[col].shift(7)
        df[f'{col}_lag30'] = df.groupby('company_id')[col].shift(30)
        df[f'{col}_volatility'] = df.groupby('company_id')[col].rolling(30, min_periods=1).std().reset_index(level=0, drop=True)
    
    # Rolling trends
    df['deposit_trend_3m'] = df.groupby('company_id')['deposit'].rolling(90, min_periods=1).mean().reset_index(level=0, drop=True)
    df['util_trend_3m'] = df.groupby('company_id')['loan_utilization'].rolling(90, min_periods=1).mean().reset_index(level=0, drop=True)
    
    # Seasonal decomposition (example for deposit)
    def decompose_series(group, col):
        result = seasonal_decompose(group[col].replace(0, np.nan).interpolate(), model='additive', period=365)
        return pd.Series(result.trend, index=group.index, name=f'{col}_trend')
    
    trends = df.groupby('company_id').apply(decompose_series, col='deposit')
    df = df.join(trends)
    
    return df.fillna(0)

# Step 4: Waterfall Chart
def create_waterfall_chart(df):
    total_clients = len(df['company_id'].unique())
    clients_with_loans = len(df[df['used_loan'] > 0]['company_id'].unique())
    clients_with_both = len(df[(df['used_loan'] > 0) & (df['deposit'] > 0)]['company_id'].unique())
    
    fig = go.Figure(go.Waterfall(
        name="Client Breakdown", orientation="v",
        measure=["absolute", "relative", "relative", "total"],
        x=["Total Clients", "Clients with Loans", "Clients with Both", "Final"],
        y=[total_clients, clients_with_loans - total_clients, clients_with_both - clients_with_loans, 0],
        connector={"line": {"color": "rgb(63, 63, 63)"}},
    ))
    fig.update_layout(title="Client Waterfall Chart", showlegend=False)
    fig.show()

# Step 5: Segment data into high, medium, low categories
def segment_data(df):
    for col in ['loan_utilization', 'used_loan', 'deposit', 'deposit_volatility']:
        quantiles = df[col].quantile([0.33, 0.66]).values
        df[f'{col}_segment'] = pd.cut(df[col], 
                                    bins=[-float('inf'), quantiles[0], quantiles[1], float('inf')],
                                    labels=['Low', 'Medium', 'High'])
    return df

# Step 6: Time-series correlation analysis
def flag_correlations(df):
    def calculate_correlations(group):
        if len(group) > 30:  # Minimum data points
            corr_util_deposit, _ = pearsonr(group['loan_utilization'], group['deposit'])
            corr_util_used, _ = pearsonr(group['loan_utilization'], group['used_loan'])
            corr_deposit_lag30, _ = pearsonr(group['deposit'], group['deposit_lag30'])
            return pd.Series({
                'corr_util_deposit': corr_util_deposit,
                'corr_util_used': corr_util_used,
                'corr_deposit_lag30': corr_deposit_lag30
            })
        return pd.Series({'corr_util_deposit': np.nan, 'corr_util_used': np.nan, 'corr_deposit_lag30': np.nan})
    
    corr_df = df.groupby('company_id').apply(calculate_correlations).reset_index()
    corr_df['high_corr_util_deposit'] = corr_df['corr_util_deposit'].apply(lambda x: abs(x) > 0.7)
    corr_df['high_anticorr_util_deposit'] = corr_df['corr_util_deposit'].apply(lambda x: x < -0.7)
    return corr_df

# Step 7: Enhanced risk flagging with heuristic rules
def flag_risks(df):
    risk_flags = []
    
    for company in df['company_id'].unique():
        company_data = df[df['company_id'] == company].sort_values('date')
        
        # Rolling averages and differences
        company_data['deposit_diff_3m'] = company_data['deposit_trend_3m'].diff(90)
        company_data['util_diff_3m'] = company_data['util_trend_3m'].diff(90)
        company_data['deposit_vol_6m'] = company_data['deposit'].rolling(180, min_periods=1).std()
        
        last_3m = company_data.iloc[-90:]
        last_6m = company_data.iloc[-180:]
        
        # Heuristic rules
        risk1 = (last_3m['util_diff_3m'].mean() > 0) and (last_3m['deposit_diff_3m'].mean() <= 0)  # Utilization up, deposit down
        risk2 = (last_6m['util_trend_3m'].std() < 0.01) and (last_6m['deposit'].diff().mean() < 0)  # Steady utilization, decreasing deposit
        risk3 = (last_3m['used_loan'].diff().mean() < 0) and \
                (abs(last_3m['deposit_diff_3m'].mean()) > abs(last_3m['used_loan'].diff().mean()) * 1.5)  # Loan down, deposit drops faster
        risk4 = (last_6m['deposit_vol_6m'].iloc[-1] > last_6m['deposit_vol_6m'].quantile(0.9)) and \
                (last_6m['loan_utilization'].diff().mean() > 0)  # High volatility in deposit, increasing utilization
        risk5 = (last_3m['deposit_lag30'] / last_3m['deposit']).mean() > 1.2  # Significant drop from 30 days ago
        
        risk_flags.append([company, risk1 or risk2 or risk3 or risk4 or risk5, 
                          {'risk1': risk1, 'risk2': risk2, 'risk3': risk3, 'risk4': risk4, 'risk5': risk5}])
    
    risk_df = pd.DataFrame(risk_flags, columns=['company_id', 'risk_flag', 'risk_details'])
    return risk_df

# Step 8: Advanced clustering with time-varying behavior
def cluster_clients(df):
    features = df.groupby('company_id').agg({
        'loan_utilization': ['mean', 'std', 'skew'],
        'deposit': ['mean', 'std', 'skew'],
        'used_loan': ['mean', 'std'],
        'deposit_volatility': 'mean',
        'deposit_trend': 'mean'
    }).fillna(0)
    features.columns = ['_'.join(col) for col in features.columns]
    
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)
    
    kmeans = KMeans(n_clusters=5, random_state=42)
    features['cluster'] = kmeans.fit_predict(scaled_features)
    
    return features.reset_index()

# Step 9: Visualizations
def create_visualizations(df, corr_df, risk_df, cluster_df):
    # Correlation heatmap
    plt.figure(figsize=(12, 6))
    sns.heatmap(corr_df[['corr_util_deposit', 'corr_util_used', 'corr_deposit_lag30']].fillna(0),
                cmap='coolwarm', center=0, annot=True)
    plt.title('Time-Series Correlations by Company')
    plt.show()
    
    # Cluster visualization
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=cluster_df, x='loan_utilization_mean', y='deposit_mean',
                   hue='cluster', size='used_loan_mean', palette='deep')
    plt.title('Client Clusters Based on Engineered Features')
    plt.show()
    
    # Risk flag distribution with details
    plt.figure(figsize=(10, 5))
    risk_counts = pd.DataFrame([d for d in risk_df['risk_details']]).sum()
    risk_counts.plot(kind='bar')
    plt.title('Distribution of Risk Types')
    plt.show()
    
    # Time-series example for a risky company
    risky_company = risk_df[risk_df['risk_flag']].iloc[0]['company_id']
    company_data = df[df['company_id'] == risky_company]
    plt.figure(figsize=(12, 6))
    plt.plot(company_data['date'], company_data['loan_utilization'], label='Loan Utilization')
    plt.plot(company_data['date'], company_data['deposit'] / company_data['deposit'].max(), label='Normalized Deposit')
    plt.title(f'Time-Series for Risky Company {risky_company}')
    plt.legend()
    plt.show()

# Main execution
def main():
    # Generate and process data
    df = generate_sample_data()
    df = filter_valid_companies(df)
    df = engineer_features(df)
    create_waterfall_chart(df)
    df = segment_data(df)
    
    # Analysis
    corr_df = flag_correlations(df)
    risk_df = flag_risks(df)
    cluster_df = cluster_clients(df)
    
    # Merge results
    results = df.merge(corr_df, on='company_id').merge(risk_df, on='company_id').merge(
        cluster_df[['company_id', 'cluster']], on='company_id')
    
    # Visualizations
    create_visualizations(df, corr_df, risk_df, cluster_df)
    
    print("Sample of final results:")
    print(results[['company_id', 'date', 'loan_utilization', 'deposit', 'risk_flag', 'cluster']].head())
    print(f"Total companies after filtering: {len(results['company_id'].unique())}")

if __name__ == "__main__":
    main()
