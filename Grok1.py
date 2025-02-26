import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr
from sklearn.cluster import KMeans
import plotly.graph_objects as go
from datetime import datetime, timedelta

# Set random seed for reproducibility
np.random.seed(42)

# Step 1: Generate realistic sample data
def generate_sample_data():
    dates = pd.date_range(start='2021-01-01', end='2024-12-31', freq='D')
    company_ids = [f'C{str(i).zfill(3)}' for i in range(1, 101)]
    
    data = []
    for company in company_ids:
        for date in dates:
            deposit = np.random.uniform(0, 1000000) if np.random.random() > 0.2 else 0
            used_loan = np.random.uniform(0, 500000) if np.random.random() > 0.1 else 0
            unused_loan = np.random.uniform(0, 200000) if np.random.random() > 0.1 else 0
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
    
    valid_companies = df.groupby('company_id').filter(check_validity)
    return valid_companies

# Step 3: Create Waterfall Chart
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

# Step 4: Segment data into high, medium, low categories
def segment_data(df):
    df['loan_utilization'] = df['used_loan'] / (df['used_loan'] + df['unused_loan'])
    
    for col in ['loan_utilization', 'used_loan', 'deposit']:
        quantiles = df[col].quantile([0.33, 0.66]).values
        df[f'{col}_segment'] = pd.cut(df[col], 
                                    bins=[-float('inf'), quantiles[0], quantiles[1], float('inf')],
                                    labels=['Low', 'Medium', 'High'])
    return df

# Step 5: Correlation analysis
def flag_correlations(df):
    def calculate_correlation(group):
        if len(group) > 10:  # Minimum data points for meaningful correlation
            corr, _ = pearsonr(group['loan_utilization'], group['deposit'])
            return pd.Series({'correlation': corr})
        return pd.Series({'correlation': np.nan})
    
    corr_df = df.groupby('company_id').apply(calculate_correlation).reset_index()
    corr_df['high_correlation'] = corr_df['correlation'].apply(lambda x: abs(x) > 0.7)
    return corr_df

# Step 6: Risk flagging based on patterns
def flag_risks(df):
    risk_flags = []
    
    for company in df['company_id'].unique():
        company_data = df[df['company_id'] == company].sort_values('date')
        
        # Rolling averages for last 3 and 6 months
        company_data['deposit_3m'] = company_data['deposit'].rolling(90, min_periods=1).mean()
        company_data['deposit_6m'] = company_data['deposit'].rolling(180, min_periods=1).mean()
        company_data['util_3m'] = company_data['loan_utilization'].rolling(90, min_periods=1).mean()
        
        # Risk conditions
        last_3m = company_data.iloc[-90:]
        last_6m = company_data.iloc[-180:]
        
        risk1 = (last_3m['util_3m'].iloc[-1] > last_3m['util_3m'].iloc[0]) and \
                (last_3m['deposit_3m'].iloc[-1] <= last_3m['deposit_3m'].iloc[0])
        risk2 = (last_6m['util_3m'].diff().mean() < 0.001) and \
                (last_6m['deposit_6m'].diff().mean() < 0)
        risk3 = (last_3m['used_loan'].diff().mean() < 0) and \
                (last_3m['deposit'].diff().mean() < last_3m['used_loan'].diff().mean() * 2)
        
        risk_flags.append([company, risk1 or risk2 or risk3])
    
    return pd.DataFrame(risk_flags, columns=['company_id', 'risk_flag'])

# Step 7: Clustering and cohort analysis
def cluster_clients(df):
    features = df.groupby('company_id').agg({
        'loan_utilization': ['mean', 'std'],
        'deposit': ['mean', 'std'],
        'used_loan': ['mean', 'std']
    }).fillna(0)
    features.columns = ['_'.join(col) for col in features.columns]
    
    kmeans = KMeans(n_clusters=4, random_state=42)
    features['cluster'] = kmeans.fit_predict(features)
    
    return features.reset_index()

# Step 8: Visualizations
def create_visualizations(df, corr_df, risk_df, cluster_df):
    # Correlation heatmap
    plt.figure(figsize=(10, 6))
    sns.heatmap(corr_df.pivot(columns='high_correlation', values='correlation').fillna(0),
                cmap='coolwarm', center=0)
    plt.title('Correlation Heatmap')
    plt.show()
    
    # Cluster visualization
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=cluster_df, x='loan_utilization_mean', y='deposit_mean',
                   hue='cluster', size='used_loan_mean')
    plt.title('Client Clusters')
    plt.show()
    
    # Risk flag distribution
    plt.figure(figsize=(8, 5))
    risk_df['risk_flag'].value_counts().plot(kind='bar')
    plt.title('Risk Flag Distribution')
    plt.show()

# Main execution
def main():
    # Generate and process data
    df = generate_sample_data()
    df = filter_valid_companies(df)
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
    print(results.head())
    print(f"Total companies after filtering: {len(results['company_id'].unique())}")

if __name__ == "__main__":
    main()
