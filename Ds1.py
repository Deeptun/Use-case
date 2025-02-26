import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy.stats import pearsonr, linregress
from datetime import datetime, timedelta

# Set random seed for reproducibility
np.random.seed(42)

# 1. Generate realistic synthetic data
def generate_data(n_companies=100, start_date='2018-01-01', end_date='2021-12-31'):
    dates = pd.date_range(start_date, end_date)
    company_ids = [f'CMP{str(i).zfill(3)}' for i in range(n_companies)]
    
    data = []
    for cmp in company_ids:
        for date in dates:
            # Base values with some zeros and NaNs
            deposit = np.random.choice([np.random.exponential(1000), 0, np.nan], p=[0.7, 0.2, 0.1])
            used = np.random.gamma(shape=2, scale=500) if np.random.rand() > 0.3 else 0
            unused = np.random.gamma(shape=3, scale=300) if np.random.rand() > 0.25 else 0
            
            # Add some infinite values occasionally
            if np.random.rand() < 0.05:
                deposit = np.inf
                
            data.append({
                'company_id': cmp,
                'date': date,
                'deposit': deposit,
                'used_loan': used,
                'unused_loan': unused
            })
    
    df = pd.DataFrame(data)
    return df

# 2. Data cleaning function
def clean_data(df):
    def is_valid(x):
        return np.isfinite(x) & (x != 0)
    
    # Calculate valid percentage for each company
    valid_stats = df.groupby('company_id').agg({
        'deposit': lambda x: is_valid(x).mean(),
        'used_loan': lambda x: is_valid(x).mean(),
        'unused_loan': lambda x: is_valid(x).mean()
    })
    
    # Filter companies with >=80% valid in all columns
    valid_companies = valid_stats[(valid_stats >= 0.8).all(axis=1)].index
    return df[df.company_id.isin(valid_companies)].copy()

# 3. Loan utilization calculation
def calculate_utilization(df):
    df = df.copy()
    df['loan_utilization'] = df['used_loan'] / (df['used_loan'] + df['unused_loan'].replace(0, np.nan))
    return df

# 4. Segmentation functions
def create_segments(df):
    df = df.copy()
    # Calculate metrics per company
    company_stats = df.groupby('company_id').agg({
        'loan_utilization': 'mean',
        'used_loan': 'mean',
        'deposit': 'mean'
    }).reset_index()
    
    # Create segments using tertiles
    for col in ['loan_utilization', 'used_loan', 'deposit']:
        tertiles = company_stats[col].quantile([0.33, 0.66])
        company_stats[f'{col}_segment'] = pd.cut(company_stats[col],
                                               bins=[-np.inf, tertiles[0.33], tertiles[0.66], np.inf],
                                               labels=['low', 'medium', 'high'])
    
    return company_stats

# 5. Correlation analysis
def flag_correlations(df):
    corr_flags = []
    for cmp, group in df.groupby('company_id'):
        clean_group = group.dropna(subset=['loan_utilization', 'deposit'])
        if len(clean_group) < 10:  # Minimum data points
            continue
            
        corr = pearsonr(clean_group['loan_utilization'], clean_group['deposit'])[0]
        if corr > 0.5:
            corr_flags.append((cmp, 'high_positive'))
        elif corr < -0.5:
            corr_flags.append((cmp, 'high_negative'))
    
    return pd.DataFrame(corr_flags, columns=['company_id', 'correlation_flag'])

# 6. Risk flagging
def calculate_trends(group, months=3):
    current_date = group['date'].max()
    cutoff = current_date - pd.DateOffset(months=months)
    recent_data = group[group['date'] > cutoff]
    
    if len(recent_data) < 10:
        return pd.Series([np.nan]*4)
    
    # Calculate slopes
    x = np.arange(len(recent_data))
    util_slope = linregress(x, recent_data['loan_utilization'].fillna(0))[0]
    deposit_slope = linregress(x, recent_data['deposit'].fillna(0))[0]
    
    # Calculate percentage changes
    util_change = recent_data['loan_utilization'].pct_change().mean()
    deposit_change = recent_data['deposit'].pct_change().mean()
    
    return pd.Series([util_slope, deposit_slope, util_change, deposit_change])

def flag_risks(df):
    risk_flags = []
    for cmp, group in df.groupby('company_id'):
        group = group.sort_values('date')
        trends_3m = calculate_trends(group, 3)
        trends_6m = calculate_trends(group, 6)
        
        flags = []
        # Rule 1: Utilization up & deposit down
        if trends_3m[0] > 0 and trends_3m[1] < 0:
            flags.append('util_up_deposit_down_3m')
        if trends_6m[0] > 0 and trends_6m[1] < 0:
            flags.append('util_up_deposit_down_6m')
            
        # Rule 2: Utilization steady & deposit slowly decreasing
        if abs(trends_6m[0]) < 0.01 and trends_6m[1] < -0.005:
            flags.append('util_steady_deposit_decreasing')
            
        # Rule 3: Loan decreasing but deposits diminishing faster
        if (trends_6m[2] < 0) and (trends_6m[3]/trends_6m[2] > 1.5):
            flags.append('deposit_decreasing_faster')
            
        if flags:
            risk_flags.append((cmp, '|'.join(flags)))
    
    return pd.DataFrame(risk_flags, columns=['company_id', 'risk_flags'])

# 7. Clustering and cohort analysis
def perform_clustering(df):
    # Create temporal features
    features = []
    for cmp, group in df.groupby('company_id'):
        group = group.sort_values('date')
        features.append({
            'company_id': cmp,
            'avg_utilization': group['loan_utilization'].mean(),
            'util_trend': linregress(np.arange(len(group)), group['loan_utilization'].fillna(0))[0],
            'deposit_trend': linregress(np.arange(len(group)), group['deposit'].fillna(0))[0],
            'util_volatility': group['loan_utilization'].std(),
            'seasonality': group.groupby(group['date'].dt.month)['loan_utilization'].mean().std()
        })
    
    feature_df = pd.DataFrame(features).set_index('company_id').dropna()
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(feature_df)
    
    # Perform K-means clustering
    kmeans = KMeans(n_clusters=4, random_state=42)
    feature_df['cluster'] = kmeans.fit_predict(scaled_features)
    
    return feature_df

# 8. Visualization functions
def plot_waterfall(original_count, filtered_count):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x=['Initial', 'With Deposits'], height=[original_count, filtered_count], color=['#1f77b4', '#2ca02c'])
    ax.set_title('Customer Waterfall Chart')
    ax.set_ylabel('Number of Companies')
    plt.show()

def plot_clusters(feature_df):
    pca = PCA(n_components=2)
    reduced = pca.fit_transform(feature_df.drop('cluster', axis=1))
    
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=reduced[:,0], y=reduced[:,1], hue=feature_df['cluster'], palette='viridis')
    plt.title('Client Clusters in PCA Space')
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.show()

# Main analysis pipeline
def main():
    # Generate and clean data
    print("Generating data...")
    df = generate_data(n_companies=100)
    print(f"Original data shape: {df.shape}")
    
    cleaned_df = clean_data(df)
    print(f"Cleaned data shape: {cleaned_df.shape}")
    
    # Calculate loan utilization
    cleaned_df = calculate_utilization(cleaned_df)
    
    # Waterfall chart
    original_companies = df['company_id'].nunique()
    filtered_companies = cleaned_df['company_id'].nunique()
    plot_waterfall(original_companies, filtered_companies)
    
    # Segmentation analysis
    segments_df = create_segments(cleaned_df)
    print("\nSegment distribution:")
    print(segments_df[['loan_utilization_segment', 'used_loan_segment', 'deposit_segment']].apply(pd.Series.value_counts))
    
    # Correlation flags
    correlation_flags = flag_correlations(cleaned_df)
    print(f"\nCompanies with significant correlations: {len(correlation_flags)}")
    
    # Risk flags
    risk_flags = flag_risks(cleaned_df)
    print(f"\nCompanies with risk flags: {len(risk_flags)}")
    
    # Clustering
    cluster_df = perform_clustering(cleaned_df)
    plot_clusters(cluster_df)
    
    # Cohort analysis over time
    print("\nCohort analysis (cluster distribution):")
    cleaned_df['year_quarter'] = cleaned_df['date'].dt.to_period('Q')
    cohort_data = cleaned_df.merge(cluster_df.reset_index()[['company_id', 'cluster']], on='company_id')
    cohort_dist = cohort_data.groupby(['year_quarter', 'cluster']).size().unstack().fillna(0)
    cohort_dist.plot(kind='area', stacked=True, figsize=(12, 6))
    plt.title('Cluster Distribution Over Time')
    plt.ylabel('Number of Companies')
    plt.show()

if __name__ == '__main__':
    main()
