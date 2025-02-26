import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import random
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import plotly.graph_objects as go
import plotly.express as px
from statsmodels.tsa.seasonal import seasonal_decompose
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)

def generate_synthetic_data(num_companies=100, years=4):
    """
    Generate synthetic financial data for companies over a specified time period.
    
    Parameters:
    -----------
    num_companies : int
        Number of companies to generate data for
    years : int
        Number of years of daily data to generate
    
    Returns:
    --------
    pandas.DataFrame
        DataFrame containing synthetic financial data
    """
    # Define date range (daily for 4 years)
    start_date = datetime(2020, 1, 1)
    end_date = datetime(2023, 12, 31)
    dates = pd.date_range(start=start_date, end=end_date, freq='D')
    
    # Create company IDs
    company_ids = [f'COMP_{i:03d}' for i in range(1, num_companies + 1)]
    
    # Initialize empty lists to store data
    data_rows = []
    
    # Generate data for each company
    for company_id in company_ids:
        # Determine if company has deposit data (70% chance)
        has_deposit = random.random() < 0.7
        
        # Generate baseline values with some randomness
        baseline_used_loan = np.random.uniform(100000, 5000000)
        baseline_unused_loan = np.random.uniform(50000, 3000000)
        baseline_deposit = np.random.uniform(50000, 10000000) if has_deposit else 0
        
        # Add seasonal patterns (quarterly)
        seasons = np.sin(np.linspace(0, 8 * np.pi, len(dates)))
        
        # Add trend components (some growing, some declining)
        trend_factor = np.random.uniform(-0.3, 0.3)
        trend = np.linspace(0, trend_factor, len(dates))
        
        # Generate values for each date
        for i, date in enumerate(dates):
            # Add seasonality and trend
            seasonal_factor = seasons[i] * 0.2
            trend_value = trend[i]
            
            # Random day-to-day variation
            daily_variation = np.random.normal(0, 0.05)
            
            # Calculate the financial values with seasonality, trend, and random variation
            used_loan = baseline_used_loan * (1 + seasonal_factor + trend_value + daily_variation)
            unused_loan = baseline_unused_loan * (1 + seasonal_factor * 0.5 + trend_value * 0.8 + daily_variation)
            
            # For some companies, create patterns for risk flags
            if random.random() < 0.2:  # 20% of companies show risk patterns
                if i > 180:  # After ~6 months
                    # Pattern: loan utilization going up but deposit balance going down
                    used_loan *= (1 + 0.001 * (i - 180))
                    if has_deposit:
                        baseline_deposit *= (1 - 0.001 * (i - 180))
            
            # Calculate deposit with its own pattern
            if has_deposit:
                deposit_trend = np.random.uniform(-0.4, 0.4)
                deposit_trend_value = np.linspace(0, deposit_trend, len(dates))[i]
                deposit_seasonal_factor = seasons[i] * 0.15
                deposit_variation = np.random.normal(0, 0.08)
                deposit = baseline_deposit * (1 + deposit_seasonal_factor + deposit_trend_value + deposit_variation)
            else:
                deposit = 0
            
            # Sometimes introduce zeros and NaNs
            if random.random() < 0.1:  # 10% chance of special values
                choice = random.random()
                if choice < 0.6:
                    used_loan = 0
                elif choice < 0.8:
                    deposit = 0 if has_deposit else 0
                elif choice < 0.95:
                    unused_loan = 0
                else:
                    used_loan = np.nan
            
            # Ensure non-negative values
            used_loan = max(0, used_loan)
            unused_loan = max(0, unused_loan)
            deposit = max(0, deposit)
            
            # Create data row
            data_rows.append({
                'company_id': company_id,
                'date': date,
                'used_loan': used_loan,
                'unused_loan': unused_loan,
                'deposit': deposit
            })
    
    # Create DataFrame
    df = pd.DataFrame(data_rows)
    
    # Calculate loan utilization
    df['loan_utilization'] = df['used_loan'] / (df['used_loan'] + df['unused_loan'])
    df['loan_utilization'] = df['loan_utilization'].replace([np.inf, -np.inf], np.nan)
    
    return df

def clean_data(df):
    """
    Clean the data by keeping only companies where >80% of deposit, used loan, 
    and unused loan values are non-zero, non-NaN, and non-infinite.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Input DataFrame with financial data
    
    Returns:
    --------
    pandas.DataFrame
        Cleaned DataFrame with only valid companies
    """
    # Calculate the percentage of valid entries for each company
    company_stats = {}
    
    for company_id in df['company_id'].unique():
        company_data = df[df['company_id'] == company_id]
        total_rows = len(company_data)
        
        # Check deposit, used_loan, and unused_loan columns
        valid_deposit = company_data['deposit'].notna() & (company_data['deposit'] != 0) & ~np.isinf(company_data['deposit'])
        valid_used = company_data['used_loan'].notna() & (company_data['used_loan'] != 0) & ~np.isinf(company_data['used_loan'])
        valid_unused = company_data['unused_loan'].notna() & (company_data['unused_loan'] != 0) & ~np.isinf(company_data['unused_loan'])
        
        # Calculate percentage of valid entries for each column
        pct_valid_deposit = valid_deposit.mean() * 100
        pct_valid_used = valid_used.mean() * 100
        pct_valid_unused = valid_unused.mean() * 100
        
        company_stats[company_id] = {
            'pct_valid_deposit': pct_valid_deposit,
            'pct_valid_used': pct_valid_used,
            'pct_valid_unused': pct_valid_unused
        }
    
    # Filter companies where all metrics have at least 80% valid data
    valid_companies = [
        company_id for company_id, stats in company_stats.items()
        if stats['pct_valid_used'] >= 80 and stats['pct_valid_unused'] >= 80 and stats['pct_valid_deposit'] >= 80
    ]
    
    # Filter data to keep only valid companies
    cleaned_df = df[df['company_id'].isin(valid_companies)].copy()
    
    print(f"Original data had {df['company_id'].nunique()} companies.")
    print(f"After cleaning, {len(valid_companies)} companies remain.")
    
    return cleaned_df

def segment_companies(df):
    """
    Segment companies into high, medium, and low groups based on loan utilization,
    used loan amount, and deposit amount.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame with financial data
    
    Returns:
    --------
    tuple
        (DataFrame with segment information, aggregated company data)
    """
    # Aggregate data at company level
    company_agg = df.groupby('company_id').agg({
        'loan_utilization': 'mean',
        'used_loan': 'mean',
        'deposit': 'mean'
    }).reset_index()
    
    # Create segments for loan utilization
    utilization_thresholds = company_agg['loan_utilization'].quantile([0.33, 0.67]).tolist()
    company_agg['utilization_segment'] = pd.cut(
        company_agg['loan_utilization'],
        bins=[-float('inf')] + utilization_thresholds + [float('inf')],
        labels=['Low', 'Medium', 'High']
    )
    
    # Create segments for used loan amount
    used_loan_thresholds = company_agg['used_loan'].quantile([0.33, 0.67]).tolist()
    company_agg['used_loan_segment'] = pd.cut(
        company_agg['used_loan'],
        bins=[-float('inf')] + used_loan_thresholds + [float('inf')],
        labels=['Low', 'Medium', 'High']
    )
    
    # Create segments for deposit
    deposit_thresholds = company_agg[company_agg['deposit'] > 0]['deposit'].quantile([0.33, 0.67]).tolist()
    company_agg['deposit_segment'] = pd.cut(
        company_agg['deposit'],
        bins=[-float('inf')] + deposit_thresholds + [float('inf')],
        labels=['Low', 'Medium', 'High']
    )
    
    # Merge segments back to the original data
    df = df.merge(company_agg[['company_id', 'utilization_segment', 'used_loan_segment', 'deposit_segment']], 
                  on='company_id')
    
    return df, company_agg

def create_waterfall_chart(df):
    """
    Create a waterfall chart showing the breakdown of companies by segments.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame with financial data and segment information
    
    Returns:
    --------
    plotly.graph_objects.Figure
        Waterfall chart figure
    """
    # Count total companies with loans
    total_companies = df['company_id'].nunique()
    
    # Count companies with both deposits and loans
    companies_with_deposits = df[df['deposit'] > 0]['company_id'].nunique()
    
    # Get companies by utilization segment
    utilization_segments = df.groupby('company_id')['utilization_segment'].first()
    high_util = (utilization_segments == 'High').sum()
    medium_util = (utilization_segments == 'Medium').sum()
    low_util = (utilization_segments == 'Low').sum()
    
    # Create waterfall chart
    fig = go.Figure(go.Waterfall(
        name="Waterfall Chart", 
        orientation="v",
        measure=["absolute", "relative", "relative", "relative", "relative"],
        x=["Total Companies", "With Deposits", "High Utilization", "Medium Utilization", "Low Utilization"],
        textposition="outside",
        text=[total_companies, companies_with_deposits, high_util, medium_util, low_util],
        y=[total_companies, companies_with_deposits - total_companies, high_util, medium_util, low_util],
        connector={"line": {"color": "rgb(63, 63, 63)"}},
    ))
    
    fig.update_layout(
        title="Company Segments Waterfall Chart",
        showlegend=False
    )
    
    return fig

def analyze_correlation(df):
    """
    Analyze correlation between loan utilization and deposits for each company.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame with financial data
    
    Returns:
    --------
    pandas.DataFrame
        DataFrame with correlation analysis results
    """
    # Calculate correlation for each company
    correlation_results = []
    
    for company_id in df['company_id'].unique():
        company_data = df[df['company_id'] == company_id]
        # Skip companies with no deposit data or insufficient data
        if company_data['deposit'].sum() == 0 or len(company_data) < 30:
            continue
            
        # Calculate correlation
        correlation, p_value = stats.pearsonr(
            company_data['loan_utilization'].fillna(0), 
            company_data['deposit'].fillna(0)
        )
        
        # Determine if correlation is significant
        is_significant = p_value < 0.05
        
        # Categorize correlation
        if correlation > 0.5 and is_significant:
            correlation_type = 'Highly Positive'
        elif correlation < -0.5 and is_significant:
            correlation_type = 'Highly Negative'
        elif correlation > 0.3 and is_significant:
            correlation_type = 'Moderately Positive'
        elif correlation < -0.3 and is_significant:
            correlation_type = 'Moderately Negative'
        elif is_significant:
            correlation_type = 'Weak'
        else:
            correlation_type = 'Not Significant'
            
        correlation_results.append({
            'company_id': company_id,
            'correlation': correlation,
            'p_value': p_value,
            'is_significant': is_significant,
            'correlation_type': correlation_type
        })
    
    correlation_df = pd.DataFrame(correlation_results)
    
    return correlation_df

def create_risk_flags(df, window_3m=90, window_6m=180):
    """
    Create early warning signals based on patterns in loan utilization and deposits.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame with financial data
    window_3m : int
        Window size for 3-month analysis (days)
    window_6m : int
        Window size for 6-month analysis (days)
    
    Returns:
    --------
    tuple
        (DataFrame with risk flags, DataFrame with risk summary)
    """
    # List to store risk flag results
    risk_results = []
    
    # Process each company
    for company_id in df['company_id'].unique():
        company_data = df[df['company_id'] == company_id].sort_values('date')
        
        # Skip if not enough data
        if len(company_data) < window_6m:
            continue
            
        # Calculate rolling metrics (3-month and 6-month windows)
        # For loan utilization
        company_data['util_3m_change'] = company_data['loan_utilization'].rolling(window=window_3m).apply(
            lambda x: (x.iloc[-1] - x.iloc[0]) / max(x.iloc[0], 0.0001) if len(x) > 0 else 0
        )
        company_data['util_6m_change'] = company_data['loan_utilization'].rolling(window=window_6m).apply(
            lambda x: (x.iloc[-1] - x.iloc[0]) / max(x.iloc[0], 0.0001) if len(x) > 0 else 0
        )
        
        # For deposits
        if company_data['deposit'].sum() > 0:  # Only if company has deposits
            company_data['deposit_3m_change'] = company_data['deposit'].rolling(window=window_3m).apply(
                lambda x: (x.iloc[-1] - x.iloc[0]) / max(x.iloc[0], 0.0001) if len(x) > 0 else 0
            )
            company_data['deposit_6m_change'] = company_data['deposit'].rolling(window=window_6m).apply(
                lambda x: (x.iloc[-1] - x.iloc[0]) / max(x.iloc[0], 0.0001) if len(x) > 0 else 0
            )
        else:
            # Set to NaN if no deposits
            company_data['deposit_3m_change'] = np.nan
            company_data['deposit_6m_change'] = np.nan
        
        # Flag 1: Loan utilization going up but deposit balance going down (3-month)
        company_data['flag_1_3m'] = (company_data['util_3m_change'] > 0.05) & (company_data['deposit_3m_change'] < -0.05)
        
        # Flag 2: Loan utilization going up but deposit balance going down (6-month)
        company_data['flag_1_6m'] = (company_data['util_6m_change'] > 0.1) & (company_data['deposit_6m_change'] < -0.1)
        
        # Flag 3: Loan utilization steady but deposit decreasing (3-month)
        company_data['flag_2_3m'] = (abs(company_data['util_3m_change']) < 0.05) & (company_data['deposit_3m_change'] < -0.05)
        
        # Flag 4: Loan utilization steady but deposit decreasing (6-month)
        company_data['flag_2_6m'] = (abs(company_data['util_6m_change']) < 0.05) & (company_data['deposit_6m_change'] < -0.1)
        
        # Flag 5: Loan decreasing but deposits diminishing faster (3-month)
        company_data['flag_3_3m'] = (company_data['util_3m_change'] < -0.05) & (company_data['deposit_3m_change'] < company_data['util_3m_change'])
        
        # Flag 6: Loan decreasing but deposits diminishing faster (6-month)
        company_data['flag_3_6m'] = (company_data['util_6m_change'] < -0.1) & (company_data['deposit_6m_change'] < company_data['util_6m_change'])
        
        # Create overall risk flag
        company_data['has_risk_flag'] = (
            company_data['flag_1_3m'].fillna(False) | 
            company_data['flag_1_6m'].fillna(False) | 
            company_data['flag_2_3m'].fillna(False) | 
            company_data['flag_2_6m'].fillna(False) | 
            company_data['flag_3_3m'].fillna(False) | 
            company_data['flag_3_6m'].fillna(False)
        )
        
        # Count risk flags for recent periods
        last_quarter = company_data.iloc[-90:] if len(company_data) >= 90 else company_data
        risk_days_count = last_quarter['has_risk_flag'].sum()
        risk_score = risk_days_count / len(last_quarter) if len(last_quarter) > 0 else 0
        
        # Store results
        risk_results.append({
            'company_id': company_id,
            'risk_days_count': risk_days_count,
            'risk_score': risk_score,
            'high_risk': risk_score > 0.2  # Flag as high risk if more than 20% of days have risk flags
        })
        
        # Update risk flags in the original dataframe (for this company)
        df.loc[company_data.index, 'has_risk_flag'] = company_data['has_risk_flag']
    
    risk_df = pd.DataFrame(risk_results)
    
    # Merge risk flags back to original data
    df = df.merge(risk_df[['company_id', 'risk_score', 'high_risk']], 
                 on='company_id', how='left')
    
    return df, risk_df

def perform_clustering(df):
    """
    Perform clustering and cohort analysis on the financial data.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame with financial data and risk flags
    
    Returns:
    --------
    tuple
        (quarterly data, cluster profiles, transition data)
    """
    # Prepare feature set for clustering
    # We'll create quarterly aggregates to capture seasonal patterns
    df['quarter'] = df['date'].dt.to_period('Q')
    
    # Create quarterly aggregates
    quarterly_data = df.groupby(['company_id', 'quarter']).agg({
        'loan_utilization': 'mean',
        'used_loan': 'mean',
        'unused_loan': 'mean',
        'deposit': 'mean',
        'has_risk_flag': 'mean'  # Proportion of days with risk flags
    }).reset_index()
    
    # Convert quarter to string for easier handling
    quarterly_data['quarter'] = quarterly_data['quarter'].astype(str)
    
    # List of quarters in the data
    quarters = sorted(quarterly_data['quarter'].unique())
    
    # Dictionary to store cluster assignments for each quarter
    cluster_assignments = {}
    cluster_profiles = {}
    
    # Perform clustering for each quarter
    for quarter in quarters:
        quarter_data = quarterly_data[quarterly_data['quarter'] == quarter]
        
        # Select features for clustering
        features = quarter_data[['loan_utilization', 'used_loan', 'deposit', 'has_risk_flag']].copy()
        
        # Handle missing values
        features = features.fillna(0)
        
        # Standardize features
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(features)
        
        # Determine optimal number of clusters (simplified - use 3 clusters)
        n_clusters = 3
        
        # Perform KMeans clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        quarter_data['cluster'] = kmeans.fit_predict(scaled_features)
        
        # Store cluster assignments
        cluster_assignments[quarter] = quarter_data[['company_id', 'cluster']].set_index('company_id')['cluster']
        
        # Create cluster profiles
        cluster_profiles[quarter] = quarter_data.groupby('cluster').agg({
            'loan_utilization': 'mean',
            'used_loan': 'mean',
            'deposit': 'mean',
            'has_risk_flag': 'mean',
            'company_id': 'count'
        }).rename(columns={'company_id': 'count'})
    
    # Track cluster transitions
    transitions = []
    
    for i in range(len(quarters) - 1):
        current_quarter = quarters[i]
        next_quarter = quarters[i + 1]
        
        # Get companies present in both quarters
        common_companies = set(cluster_assignments[current_quarter].index) & set(cluster_assignments[next_quarter].index)
        
        for company in common_companies:
            from_cluster = cluster_assignments[current_quarter][company]
            to_cluster = cluster_assignments[next_quarter][company]
            
            transitions.append({
                'company_id': company,
                'from_quarter': current_quarter,
                'to_quarter': next_quarter,
                'from_cluster': from_cluster,
                'to_cluster': to_cluster,
                'changed_cluster': from_cluster != to_cluster
            })
    
    transitions_df = pd.DataFrame(transitions)
    
    # Convert cluster profiles to DataFrame for easier handling
    cluster_profile_df = pd.DataFrame([
        {
            'quarter': quarter,
            'cluster': cluster,
            'loan_utilization': profile.loc[cluster, 'loan_utilization'],
            'used_loan': profile.loc[cluster, 'used_loan'],
            'deposit': profile.loc[cluster, 'deposit'],
            'risk_flag_rate': profile.loc[cluster, 'has_risk_flag'],
            'count': profile.loc[cluster, 'count']
        }
        for quarter, profile in cluster_profiles.items()
        for cluster in profile.index
    ])
    
    return quarterly_data, cluster_profile_df, transitions_df

def analyze_time_patterns(df):
    """
    Analyze seasonality and trends in the financial data.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame with financial data
    
    Returns:
    --------
    tuple
        (monthly data, utilization decomposition, deposit decomposition)
    """
    # Create monthly aggregates for key metrics
    df['month'] = df['date'].dt.to_period('M')
    monthly_data = df.groupby('month').agg({
        'loan_utilization': 'mean',
        'used_loan': 'sum',
        'deposit': 'sum',
        'has_risk_flag': 'mean'
    }).reset_index()
    
    # Convert month to datetime for time series analysis
    monthly_data['month'] = monthly_data['month'].dt.to_timestamp()
    
    # Analyze seasonality and trend for loan utilization
    utilization_series = monthly_data.set_index('month')['loan_utilization']
    
    # Perform seasonal decomposition
    decomposition_util = seasonal_decompose(utilization_series, model='additive', period=12)
    
    # Similarly for deposits
    deposit_series = monthly_data.set_index('month')['deposit']
    decomposition_deposit = seasonal_decompose(deposit_series, model='additive', period=12)
    
    return monthly_data, decomposition_util, decomposition_deposit

def create_visualizations(df, company_agg, correlation_df, risk_df, cluster_profile_df, monthly_data):
    """
    Create visualizations for the financial data analysis.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame with financial data
    company_agg : pandas.DataFrame
        Aggregated company data
    correlation_df : pandas.DataFrame
        Correlation analysis results
    risk_df : pandas.DataFrame
        Risk analysis results
    cluster_profile_df : pandas.DataFrame
        Cluster profiles
    monthly_data : pandas.DataFrame
        Monthly aggregated data
    
    Returns:
    --------
    list
        List of matplotlib figures
    """
    figures = []
    
    # Figure 1: Segment distribution
    fig_segments = plt.figure(figsize=(15, 10))
    
    # Plot 1: Distribution of loan utilization segments
    plt.subplot(2, 2, 1)
    utilization_counts = company_agg['utilization_segment'].value_counts()
    plt.pie(utilization_counts, labels=utilization_counts.index, autopct='%1.1f%%')
    plt.title('Distribution of Loan Utilization Segments')
    
    # Plot 2: Distribution of used loan segments
    plt.subplot(2, 2, 2)
    used_loan_counts = company_agg['used_loan_segment'].value_counts()
    plt.pie(used_loan_counts, labels=used_loan_counts.index, autopct='%1.1f%%')
    plt.title('Distribution of Used Loan Segments')
    
    # Plot 3: Distribution of deposit segments
    plt.subplot(2, 2, 3)
    deposit_counts = company_agg[company_agg['deposit'] > 0]['deposit_segment'].value_counts()
    plt.pie(deposit_counts, labels=deposit_counts.index, autopct='%1.1f%%')
    plt.title('Distribution of Deposit Segments (Companies with Deposits)')
    
    # Plot 4: Risk distribution
    plt.subplot(2, 2, 4)
    risk_counts = risk_df['high_risk'].value_counts()
    plt.pie(risk_counts, labels=['Low Risk', 'High Risk'] if len(risk_counts) > 1 else ['Low Risk'], 
            autopct='%1.1f%%')
    plt.title('Risk Distribution')
    
    plt.tight_layout()
    figures.append(fig_segments)
    
    # Figure 2: Correlation visualization
    if not correlation_df.empty:
        fig_corr = plt.figure(figsize=(10, 6))
        correlation_counts = correlation_df['correlation_type'].value_counts()
        plt.bar(correlation_counts.index, correlation_counts.values)
        plt.xlabel('Correlation Type')
        plt.ylabel('Number of Companies')
        plt.title('Distribution of Correlation Types between Loan Utilization and Deposits')
        plt.xticks(rotation=45)
        plt.tight_layout()
        figures.append(fig_corr)
    
    # Figure 3: Time series visualization
    fig_time = plt.figure(figsize=(15, 10))
    
    # Plot 1: Average loan utilization over time
    plt.subplot(2, 2, 1)
    plt.plot(monthly_data['month'], monthly_data['loan_utilization'])
    plt.xlabel('Month')
    plt.ylabel('Average Loan Utilization')
    plt.title('Average Loan Utilization Over Time')
    
    # Plot 2: Total used loan over time
    plt.subplot(2, 2, 2)
    plt.plot(monthly_data['month'], monthly_data['used_loan'] / 1e6)
    plt.xlabel('Month')
    plt.ylabel('Total Used Loan (Millions)')
    plt.title('Total Used Loan Over Time')
    
    # Plot 3: Total deposits over time
    plt.subplot(2, 2, 3)
    plt.plot(monthly_data['month'], monthly_data['deposit'] / 1e6)
    plt.xlabel('Month')
    plt.ylabel('Total Deposits (Millions)')
    plt.title('Total Deposits Over Time')
    
    # Plot 4: Risk flag rate over time
    plt.subplot(2, 2, 4)
    plt.plot(monthly_data['month'], monthly_data['has_risk_flag'].fillna(0) * 100)
    plt.xlabel('Month')
    plt.ylabel('Risk Flag Rate (%)')
    plt.title('Risk Flag Rate Over Time')
    
    plt.tight_layout()
    figures.append(fig_time)
    
    # Figure 4: Cluster profile visualization
    fig_cluster = plt.figure(figsize=(15, 10))
    
    # Select a few quarters for demonstration
    sample_quarters = sorted(cluster_profile_df['quarter'].unique())[-4:]
    
    for i, quarter in enumerate(sample_quarters):
        plt.subplot(2, 2, i+1)
        quarter_data = cluster_profile_df[cluster_profile_df['quarter'] == quarter]
        
        # Create a scatter plot with loan utilization vs. deposit, size by used loan, color by risk
        plt.scatter(
            quarter_data['loan_utilization'],
            quarter_data['deposit'] / 1e6,
            s=quarter_data['count'] * 10,
            c=quarter_data['risk_flag_rate'],
            cmap='YlOrRd',
            alpha=0.7
        )
        
        plt.xlabel('Loan Utilization')
        plt.ylabel('Average Deposit (Millions)')
        plt.title(f'Cluster Profiles - {quarter}')
        plt.colorbar(label='Risk Flag Rate')
        
    plt.tight_layout()
    figures.append(fig_cluster)
    
    # Figure 5: Heatmap of loan utilization vs. deposits by segment
    fig_heatmap = plt.figure(figsize=(12, 8))
    
    # Create a cross-tabulation of utilization segment vs. deposit segment
    cross_tab = pd.crosstab(
        company_agg['utilization_segment'], 
        company_agg['deposit_segment'].fillna('No Deposits'),
        values=company_agg['risk_score'].fillna(0),
        aggfunc='mean'
    )
    
    # Create heatmap
    sns.heatmap(cross_tab, annot=True, cmap='YlOrRd', fmt='.2f')
    plt.title('Average Risk Score by Loan Utilization and Deposit Segments')
    plt.tight_layout()
    figures.append(fig_heatmap)
    
    return figures

def main():
    """
    Main function to orchestrate the entire analysis.
    """
    # Generate synthetic data
    print("Generating synthetic data...")
    df = generate_synthetic_data()
    
    # Clean data
    print("Cleaning data based on 80% non-zero, non-NaN, non-infinite rule...")
    df = clean_data(df)
    
    # Segment companies
    print("Segmenting companies into high, medium, and low groups...")
    df, company_agg = segment_companies(df)
    
    # Create waterfall chart
    print("Creating waterfall chart for customer segmentation...")
    waterfall_fig = create_waterfall_chart(df)
    
    # Analyze correlation
    print("Analyzing correlation between loan utilization and deposits...")
    correlation_df = analyze_correlation(df)
    
    # Create risk flags
    print("Creating early warning signals based on risk patterns...")
    df, risk_df = create_risk_flags(df)
    
    # Perform clustering and cohort analysis
    print("Performing clustering and cohort analysis...")
    quarterly_data, cluster_profile_df, transitions_df = perform_clustering(df)
    
    # Analyze time patterns
    print("Analyzing seasonality and trends...")
    monthly_data, decomposition_util, decomposition_deposit = analyze_time_patterns(df)
    
    # Create visualizations
    print("Creating visualizations...")
    figures = create_visualizations(
        df, company_agg, correlation_df, risk_df, cluster_profile_df, monthly_data
    )
    
    # Display waterfall chart using plotly
    waterfall_fig.show()
    
    # Display matplotlib figures
    for fig in figures:
        plt.figure(fig.number)
        plt.show()
    
    # Print summary statistics
    print("\nSummary Statistics:")
    print(f"Number of companies after cleaning: {df['company_id'].nunique()}")
    print(f"Companies with deposits: {df[df['deposit'] > 0]['company_id'].nunique()}")
    print(f"Companies with high risk flags: {risk_df[risk_df['high_risk']].shape[0]}")
    
    if not correlation_df.empty:
        print(f"Companies with significant positive correlation: {correlation_df[correlation_df['correlation_type'].str.contains('Positive')].shape[0]}")
        print(f"Companies with significant negative correlation: {correlation_df[correlation_df['correlation_type'].str.contains('Negative')].shape[0]}")
    
    # Identify top risk companies
    top_risk_companies = risk_df.sort_values('risk_score', ascending=False).head(5)
    print("\nTop 5 Companies with Highest Risk:")
    print(top_risk_companies[['company_id', 'risk_score']])
    
    # Identify anti-correlated companies (potential risk)
    if not correlation_df.empty:
        anticorrelated = correlation_df[correlation_df['correlation_type'] == 'Highly Negative']
        print("\nCompanies with Highly Negative Correlation (Deposit vs. Loan Utilization):")
        if not anticorrelated.empty:
            print(anticorrelated[['company_id', 'correlation']])
        else:
            print("None found.")
    
    return df, waterfall_fig, correlation_df, risk_df, cluster_profile_df, transitions_df

# Run the analysis if this script is executed directly
if __name__ == "__main__":
    main()
