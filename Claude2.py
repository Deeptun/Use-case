import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from scipy import stats
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Set plotting style
plt.style.use('ggplot')
sns.set_palette('Set2')

def generate_realistic_financial_data(num_companies=100, start_date='2021-01-01', end_date='2024-12-31'):
    """
    Generate realistic financial data for companies over a period of time.
    
    Parameters:
    -----------
    num_companies : int
        Number of companies to generate data for
    start_date : str
        Start date in format 'YYYY-MM-DD'
    end_date : str
        End date in format 'YYYY-MM-DD'
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame containing the generated data
    """
    print(f"Generating data for {num_companies} companies from {start_date} to {end_date}...")
    
    # Convert start and end dates to datetime
    start = datetime.strptime(start_date, '%Y-%m-%d')
    end = datetime.strptime(end_date, '%Y-%m-%d')
    
    # Calculate number of days
    days = (end - start).days + 1
    
    # Create company IDs
    company_ids = [f'COMP{str(i).zfill(3)}' for i in range(1, num_companies + 1)]
    
    # Create date range
    date_range = [start + timedelta(days=i) for i in range(days)]
    date_strings = [d.strftime('%Y-%m-%d') for d in date_range]
    
    # Initialize empty list to store data
    data = []
    
    # For each company
    for company_id in company_ids:
        # Decide if company has deposit data
        has_deposit = np.random.random() < 0.7  # 70% have deposit data
        
        # Generate base values that will be modified over time
        base_deposit = np.random.randint(100000, 2000000) if has_deposit else 0
        base_used_loan = np.random.randint(50000, 1500000)
        base_unused_loan = np.random.randint(50000, 750000)
        
        # Company behavior pattern
        # 1: Stable, 2: Deteriorating, 3: Improving, 4: Seasonal, 5: Volatile
        pattern = np.random.choice([1, 2, 3, 4, 5], p=[0.3, 0.2, 0.2, 0.2, 0.1])
        
        # Seasonal factors (quarterly)
        seasonal_factors = np.array([1.0, 1.2, 0.9, 1.1])
        
        # For each day
        for i, date_str in enumerate(date_strings):
            # Get day of year for seasonal factors
            day_of_year = datetime.strptime(date_str, '%Y-%m-%d').timetuple().tm_yday
            quarter = (day_of_year // 90) % 4
            seasonal_factor = seasonal_factors[quarter]
            
            # Initialize values based on pattern
            date_obj = datetime.strptime(date_str, '%Y-%m-%d')
            days_from_start = (date_obj - start).days
            years_from_start = days_from_start / 365.0
            
            # Base noise factors
            noise_factor_deposit = np.random.normal(1, 0.05)  # 5% standard deviation
            noise_factor_used = np.random.normal(1, 0.03)
            noise_factor_unused = np.random.normal(1, 0.04)
            
            # Initialize with base values
            deposit = base_deposit
            used_loan = base_used_loan
            unused_loan = base_unused_loan
            
            # Apply pattern modifications
            if pattern == 1:  # Stable
                deposit *= noise_factor_deposit
                used_loan *= noise_factor_used
                unused_loan *= noise_factor_unused
            
            elif pattern == 2:  # Deteriorating
                # Deposits decrease over time, used loans increase
                deposit_trend = max(0.6, 1 - years_from_start * 0.1)  # Gradual decrease
                used_trend = min(1.5, 1 + years_from_start * 0.1)     # Gradual increase
                unused_trend = max(0.7, 1 - years_from_start * 0.08)  # Slight decrease
                
                deposit *= deposit_trend * noise_factor_deposit
                used_loan *= used_trend * noise_factor_used
                unused_loan *= unused_trend * noise_factor_unused
            
            elif pattern == 3:  # Improving
                # Deposits increase, used loans decrease
                deposit_trend = min(1.6, 1 + years_from_start * 0.15)
                used_trend = max(0.7, 1 - years_from_start * 0.07)
                unused_trend = min(1.3, 1 + years_from_start * 0.08)
                
                deposit *= deposit_trend * noise_factor_deposit
                used_loan *= used_trend * noise_factor_used
                unused_loan *= unused_trend * noise_factor_unused
            
            elif pattern == 4:  # Seasonal
                # Apply seasonal factors
                deposit *= seasonal_factor * noise_factor_deposit
                used_loan *= (2 - seasonal_factor) * noise_factor_used  # Inverse seasonal effect
                unused_loan *= noise_factor_unused
            
            else:  # Volatile
                # More random variations
                deposit *= noise_factor_deposit * np.random.normal(1, 0.2)
                used_loan *= noise_factor_used * np.random.normal(1, 0.15)
                unused_loan *= noise_factor_unused * np.random.normal(1, 0.12)
            
            # Occasionally set to zero (missing data)
            # More likely to have zeros near the beginning for realistic data quality issues
            zero_probability = max(0.01, 0.05 - years_from_start * 0.01)
            
            if not has_deposit or np.random.random() < zero_probability:
                deposit = 0
                
            if np.random.random() < zero_probability:
                used_loan = 0
                
            if np.random.random() < zero_probability:
                unused_loan = 0
            
            # Add row to data
            data.append({
                'company_id': company_id,
                'date': date_str,
                'deposit_balance': round(deposit),
                'used_loan': round(used_loan),
                'unused_loan': round(unused_loan)
            })
    
    # Convert to DataFrame
    df = pd.DataFrame(data)
    
    # Sort by company_id and date
    df = df.sort_values(['company_id', 'date'])
    
    print(f"Generated {len(df)} records.")
    return df

def filter_valid_companies(df, threshold=0.8):
    """
    Filter companies with at least 80% of non-zero, non-NaN, non-infinite values.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing financial data
    threshold : float
        Minimum percentage of valid values
        
    Returns:
    --------
    pandas.DataFrame
        Filtered DataFrame
    """
    print(f"Filtering companies with at least {threshold*100}% valid values...")
    
    # Count total days per company
    company_days = df.groupby('company_id').size()
    
    valid_companies = []
    
    for company in df['company_id'].unique():
        company_data = df[df['company_id'] == company]
        
        # Check deposit balance
        valid_deposit = ((company_data['deposit_balance'] > 0) & 
                         (~company_data['deposit_balance'].isna()) & 
                         (~np.isinf(company_data['deposit_balance']))).mean()
        
        # Check used loan
        valid_used_loan = ((company_data['used_loan'] > 0) & 
                          (~company_data['used_loan'].isna()) & 
                          (~np.isinf(company_data['used_loan']))).mean()
        
        # Check unused loan
        valid_unused_loan = ((company_data['unused_loan'] > 0) & 
                            (~company_data['unused_loan'].isna()) & 
                            (~np.isinf(company_data['unused_loan']))).mean()
        
        # If all three metrics meet the threshold, keep the company
        if (valid_deposit >= threshold and 
            valid_used_loan >= threshold and 
            valid_unused_loan >= threshold):
            valid_companies.append(company)
    
    # Filter DataFrame to include only valid companies
    filtered_df = df[df['company_id'].isin(valid_companies)]
    
    print(f"Retained {len(valid_companies)} companies out of {len(df['company_id'].unique())}")
    print(f"Retained {len(filtered_df)} records out of {len(df)}")
    
    return filtered_df

def calculate_loan_utilization(df):
    """
    Calculate loan utilization ratio.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing financial data
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame with added loan_utilization column
    """
    print("Calculating loan utilization...")
    
    # Calculate total loan (used + unused)
    df['total_loan'] = df['used_loan'] + df['unused_loan']
    
    # Calculate loan utilization ratio
    df['loan_utilization'] = np.where(
        df['total_loan'] > 0,
        df['used_loan'] / df['total_loan'],
        0
    )
    
    return df

def create_segments(df):
    """
    Create high, medium, and low segments for loan utilization, used loan, and deposits.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing financial data
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame with added segmentation columns
    """
    print("Creating segments for loan utilization, used loan, and deposits...")
    
    # Function to categorize values into High, Medium, Low
    def categorize(values):
        # Handle zero values separately
        non_zero = values[values > 0]
        if len(non_zero) == 0:
            return 'Low'
        
        q33 = np.percentile(non_zero, 33)
        q66 = np.percentile(non_zero, 66)
        
        if values <= 0:
            return 'Low'
        elif values <= q33:
            return 'Low'
        elif values <= q66:
            return 'Medium'
        else:
            return 'High'
    
    # Vectorize the categorize function
    categorize_vec = np.vectorize(categorize)
    
    # Create segmentation for loan utilization
    df['utilization_segment'] = pd.cut(
        df['loan_utilization'], 
        bins=[0, 0.33, 0.66, 1], 
        labels=['Low', 'Medium', 'High'],
        include_lowest=True
    )
    
    # Create segmentation for used loan
    non_zero_used = df['used_loan'][df['used_loan'] > 0]
    q33_used = non_zero_used.quantile(0.33)
    q66_used = non_zero_used.quantile(0.66)
    
    df['used_loan_segment'] = pd.cut(
        df['used_loan'], 
        bins=[0, q33_used, q66_used, float('inf')], 
        labels=['Low', 'Medium', 'High'],
        include_lowest=True
    )
    
    # Create segmentation for deposits
    non_zero_deposits = df['deposit_balance'][df['deposit_balance'] > 0]
    if len(non_zero_deposits) > 0:
        q33_deposit = non_zero_deposits.quantile(0.33)
        q66_deposit = non_zero_deposits.quantile(0.66)
        
        df['deposit_segment'] = pd.cut(
            df['deposit_balance'], 
            bins=[0, q33_deposit, q66_deposit, float('inf')], 
            labels=['Low', 'Medium', 'High'],
            include_lowest=True
        )
    else:
        # Handle the case when there are no positive deposits
        df['deposit_segment'] = 'Low'
    
    return df

def analyze_correlations(df):
    """
    Analyze correlations between loan utilization and deposits.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing financial data
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame with added correlation flags
    """
    print("Analyzing correlations between loan utilization and deposits...")
    
    # Initialize correlation columns
    df['corr_flag'] = 'No correlation'
    
    # For each company
    for company in df['company_id'].unique():
        # Get company data
        company_data = df[df['company_id'] == company].copy()
        
        # Check if company has sufficient deposit and loan utilization data
        valid_data = company_data[(company_data['deposit_balance'] > 0) & 
                                  (company_data['loan_utilization'] > 0)]
        
        if len(valid_data) > 30:  # Minimum 30 days of data for meaningful correlation
            # Calculate correlation
            corr, p_value = stats.pearsonr(
                valid_data['loan_utilization'], 
                valid_data['deposit_balance']
            )
            
            # Flag based on correlation
            if p_value < 0.05:  # Statistically significant
                if corr > 0.6:
                    flag = 'Highly correlated (+)'
                elif corr < -0.6:
                    flag = 'Highly anti-correlated (-)'
                elif corr > 0.3:
                    flag = 'Moderately correlated (+)'
                elif corr < -0.3:
                    flag = 'Moderately anti-correlated (-)'
                else:
                    flag = 'Weakly correlated'
            else:
                flag = 'No significant correlation'
            
            # Assign flag to company
            df.loc[df['company_id'] == company, 'corr_flag'] = flag
    
    return df

def create_risk_flags(df):
    """
    Create risk flags based on patterns in loan utilization and deposits.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing financial data
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame with added risk flags
    """
    print("Creating risk flags for early warning signals...")
    
    # Ensure data is sorted by company_id and date
    df = df.sort_values(['company_id', 'date'])
    
    # Initialize risk flag columns
    df['risk_flag_3m'] = 'No risk'
    df['risk_flag_6m'] = 'No risk'
    
    # For each company
    for company in df['company_id'].unique():
        # Get company data
        company_data = df[df['company_id'] == company].copy()
        
        # Convert date to datetime if it's not already
        if company_data['date'].dtype != 'datetime64[ns]':
            company_data['date'] = pd.to_datetime(company_data['date'])
        
        # Calculate rolling averages for 3 months (90 days) and 6 months (180 days)
        company_data['utilization_3m_avg'] = company_data['loan_utilization'].rolling(window=90, min_periods=60).mean()
        company_data['utilization_6m_avg'] = company_data['loan_utilization'].rolling(window=180, min_periods=120).mean()
        company_data['deposit_3m_avg'] = company_data['deposit_balance'].rolling(window=90, min_periods=60).mean()
        company_data['deposit_6m_avg'] = company_data['deposit_balance'].rolling(window=180, min_periods=120).mean()
        
        # Calculate percentage changes
        company_data['utilization_3m_change'] = company_data['utilization_3m_avg'].pct_change(periods=90)
        company_data['utilization_6m_change'] = company_data['utilization_6m_avg'].pct_change(periods=180)
        company_data['deposit_3m_change'] = company_data['deposit_3m_avg'].pct_change(periods=90)
        company_data['deposit_6m_change'] = company_data['deposit_6m_avg'].pct_change(periods=180)
        
        # Create 3-month risk flags
        for i in range(len(company_data) - 90):
            current_idx = company_data.index[i + 90]
            
            # Skip if we don't have enough data
            if pd.isna(company_data.loc[current_idx, 'utilization_3m_change']) or pd.isna(company_data.loc[current_idx, 'deposit_3m_change']):
                continue
            
            # Warning pattern 1: Loan utilization increasing, deposits decreasing
            if (company_data.loc[current_idx, 'utilization_3m_change'] > 0.05 and 
                company_data.loc[current_idx, 'deposit_3m_change'] < -0.05):
                df.loc[current_idx, 'risk_flag_3m'] = 'High risk: Utilization ↑, Deposit ↓'
            
            # Warning pattern 2: Loan utilization steady, deposits decreasing
            elif (abs(company_data.loc[current_idx, 'utilization_3m_change']) < 0.03 and 
                  company_data.loc[current_idx, 'deposit_3m_change'] < -0.07):
                df.loc[current_idx, 'risk_flag_3m'] = 'Medium risk: Utilization →, Deposit ↓'
            
            # Warning pattern 3: Loan utilization decreasing, deposits decreasing faster
            elif (company_data.loc[current_idx, 'utilization_3m_change'] < -0.05 and 
                  company_data.loc[current_idx, 'deposit_3m_change'] < -0.1):
                df.loc[current_idx, 'risk_flag_3m'] = 'Medium risk: Utilization ↓, Deposit ↓↓'
        
        # Create 6-month risk flags
        for i in range(len(company_data) - 180):
            current_idx = company_data.index[i + 180]
            
            # Skip if we don't have enough data
            if pd.isna(company_data.loc[current_idx, 'utilization_6m_change']) or pd.isna(company_data.loc[current_idx, 'deposit_6m_change']):
                continue
            
            # Warning pattern 1: Loan utilization increasing, deposits decreasing
            if (company_data.loc[current_idx, 'utilization_6m_change'] > 0.1 and 
                company_data.loc[current_idx, 'deposit_6m_change'] < -0.1):
                df.loc[current_idx, 'risk_flag_6m'] = 'High risk: Utilization ↑, Deposit ↓'
            
            # Warning pattern 2: Loan utilization steady, deposits decreasing
            elif (abs(company_data.loc[current_idx, 'utilization_6m_change']) < 0.05 and 
                  company_data.loc[current_idx, 'deposit_6m_change'] < -0.15):
                df.loc[current_idx, 'risk_flag_6m'] = 'Medium risk: Utilization →, Deposit ↓'
            
            # Warning pattern 3: Loan utilization decreasing, deposits decreasing faster
            elif (company_data.loc[current_idx, 'utilization_6m_change'] < -0.1 and 
                  company_data.loc[current_idx, 'deposit_6m_change'] < -0.2):
                df.loc[current_idx, 'risk_flag_6m'] = 'Medium risk: Utilization ↓, Deposit ↓↓'
    
    return df

def cluster_companies(df):
    """
    Cluster companies based on their financial behavior.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing financial data
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame with added cluster information
    """
    print("Clustering companies based on financial behavior...")
    
    # Aggregate data by company
    company_features = []
    
    for company in df['company_id'].unique():
        company_data = df[df['company_id'] == company]
        
        # Calculate various metrics
        avg_utilization = company_data['loan_utilization'].mean()
        avg_deposit = company_data['deposit_balance'].mean()
        avg_used_loan = company_data['used_loan'].mean()
        deposit_volatility = company_data['deposit_balance'].std() / (avg_deposit + 1)  # Add 1 to avoid division by zero
        utilization_volatility = company_data['loan_utilization'].std()
        
        # Count risk flags
        high_risk_count_3m = (company_data['risk_flag_3m'].str.contains('High risk')).sum()
        high_risk_count_6m = (company_data['risk_flag_6m'].str.contains('High risk')).sum()
        medium_risk_count_3m = (company_data['risk_flag_3m'].str.contains('Medium risk')).sum()
        medium_risk_count_6m = (company_data['risk_flag_6m'].str.contains('Medium risk')).sum()
        
        # Correlation type
        corr_type = company_data['corr_flag'].iloc[0]
        
        # Create feature dictionary
        features = {
            'company_id': company,
            'avg_utilization': avg_utilization,
            'avg_deposit': avg_deposit,
            'avg_used_loan': avg_used_loan,
            'deposit_volatility': deposit_volatility,
            'utilization_volatility': utilization_volatility,
            'high_risk_3m': high_risk_count_3m,
            'high_risk_6m': high_risk_count_6m,
            'medium_risk_3m': medium_risk_count_3m,
            'medium_risk_6m': medium_risk_count_6m,
            'corr_type': corr_type
        }
        
        company_features.append(features)
    
    # Create DataFrame from features
    feature_df = pd.DataFrame(company_features)
    
    # One-hot encode correlation type
    corr_dummies = pd.get_dummies(feature_df['corr_type'], prefix='corr')
    feature_df = pd.concat([feature_df, corr_dummies], axis=1)
    
    # Select numerical features for clustering
    cluster_features = [
        'avg_utilization', 'avg_deposit', 'avg_used_loan',
        'deposit_volatility', 'utilization_volatility',
        'high_risk_3m', 'high_risk_6m', 'medium_risk_3m', 'medium_risk_6m'
    ]
    
    # Scale features
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(feature_df[cluster_features])
    
    # Determine optimal number of clusters using the elbow method
    inertia = []
    k_range = range(2, 10)
    
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(scaled_features)
        inertia.append(kmeans.inertia_)
    
    # Simple method to find the elbow point
    inertia_diff = np.diff(inertia)
    optimal_k = k_range[np.argmax(np.abs(np.diff(inertia_diff))) + 1]
    
    # Perform clustering with optimal k
    kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
    kmeans.fit(scaled_features)
    
    # Add cluster labels to feature DataFrame
    feature_df['cluster'] = kmeans.labels_
    
    # Create cluster descriptions based on characteristics
    cluster_descriptions = {}
    
    for cluster in range(optimal_k):
        cluster_data = feature_df[feature_df['cluster'] == cluster]
        
        # Calculate key metrics for this cluster
        avg_util = cluster_data['avg_utilization'].mean()
        avg_dep = cluster_data['avg_deposit'].mean()
        high_risk = (cluster_data['high_risk_3m'] + cluster_data['high_risk_6m']).mean()
        volatility = (cluster_data['deposit_volatility'] + cluster_data['utilization_volatility']).mean()
        
        # Create description
        if high_risk > 20:
            risk_status = "High-Risk"
        elif high_risk > 5:
            risk_status = "Medium-Risk"
        else:
            risk_status = "Low-Risk"
        
        if avg_util > 0.7:
            util_status = "High-Utilization"
        elif avg_util > 0.4:
            util_status = "Medium-Utilization"
        else:
            util_status = "Low-Utilization"
        
        if volatility > 0.5:
            vol_status = "Volatile"
        else:
            vol_status = "Stable"
        
        description = f"{risk_status} {util_status} {vol_status} Companies"
        cluster_descriptions[cluster] = description
    
    # Add cluster description to feature DataFrame
    feature_df['cluster_description'] = feature_df['cluster'].map(cluster_descriptions)
    
    # Join cluster information back to main DataFrame
    cluster_info = feature_df[['company_id', 'cluster', 'cluster_description']]
    df = df.merge(cluster_info, on='company_id', how='left')
    
    return df

def perform_cohort_analysis(df):
    """
    Perform cohort analysis to track behavior over time.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing financial data
        
    Returns:
    --------
    tuple
        (cohort_df, seasonal_effects_df)
    """
    print("Performing cohort analysis...")
    
    # Ensure date is in datetime format
    if df['date'].dtype != 'datetime64[ns]':
        df['date'] = pd.to_datetime(df['date'])
    
    # Extract year and month for cohort analysis
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['quarter'] = df['date'].dt.quarter
    df['yearmonth'] = df['year'].astype(str) + '-' + df['month'].astype(str).str.zfill(2)
    df['yearquarter'] = df['year'].astype(str) + '-Q' + df['quarter'].astype(str)
    
    # Calculate monthly metrics by cluster
    monthly_metrics = []
    
    for (year, month, cluster) in df.groupby(['year', 'month', 'cluster']).groups.keys():
        cluster_data = df[(df['year'] == year) & (df['month'] == month) & (df['cluster'] == cluster)]
        
        # Calculate metrics
        avg_utilization = cluster_data['loan_utilization'].mean()
        avg_deposit = cluster_data['deposit_balance'].mean()
        high_risk_pct = (cluster_data['risk_flag_3m'].str.contains('High risk')).mean() * 100
        
        monthly_metrics.append({
            'year': year,
            'month': month,
            'yearmonth': f"{year}-{str(month).zfill(2)}",
            'cluster': cluster,
            'cluster_description': cluster_data['cluster_description'].iloc[0],
            'avg_utilization': avg_utilization,
            'avg_deposit': avg_deposit,
            'high_risk_pct': high_risk_pct
        })
    
    # Create DataFrame
    cohort_df = pd.DataFrame(monthly_metrics)
    
    # Create seasonal effects analysis
    seasonal_effects = []
    
    for cluster in df['cluster'].unique():
        cluster_data = df[df['cluster'] == cluster]
        
        # Calculate quarterly metrics
        for quarter in range(1, 5):
            quarter_data = cluster_data[cluster_data['quarter'] == quarter]
            
            if len(quarter_data) > 0:
                avg_util = quarter_data['loan_utilization'].mean()
                avg_dep = quarter_data['deposit_balance'].mean()
                
                seasonal_effects.append({
                    'cluster': cluster,
                    'cluster_description': cluster_data['cluster_description'].iloc[0],
                    'quarter': f"Q{quarter}",
                    'avg_utilization': avg_util,
                    'avg_deposit': avg_dep
                })
    
    seasonal_effects_df = pd.DataFrame(seasonal_effects)
    
    return cohort_df, seasonal_effects_df

def create_waterfall_chart(df):
    """
    Create a waterfall chart showing client segmentation.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing financial data
        
    Returns:
    --------
    None (displays chart)
    """
    print("Creating waterfall chart...")
    
    # Count total companies
    total_companies = len(df['company_id'].unique())
    
    # Count companies with loans
    companies_with_loans = len(df[df['used_loan'] > 0]['company_id'].unique())
    
    # Count companies with both loans and deposits
    companies_with_both = len(df[(df['used_loan'] > 0) & (df['deposit_balance'] > 0)]['company_id'].unique())
    
    # Count companies by utilization segment
    companies_by_util = df.groupby(['company_id']).agg({'utilization_segment': lambda x: x.mode()[0]})
    high_util = len(companies_by_util[companies_by_util['utilization_segment'] == 'High'])
    medium_util = len(companies_by_util[companies_by_util['utilization_segment'] == 'Medium'])
    low_util = len(companies_by_util[companies_by_util['utilization_segment'] == 'Low'])
    
    # Create waterfall chart data
    measure = ['absolute', 'relative', 'relative', 'relative', 'relative', 'relative', 'total']
    
    fig = go.Figure(go.Waterfall(
        name="Client Segmentation",
        orientation="v",
        measure=measure,
        x=['Total Clients', 'Clients with Loans', 'Clients with Both', 
           'High Utilization', 'Medium Utilization', 'Low Utilization', 'Final'],
        y=[total_companies, 
           -(total_companies - companies_with_loans), 
           -(companies_with_loans - companies_with_both),
           -((companies_with_both) - (high_util + medium_util + low_util)),
           high_util, medium_util, low_util],
        connector={"line": {"color": "rgb(63, 63, 63)"}},
        decreasing={"marker": {"color": "Maroon"}},
        increasing={"marker": {"color": "Teal"}},
        totals={"marker": {"color": "deep"}},
        text=[f"{total_companies}", 
              f"-{total_companies - companies_with_loans}", 
              f"-{companies_with_loans - companies_with_both}",
              f"-{(companies_with_both) - (high_util + medium_util + low_util)}",
              f"{high_util}", f"{medium_util}", f"{low_util}"],
        textposition="outside"
    ))
    
    fig.update_layout(
        title="Client Segmentation Waterfall Chart",
        showlegend=False,
        height=600,
        width=800
    )
    
    # Display chart
    fig.show()
    
    return fig

def create_segment_distribution(df):
    """
    Create a visualization of segment distributions.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing financial data
        
    Returns:
    --------
    None (displays chart)
    """
    print("Creating segment distribution chart...")
    
    # Create a snapshot of the latest data for each company
    latest_date = df['date'].max()
    snapshot = df[df['date'] == latest_date]
    
    # Create a figure with subplots
    fig = make_subplots(rows=2, cols=2, 
                        subplot_titles=("Loan Utilization Segments", 
                                        "Used Loan Segments",
                                        "Deposit Segments",
                                        "Risk Flags Distribution"))
    
    # Add utilization segment distribution
    util_counts = snapshot['utilization_segment'].value_counts()
    fig.add_trace(
        go.Bar(x=util_counts.index, y=util_counts.values, name="Utilization", 
               marker_color=['#90CAF9', '#64B5F6', '#42A5F5']),
        row=1, col=1
    )
    
    # Add used loan segment distribution
    used_counts = snapshot['used_loan_segment'].value_counts()
    fig.add_trace(
        go.Bar(x=used_counts.index, y=used_counts.values, name="Used Loan",
               marker_color=['#A5D6A7', '#81C784', '#66BB6A']),
        row=1, col=2
    )
    
    # Add deposit segment distribution
    deposit_counts = snapshot['deposit_segment'].value_counts()
    fig.add_trace(
        go.Bar(x=deposit_counts.index, y=deposit_counts.values, name="Deposit",
               marker_color=['#FFCC80', '#FFB74D', '#FFA726']),
        row=2, col=1
    )
    
    # Add risk flag distribution
    risk_counts = snapshot['risk_flag_3m'].value_counts()
    fig.add_trace(
        go.Bar(x=risk_counts.index, y=risk_counts.values, name="Risk Flags",
               marker_color=['#EF9A9A', '#E57373', '#EF5350', '#F44336']),
        row=2, col=2
    )
    
    # Update layout
    fig.update_layout(
        title="Segment Distributions",
        height=800,
        width=1000,
        showlegend=False
    )
    
    # Display chart
    fig.show()
    
    return fig

def visualize_correlations(df):
    """
    Visualize correlations between loan utilization and deposits.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing financial data
        
    Returns:
    --------
    None (displays chart)
    """
    print("Creating correlation visualization...")
    
    # Create a figure
    fig = go.Figure()
    
    # Sample a few companies from each correlation flag
    corr_flags = df['corr_flag'].unique()
    
    for flag in corr_flags:
        companies = df[df['corr_flag'] == flag]['company_id'].unique()
        
        if len(companies) > 0:
            # Sample up to 2 companies from each flag
            sample_size = min(2, len(companies))
            sampled_companies = np.random.choice(companies, size=sample_size, replace=False)
            
            for company in sampled_companies:
                company_data = df[df['company_id'] == company]
                
                # Keep only rows with both metrics > 0
                valid_data = company_data[(company_data['deposit_balance'] > 0) & 
                                         (company_data['loan_utilization'] > 0)]
                
                if len(valid_data) > 0:
                    fig.add_trace(
                        go.Scatter(
                            x=valid_data['loan_utilization'],
                            y=valid_data['deposit_balance'],
                            mode='markers',
                            name=f"{company} ({flag})",
                            opacity=0.7
                        )
                    )
    
    # Update layout
    fig.update_layout(
        title="Correlation between Loan Utilization and Deposits",
        xaxis_title="Loan Utilization Ratio",
        yaxis_title="Deposit Balance",
        height=600,
        width=800
    )
    
    # Display chart
    fig.show()
    
    return fig

def visualize_risk_flags(df):
    """
    Visualize risk flags over time.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing financial data
        
    Returns:
    --------
    None (displays chart)
    """
    print("Creating risk flag visualization...")
    
    # Convert date to datetime
    if df['date'].dtype != 'datetime64[ns]':
        df['date'] = pd.to_datetime(df['date'])
    
    # Extract yearmonth
    df['yearmonth'] = df['date'].dt.strftime('%Y-%m')
    
    # Count high risk flags by month
    high_risk_3m = df[df['risk_flag_3m'].str.contains('High risk')].groupby('yearmonth').size()
    medium_risk_3m = df[df['risk_flag_3m'].str.contains('Medium risk')].groupby('yearmonth').size()
    high_risk_6m = df[df['risk_flag_6m'].str.contains('High risk')].groupby('yearmonth').size()
    medium_risk_6m = df[df['risk_flag_6m'].str.contains('Medium risk')].groupby('yearmonth').size()
    
    # Create figure
    fig = go.Figure()
    
    # Add traces
    fig.add_trace(
        go.Scatter(x=high_risk_3m.index, y=high_risk_3m.values, mode='lines+markers',
                  name='High Risk (3m)', line=dict(color='#F44336', width=2))
    )
    
    fig.add_trace(
        go.Scatter(x=medium_risk_3m.index, y=medium_risk_3m.values, mode='lines+markers',
                  name='Medium Risk (3m)', line=dict(color='#FF9800', width=2))
    )
    
    fig.add_trace(
        go.Scatter(x=high_risk_6m.index, y=high_risk_6m.values, mode='lines+markers',
                  name='High Risk (6m)', line=dict(color='#E91E63', width=2))
    )
    
    fig.add_trace(
        go.Scatter(x=medium_risk_6m.index, y=medium_risk_6m.values, mode='lines+markers',
                  name='Medium Risk (6m)', line=dict(color='#FFC107', width=2))
    )
    
    # Update layout
    fig.update_layout(
        title="Risk Flags Over Time",
        xaxis_title="Month",
        yaxis_title="Number of Risk Flags",
        height=600,
        width=1000
    )
    
    # Display chart
    fig.show()
    
    return fig

def visualize_cohort_analysis(cohort_df, seasonal_effects_df):
    """
    Visualize cohort analysis and seasonal effects.
    
    Parameters:
    -----------
    cohort_df : pandas.DataFrame
        DataFrame containing cohort analysis data
    seasonal_effects_df : pandas.DataFrame
        DataFrame containing seasonal effects data
        
    Returns:
    --------
    None (displays chart)
    """
    print("Creating cohort analysis visualizations...")
    
    # Create a figure with subplots
    fig = make_subplots(rows=2, cols=1, 
                        subplot_titles=("Cluster Behavior Over Time", 
                                        "Seasonal Effects by Cluster"),
                        vertical_spacing=0.2)
    
    # Add utilization over time by cluster
    for cluster in cohort_df['cluster'].unique():
        cluster_data = cohort_df[cohort_df['cluster'] == cluster]
        description = cluster_data['cluster_description'].iloc[0]
        
        fig.add_trace(
            go.Scatter(x=cluster_data['yearmonth'], y=cluster_data['avg_utilization'], 
                      mode='lines+markers', name=f"Cluster {cluster}: {description} (Utilization)",
                      line=dict(dash='solid')),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(x=cluster_data['yearmonth'], y=cluster_data['high_risk_pct'] / 100, 
                      mode='lines+markers', name=f"Cluster {cluster}: {description} (Risk %)",
                      line=dict(dash='dot')),
            row=1, col=1
        )
    
    # Add seasonal effects by cluster
    for cluster in seasonal_effects_df['cluster'].unique():
        cluster_data = seasonal_effects_df[seasonal_effects_df['cluster'] == cluster]
        description = cluster_data['cluster_description'].iloc[0]
        
        fig.add_trace(
            go.Bar(x=cluster_data['quarter'], y=cluster_data['avg_utilization'], 
                   name=f"Cluster {cluster}: {description}"),
            row=2, col=1
        )
    
    # Update layout
    fig.update_layout(
        title="Cohort Analysis and Seasonal Effects",
        height=1000,
        width=1200
    )
    
    # Display chart
    fig.show()
    
    return fig

def visualize_cluster_profiles(df):
    """
    Visualize profiles of different clusters.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing financial data
        
    Returns:
    --------
    None (displays chart)
    """
    print("Creating cluster profile visualization...")
    
    # Get latest data point for each company
    latest_date = df['date'].max()
    latest_data = df[df['date'] == latest_date]
    
    # Calculate average metrics by cluster
    cluster_metrics = []
    
    for cluster in latest_data['cluster'].unique():
        cluster_data = latest_data[latest_data['cluster'] == cluster]
        description = cluster_data['cluster_description'].iloc[0]
        
        # Calculate metrics
        avg_util = cluster_data['loan_utilization'].mean()
        avg_deposit = cluster_data['deposit_balance'].mean()
        high_risk_pct = (cluster_data['risk_flag_3m'].str.contains('High risk')).mean() * 100
        
        # Count companies by corr_flag
        corr_counts = cluster_data['corr_flag'].value_counts()
        
        for corr_type, count in corr_counts.items():
            cluster_metrics.append({
                'cluster': cluster,
                'description': description,
                'correlation_type': corr_type,
                'count': count,
                'avg_utilization': avg_util,
                'avg_deposit': avg_deposit,
                'high_risk_pct': high_risk_pct
            })
    
    cluster_metrics_df = pd.DataFrame(cluster_metrics)
    
    # Create a figure
    fig = px.treemap(
        cluster_metrics_df,
        path=['cluster', 'correlation_type'],
        values='count',
        color='avg_utilization',
        color_continuous_scale='RdYlGn_r',
        hover_data=['avg_deposit', 'high_risk_pct'],
        title="Cluster Profiles"
    )
    
    # Update layout
    fig.update_layout(
        height=600,
        width=1000
    )
    
    # Display chart
    fig.show()
    
    return fig

def main():
    """
    Main function to run the analysis pipeline.
    """
    print("Starting financial data analysis...")
    
    # Generate data
    df = generate_realistic_financial_data(num_companies=100, 
                                           start_date='2021-01-01', 
                                           end_date='2024-12-31')
    
    # Filter valid companies
    filtered_df = filter_valid_companies(df, threshold=0.8)
    
    # Calculate loan utilization
    df_with_util = calculate_loan_utilization(filtered_df)
    
    # Create segments
    segmented_df = create_segments(df_with_util)
    
    # Analyze correlations
    df_with_corr = analyze_correlations(segmented_df)
    
    # Create risk flags
    risk_df = create_risk_flags(df_with_corr)
    
    # Cluster companies
    clustered_df = cluster_companies(risk_df)
    
    # Perform cohort analysis
    cohort_df, seasonal_effects_df = perform_cohort_analysis(clustered_df)
    
    # Create visualizations
    waterfall_fig = create_waterfall_chart(clustered_df)
    segment_fig = create_segment_distribution(clustered_df)
    corr_fig = visualize_correlations(clustered_df)
    risk_fig = visualize_risk_flags(clustered_df)
    cohort_fig = visualize_cohort_analysis(cohort_df, seasonal_effects_df)
    profile_fig = visualize_cluster_profiles(clustered_df)
    
    print("Analysis complete!")
    return clustered_df, cohort_df, seasonal_effects_df

if __name__ == "__main__":
    # Run the analysis
    final_df, cohort_analysis, seasonal_effects = main()
    
    # Display sample of the final dataframe
    print("\nSample of the final dataframe:")
    print(final_df.head())
    
    # Display sample of the cohort analysis
    print("\nSample of the cohort analysis:")
    print(cohort_analysis.head())
    
    # Display sample of the seasonal effects
    print("\nSample of the seasonal effects:")
    print(seasonal_effects.head())
