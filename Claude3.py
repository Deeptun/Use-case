import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from scipy.stats import pearsonr
import datetime
from dateutil.relativedelta import relativedelta
import matplotlib.dates as mdates
from matplotlib.ticker import FuncFormatter
import warnings
warnings.filterwarnings('ignore')

# Set style for plots
plt.style.use('ggplot')
sns.set_palette("Set2")

def generate_financial_data(n_companies=100, years=4, seed=42):
    """
    Generate synthetic financial data for n_companies over specified number of years
    with daily frequency, including trends, seasonality, and some missing/zero values.
    
    Returns:
        pd.DataFrame: DataFrame with columns company_id, date, deposit_balance, used_loan, unused_loan
    """
    np.random.seed(seed)
    
    # Generate dates
    start_date = datetime.datetime.now() - relativedelta(years=years)
    end_date = datetime.datetime.now() - relativedelta(days=1)
    dates = pd.date_range(start=start_date, end=end_date, freq='D')
    
    company_ids = [f'COMP_{i:03d}' for i in range(1, n_companies+1)]
    
    # Lists to store data
    all_companies = []
    all_dates = []
    all_deposits = []
    all_used_loans = []
    all_unused_loans = []
    
    for company_id in company_ids:
        # Base values that differ by company
        base_deposit = np.random.uniform(100000, 10000000)
        base_loan_limit = np.random.uniform(500000, 20000000)
        
        # Company-specific trends
        deposit_trend = np.random.uniform(-0.1, 0.15)  # Annual trend
        loan_usage_trend = np.random.uniform(-0.08, 0.12)  # Annual trend
        
        # Decide if this company has deposits (some don't)
        has_deposits = np.random.random() > 0.15  # 85% of companies have deposits
        
        # Seasonal components
        deposit_seasonal_amp = base_deposit * np.random.uniform(0.05, 0.2)
        loan_seasonal_amp = base_loan_limit * np.random.uniform(0.03, 0.15)
        
        for date_idx, date in enumerate(dates):
            # Time components
            day_of_year = date.dayofyear
            year_progress = date_idx / len(dates)
            
            # Trend component
            deposit_trend_factor = 1 + deposit_trend * year_progress
            loan_trend_factor = 1 + loan_usage_trend * year_progress
            
            # Seasonal component (higher deposits at start/end of year, higher loans in middle)
            deposit_seasonal = deposit_seasonal_amp * np.sin(2 * np.pi * day_of_year / 365 + np.random.uniform(0, np.pi))
            loan_seasonal = loan_seasonal_amp * np.sin(2 * np.pi * day_of_year / 365 + np.pi + np.random.uniform(0, np.pi/2))
            
            # Random noise
            deposit_noise = np.random.normal(0, base_deposit * 0.02)
            loan_noise = np.random.normal(0, base_loan_limit * 0.015)
            
            # Calculate values
            if has_deposits:
                deposit = max(0, base_deposit * deposit_trend_factor + deposit_seasonal + deposit_noise)
            else:
                deposit = 0
                
            total_loan_limit = max(50000, base_loan_limit * loan_trend_factor)
            
            # Calculate used loan with seasonal and noise components
            used_loan_ratio = np.random.beta(2, 3)  # Beta distribution for loan utilization
            used_loan_ratio = used_loan_ratio * loan_trend_factor
            used_loan_ratio = min(used_loan_ratio, 0.95)  # Cap utilization at 95%
            
            used_loan = total_loan_limit * used_loan_ratio + loan_seasonal + loan_noise
            used_loan = max(0, used_loan)
            used_loan = min(used_loan, total_loan_limit)
            
            unused_loan = total_loan_limit - used_loan
            
            # Add missing values occasionally (NaN, zeros, etc.)
            if np.random.random() < 0.03:  # 3% chance of missing deposit
                deposit = np.nan
            if np.random.random() < 0.02:  # 2% chance of missing loan data
                used_loan = np.nan
                unused_loan = np.nan
            
            # Some companies have periods of zero deposits or loans
            if np.random.random() < 0.01:  # 1% chance of zero deposit
                deposit = 0
            if np.random.random() < 0.005:  # 0.5% chance of zero loans
                used_loan = 0
                unused_loan = 0
                
            # Store data
            all_companies.append(company_id)
            all_dates.append(date)
            all_deposits.append(deposit)
            all_used_loans.append(used_loan)
            all_unused_loans.append(unused_loan)
    
    # Create DataFrame
    df = pd.DataFrame({
        'company_id': all_companies,
        'date': all_dates,
        'deposit_balance': all_deposits,
        'used_loan': all_used_loans,
        'unused_loan': all_unused_loans
    })
    
    return df

def clean_financial_data(df, non_zero_threshold=0.8):
    """
    Clean financial data by:
    1. Removing NaN, inf values
    2. Keeping only companies where at least non_zero_threshold (e.g., 80%) 
       of their deposit, used loan, and unused loan data are non-zero.
    
    Args:
        df (pd.DataFrame): Input financial dataframe
        non_zero_threshold (float): Threshold for acceptable non-zero data (0.0-1.0)
        
    Returns:
        pd.DataFrame: Cleaned dataframe
    """
    print(f"Original data shape: {df.shape}")
    
    # Replace inf with NaN
    df = df.replace([np.inf, -np.inf], np.nan)
    
    # Calculate the percentage of non-zero, non-NaN values for each company
    company_stats = {}
    
    for company in df['company_id'].unique():
        company_data = df[df['company_id'] == company]
        
        # Calculate percentages of valid data (non-zero, non-NaN)
        deposit_valid = (company_data['deposit_balance'] > 0).mean()
        used_loan_valid = (company_data['used_loan'] > 0).mean()
        unused_loan_valid = (company_data['unused_loan'] > 0).mean()
        
        # Calculate overall validity score
        company_stats[company] = {
            'deposit_valid': deposit_valid,
            'used_loan_valid': used_loan_valid,
            'unused_loan_valid': unused_loan_valid,
            'overall_valid': (deposit_valid + used_loan_valid + unused_loan_valid) / 3
        }
    
    # Select companies that meet the threshold for all metrics
    valid_companies = [
        company for company, stats in company_stats.items()
        if (stats['deposit_valid'] >= non_zero_threshold or pd.isna(stats['deposit_valid'])) and 
           stats['used_loan_valid'] >= non_zero_threshold and 
           stats['unused_loan_valid'] >= non_zero_threshold
    ]
    
    # Filter the dataframe
    df_clean = df[df['company_id'].isin(valid_companies)].copy()
    
    # Fill remaining NaNs with interpolation within each company
    for company in df_clean['company_id'].unique():
        idx = df_clean['company_id'] == company
        for col in ['deposit_balance', 'used_loan', 'unused_loan']:
            df_clean.loc[idx, col] = df_clean.loc[idx, col].interpolate(method='linear', limit_direction='both')
    
    # If any NaNs remain, fill with 0 for deposits and median for loans
    df_clean['deposit_balance'] = df_clean['deposit_balance'].fillna(0)
    for col in ['used_loan', 'unused_loan']:
        median_value = df_clean[col].median()
        df_clean[col] = df_clean[col].fillna(median_value)
    
    print(f"Cleaned data shape: {df_clean.shape}")
    print(f"Kept {len(valid_companies)} out of {df['company_id'].nunique()} companies")
    
    return df_clean

def calculate_features(df):
    """
    Calculate key financial features and metrics from the raw data.
    
    Args:
        df (pd.DataFrame): Input financial dataframe
        
    Returns:
        pd.DataFrame: DataFrame with additional calculated features
    """
    # Create a copy to avoid modifying the original dataframe
    df_features = df.copy()
    
    # Calculate loan utilization rate: used_loan / (used_loan + unused_loan)
    df_features['loan_utilization'] = df_features['used_loan'] / (df_features['used_loan'] + df_features['unused_loan'])
    
    # Calculate additional metrics
    df_features['total_loan'] = df_features['used_loan'] + df_features['unused_loan']
    df_features['deposit_to_loan_ratio'] = df_features['deposit_balance'] / df_features['total_loan']
    
    # Ensure no infinity or NaN
    df_features = df_features.replace([np.inf, -np.inf], np.nan)
    
    # Fill NaNs for calculated metrics
    for col in ['loan_utilization', 'deposit_to_loan_ratio']:
        # Fill within company first using interpolation
        for company in df_features['company_id'].unique():
            mask = df_features['company_id'] == company
            df_features.loc[mask, col] = df_features.loc[mask, col].interpolate(method='linear', limit_direction='both')
        
        # For any remaining NaNs, fill with median of the entire column
        median_val = df_features[col].median()
        df_features[col] = df_features[col].fillna(median_val)
    
    return df_features

def segment_companies(df):
    """
    Segment companies into High, Medium, and Low categories
    based on loan utilization, used loan amount, and deposits.
    
    Args:
        df (pd.DataFrame): Input financial dataframe with calculated features
        
    Returns:
        pd.DataFrame: DataFrame with added segment columns
    """
    df_segmented = df.copy()
    
    # Calculate company averages for key metrics
    company_avgs = df_segmented.groupby('company_id').agg({
        'loan_utilization': 'mean',
        'used_loan': 'mean',
        'deposit_balance': 'mean'
    }).reset_index()
    
    # Define segments using quantiles (33% and 66%)
    for col in ['loan_utilization', 'used_loan', 'deposit_balance']:
        q33 = company_avgs[col].quantile(0.33)
        q66 = company_avgs[col].quantile(0.66)
        
        # Create segment column
        segment_col = f"{col}_segment"
        
        # Assign segments
        company_avgs[segment_col] = 'Medium'
        company_avgs.loc[company_avgs[col] <= q33, segment_col] = 'Low'
        company_avgs.loc[company_avgs[col] >= q66, segment_col] = 'High'
    
    # Merge segments back to the main dataframe
    segment_cols = ['company_id'] + [f"{col}_segment" for col in ['loan_utilization', 'used_loan', 'deposit_balance']]
    df_segmented = df_segmented.merge(company_avgs[segment_cols], on='company_id', how='left')
    
    return df_segmented

def analyze_correlations(df, corr_threshold=0.7):
    """
    Analyze correlations between loan utilization and deposits for each company.
    Flag companies where metrics are highly correlated or anti-correlated.
    
    Args:
        df (pd.DataFrame): Input financial dataframe with calculated features
        corr_threshold (float): Correlation threshold magnitude (0.0-1.0)
        
    Returns:
        pd.DataFrame: DataFrame of companies with correlation info
        pd.DataFrame: Updated input dataframe with correlation flags
    """
    df_corr = df.copy()
    correlation_results = []
    
    for company in df_corr['company_id'].unique():
        company_data = df_corr[df_corr['company_id'] == company]
        
        # Calculate correlation between loan utilization and deposit_balance
        try:
            corr, p_value = pearsonr(
                company_data['loan_utilization'].values, 
                company_data['deposit_balance'].values
            )
            
            # Determine if correlation is significant
            is_significant = p_value < 0.05
            
            # Determine correlation flag
            if abs(corr) >= corr_threshold and is_significant:
                if corr > 0:
                    corr_flag = 'Highly Positively Correlated'
                else:
                    corr_flag = 'Highly Negatively Correlated'
            else:
                corr_flag = 'Not Significantly Correlated'
                
            correlation_results.append({
                'company_id': company,
                'correlation': corr,
                'p_value': p_value,
                'is_significant': is_significant,
                'correlation_flag': corr_flag
            })
        except:
            # Handle cases with insufficient data
            correlation_results.append({
                'company_id': company,
                'correlation': np.nan,
                'p_value': np.nan,
                'is_significant': False,
                'correlation_flag': 'Insufficient Data'
            })
    
    # Create correlation dataframe
    corr_df = pd.DataFrame(correlation_results)
    
    # Add flags to original dataframe
    df_corr = df_corr.merge(corr_df[['company_id', 'correlation_flag']], on='company_id', how='left')
    
    return corr_df, df_corr

def detect_risks(df, lookback_periods={'3m': 90, '6m': 180, '12m': 365}):
    """
    Detect risk patterns and create early warning signals based on multiple rules.
    
    Rules include:
    1. Loan utilization increasing but deposits decreasing/stagnant (3m, 6m, 12m periods)
    2. Loan utilization steady but deposits decreasing
    3. Loan decreasing but deposits decreasing faster
    4. Deposit volatility increasing
    5. Sudden spikes in loan utilization
    6. Deposit-to-loan ratio decreasing consistently
    7. Seasonal deposit pattern breaking
    8. Utilization approaching credit limit
    9. Increasing deposit withdrawal frequency
    10. Consistent decrease in average deposit balance
    11. Unstable deposit pattern with increasing loan utilization
    12. Loan utilization increasing while deposit-to-loan ratio decreasing
    13. Accelerating loan utilization growth rate
    14. Deposit concentration risk (deposits largely coming in single transactions)
    15. Deposit balance falling below historical low while loan utilization high
    
    Args:
        df (pd.DataFrame): Input financial dataframe with calculated features
        lookback_periods (dict): Dictionary of lookback periods with names and days
        
    Returns:
        pd.DataFrame: DataFrame with added risk flags and reasons
    """
    df_risk = df.copy()
    
    # Add a date column if it's a datetime object
    if not pd.api.types.is_datetime64_any_dtype(df_risk['date']):
        df_risk['date'] = pd.to_datetime(df_risk['date'])
    
    # Sort by company and date to ensure correct trend analysis
    df_risk = df_risk.sort_values(['company_id', 'date'])
    
    # Add columns for risk flags and reasons
    df_risk['risk_flag'] = False
    df_risk['risk_reasons'] = ''
    df_risk['risk_level'] = 'Low'  # Default risk level
    
    # Process each company individually
    for company in df_risk['company_id'].unique():
        company_data = df_risk[df_risk['company_id'] == company].copy()
        company_data = company_data.sort_values('date')
        
        # Skip if less than 6 months of data
        if len(company_data) < 180:
            continue
            
        # Calculate company-specific metrics for risk assessment
        # 1. Historical min/max values
        hist_min_deposit = company_data['deposit_balance'].min()
        hist_max_deposit = company_data['deposit_balance'].max()
        hist_min_util = company_data['loan_utilization'].min()
        hist_max_util = company_data['loan_utilization'].max()
        
        # Store original index before resetting for later reference
        original_index_map = dict(zip(range(len(company_data)), company_data.index))
        # Reset index for position-based operations
        company_data_reset = company_data.reset_index(drop=True)
        
        # 2. Calculate rolling statistics for deposits and loan utilization
        company_data_reset['deposit_3m_avg'] = company_data_reset['deposit_balance'].rolling(window=90, min_periods=30).mean()
        company_data_reset['deposit_6m_avg'] = company_data_reset['deposit_balance'].rolling(window=180, min_periods=60).mean()
        company_data_reset['deposit_3m_std'] = company_data_reset['deposit_balance'].rolling(window=90, min_periods=30).std()
        company_data_reset['deposit_6m_std'] = company_data_reset['deposit_balance'].rolling(window=180, min_periods=60).std()
        
        company_data_reset['util_3m_avg'] = company_data_reset['loan_utilization'].rolling(window=90, min_periods=30).mean()
        company_data_reset['util_6m_avg'] = company_data_reset['loan_utilization'].rolling(window=180, min_periods=60).mean()
        company_data_reset['util_3m_std'] = company_data_reset['loan_utilization'].rolling(window=90, min_periods=30).std()
        
        # 3. Calculate deposit changes (day-to-day)
        company_data_reset['deposit_change'] = company_data_reset['deposit_balance'].diff()
        company_data_reset['deposit_pct_change'] = company_data_reset['deposit_balance'].pct_change()
        
        # 4. Identify deposit withdrawals (negative changes)
        company_data_reset['withdrawal'] = company_data_reset['deposit_change'] < 0
        company_data_reset['withdrawal_30d_count'] = company_data_reset['withdrawal'].rolling(window=30, min_periods=10).sum()
        
        # 5. Calculate utilization change rates
        company_data_reset['util_change'] = company_data_reset['loan_utilization'].diff()
        company_data_reset['util_change_3m_avg'] = company_data_reset['util_change'].rolling(window=90, min_periods=30).mean()
        company_data_reset['util_acceleration'] = company_data_reset['util_change'].diff()  # Second derivative
        
        # Fill NaN values created by rolling calculations
        company_data_reset = company_data_reset.fillna(method='bfill').fillna(method='ffill')
        
        # Get indices for this company's data
        company_indices = company_data.index
        
        # Create a reset index version of company_data for proper positional indexing
        company_data_reset = company_data.reset_index(drop=True)
        
        # For each date, analyze risk patterns
        for pos in range(len(company_data_reset)):
            row = company_data_reset.iloc[pos]
            current_date = row['date']
            risk_reasons = []
            
            # Get the original index for this row
            original_idx = original_index_map[pos]
            
            # Skip first 180 days (need history for meaningful analysis)
            if pos < 180:
                continue
            
            # -- CORE RISK RULES --
            
            # Check different lookback periods for standard risk patterns
            for period_name, days in lookback_periods.items():
                # Get data for lookback period
                period_start_date = current_date - pd.Timedelta(days=days)
                period_data = company_data[
                    (company_data['date'] >= period_start_date) & 
                    (company_data['date'] <= current_date)
                ]
                
                # Skip if insufficient data for this period
                if len(period_data) < days * 0.7:  # Require at least 70% of days
                    continue
                
                # Get start and end values
                start_util = period_data.iloc[0]['loan_utilization']
                end_util = period_data.iloc[-1]['loan_utilization']
                start_deposit = period_data.iloc[0]['deposit_balance']
                end_deposit = period_data.iloc[-1]['deposit_balance']
                
                # Calculate trends
                util_change = end_util - start_util
                deposit_change_pct = (end_deposit - start_deposit) / start_deposit if start_deposit > 0 else 0
                
                # Rule 1: Loan utilization increasing but deposits decreasing/stagnant
                if util_change > 0.05 and deposit_change_pct < 0.02:
                    reason = f"RISK: {period_name} - Loan utilization increased by {util_change:.1%} " + \
                             f"while deposits {deposit_change_pct:.1%}"
                    risk_reasons.append(reason)
                
                # Rule 2: Loan utilization steady but deposits decreasing
                if abs(util_change) < 0.03 and deposit_change_pct < -0.05:
                    reason = f"RISK: {period_name} - Deposits decreased by {-deposit_change_pct:.1%} " + \
                             f"while loan utilization remained steady"
                    risk_reasons.append(reason)
                
                # Rule 3: Loan decreasing but deposits decreasing faster
                loan_change_pct = (period_data.iloc[-1]['total_loan'] - period_data.iloc[0]['total_loan']) / period_data.iloc[0]['total_loan'] if period_data.iloc[0]['total_loan'] > 0 else 0
                if loan_change_pct < -0.03 and deposit_change_pct < loan_change_pct * 1.5:
                    reason = f"RISK: {period_name} - Deposits decreased {-deposit_change_pct:.1%}, " + \
                             f"faster than loan decrease {-loan_change_pct:.1%}"
                    risk_reasons.append(reason)
            
            # -- ADDITIONAL ADVANCED RISK RULES --
            
            # Rule 4: Deposit volatility increasing significantly
            current_deposit_std_3m = row['deposit_3m_std']
            if pos >= 90:
                past_deposit_std_3m = company_data_reset.iloc[pos-90]['deposit_3m_std']
            else:
                past_deposit_std_3m = current_deposit_std_3m
            
            if current_deposit_std_3m > past_deposit_std_3m * 1.5 and current_deposit_std_3m > row['deposit_balance'] * 0.1:
                reason = "RISK: Increasing deposit volatility - Indicates potential cash flow instability"
                risk_reasons.append(reason)
            
            # Rule 5: Sudden spike in loan utilization
            if row['loan_utilization'] > row['util_3m_avg'] + 2 * row['util_3m_std']:
                reason = f"RISK: Unusual spike in loan utilization - Current: {row['loan_utilization']:.1%}, " + \
                         f"3m Avg: {row['util_3m_avg']:.1%}"
                risk_reasons.append(reason)
            
            # Rule 6: Deposit-to-loan ratio decreasing consistently
            if pos >= 180:
                ratio_6m_ago = company_data_reset.iloc[pos-180]['deposit_to_loan_ratio']
                ratio_3m_ago = company_data_reset.iloc[pos-90]['deposit_to_loan_ratio']
            
                if ratio_6m_ago > ratio_3m_ago > row['deposit_to_loan_ratio'] and row['deposit_to_loan_ratio'] < 0.8 * ratio_6m_ago:
                    reason = "RISK: Consistent decline in deposit-to-loan ratio over 6 months"
                    risk_reasons.append(reason)
            
            # Rule 7: Seasonal deposit pattern breaking
            if len(company_data_reset) >= 365:  # Need at least a year of history
                # Get data from one year ago for same season (±15 days)
                season_start_date = current_date - pd.Timedelta(days=365) - pd.Timedelta(days=15)
                season_end_date = current_date - pd.Timedelta(days=365) + pd.Timedelta(days=15)
                
                # Use date-based filtering instead of index-based
                seasonal_data = company_data[
                    (company_data['date'] >= season_start_date) & 
                    (company_data['date'] <= season_end_date)
                ]
                
                if len(seasonal_data) > 0:
                    # Compare current values to seasonal expectations
                    seasonal_deposit_avg = seasonal_data['deposit_balance'].mean()
                    current_deposit = row['deposit_balance']
                    
                    if current_deposit < seasonal_deposit_avg * 0.7 and seasonal_deposit_avg > 0:
                        reason = "RISK: Seasonal - Deposits significantly lower than same period last year"
                        risk_reasons.append(reason)
            
            # Rule 8: Utilization approaching credit limit
            if row['loan_utilization'] > 0.9:
                reason = "RISK: Critical - Loan utilization above 90% of credit limit"
                risk_reasons.append(reason)
                
            # Rule 9: Increasing deposit withdrawal frequency
            withdrawal_count_30d = row['withdrawal_30d_count']
            if pos >= 30:
                past_withdrawal_count = company_data_reset.iloc[pos-30]['withdrawal_30d_count']
            else:
                past_withdrawal_count = withdrawal_count_30d
            
            if withdrawal_count_30d > past_withdrawal_count * 1.5 and withdrawal_count_30d > 15:  # More than 15 days with withdrawals in 30 days
                reason = "RISK: Increasing frequency of deposit withdrawals"
                risk_reasons.append(reason)
                
            # Rule 10: Consistent decrease in average deposit balance
            if pos >= 90:
                past_deposit_avg = company_data_reset.iloc[pos-90]['deposit_6m_avg']
                if row['deposit_6m_avg'] < past_deposit_avg * 0.85:  # 15% drop in 3 months
                    reason = "RISK: Consistent decline in average deposit balance over 3 months"
                    risk_reasons.append(reason)
                
            # Rule 11: Unstable deposit pattern with increasing loan utilization
            if row['deposit_3m_std'] / (row['deposit_3m_avg'] + 1e-10) > 0.2 and row['util_change_3m_avg'] > 0:
                reason = "RISK: Unstable deposits with increasing loan utilization"
                risk_reasons.append(reason)
                
            # Rule 12: Loan utilization increasing while deposit-to-loan ratio decreasing
            if pos >= 90:
                past_ratio = company_data_reset.iloc[pos-90]['deposit_to_loan_ratio']
                if row['util_change_3m_avg'] > 0 and row['deposit_to_loan_ratio'] < past_ratio * 0.9:
                    reason = "RISK: Increasing loan utilization with declining deposit coverage"
                    risk_reasons.append(reason)
                
            # Rule 13: Accelerating loan utilization growth rate
            if row['util_acceleration'] > 0 and row['util_change'] > 0 and row['loan_utilization'] > 0.7:
                reason = "RISK: Accelerating loan utilization growth rate with high current utilization"
                risk_reasons.append(reason)
                
            # Rule 14: Deposit concentration risk
            # Get data from the last 30 days using datetime-based selection
            thirty_days_ago = current_date - pd.Timedelta(days=30)
            recent_data = company_data[(company_data['date'] >= thirty_days_ago) & (company_data['date'] <= current_date)]
            recent_deposits = recent_data['deposit_change']
            positive_deposits = recent_deposits[recent_deposits > 0]
            
            if len(positive_deposits) > 0:
                max_deposit = positive_deposits.max()
                avg_deposit = positive_deposits.mean()
                if max_deposit > avg_deposit * 5 and max_deposit > row['deposit_balance'] * 0.3:
                    reason = "RISK: Deposit concentration - Large portion of deposits from single transaction"
                    risk_reasons.append(reason)
                    
            # Rule 15: Deposit balance falling below historical low while loan utilization high
            if row['deposit_balance'] < hist_min_deposit * 1.1 and row['loan_utilization'] > hist_max_util * 0.9:
                reason = "RISK: Critical - Deposit near historical low while loan utilization near maximum"
                risk_reasons.append(reason)
            
            # Update risk flag and reasons in the original dataframe
            if risk_reasons:
                df_risk.at[original_idx, 'risk_flag'] = True
                df_risk.at[original_idx, 'risk_reasons'] = "; ".join(risk_reasons)
                
                # Assign risk level based on number and severity of risk factors
                num_risks = len(risk_reasons)
                high_severity_keywords = ['Critical', 'significantly lower', 'above 90%']
                
                # Count high severity risk factors
                high_severity_count = sum(1 for reason in risk_reasons if any(keyword in reason for keyword in high_severity_keywords))
                
                if high_severity_count > 0 or num_risks >= 3:
                    df_risk.at[original_idx, 'risk_level'] = 'High'
                elif num_risks >= 1:
                    df_risk.at[original_idx, 'risk_level'] = 'Medium'
    
    # Create a risk count column (number of risk reasons)
    df_risk['risk_count'] = df_risk['risk_reasons'].apply(
        lambda x: len(x.split(';')) if x else 0
    )
    
    return df_risk

def cluster_companies(df, n_clusters=4):
    """
    Cluster companies based on their financial behavior patterns.
    
    Args:
        df (pd.DataFrame): Input financial dataframe with calculated features
        n_clusters (int): Number of clusters to create
        
    Returns:
        pd.DataFrame: Company info with cluster assignments
        pd.DataFrame: Original dataframe with cluster assignments
        pd.DataFrame: Cluster summary statistics
    """
    # Calculate company-level features for clustering
    company_features = df.groupby('company_id').agg({
        'loan_utilization': ['mean', 'std', 'max', 'min'],
        'deposit_balance': ['mean', 'std', 'max', 'min'],
        'used_loan': ['mean', 'std', 'max', 'min'],
        'unused_loan': ['mean', 'std'],
        'deposit_to_loan_ratio': ['mean', 'std'],
        'risk_flag': 'mean'  # Percentage of days with risk flags
    })
    
    # Flatten column names
    company_features.columns = ['_'.join(col).strip() for col in company_features.columns.values]
    
    # Calculate additional features
    company_features['deposit_volatility'] = company_features['deposit_balance_std'] / company_features['deposit_balance_mean']
    company_features['loan_volatility'] = company_features['used_loan_std'] / company_features['used_loan_mean']
    company_features['utilization_range'] = company_features['loan_utilization_max'] - company_features['loan_utilization_min']
    
    # Handle NaN and inf values
    company_features = company_features.replace([np.inf, -np.inf], np.nan)
    company_features = company_features.fillna(company_features.median())
    
    # Select features for clustering
    cluster_features = [
        'loan_utilization_mean', 'loan_utilization_std', 
        'deposit_balance_mean', 'deposit_volatility',
        'used_loan_mean', 'loan_volatility',
        'deposit_to_loan_ratio_mean', 'utilization_range',
        'risk_flag_mean'
    ]
    
    # Scale features
    X = company_features[cluster_features].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Apply KMeans clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X_scaled)
    
    # Add cluster labels to company features
    company_features['cluster'] = clusters
    
    # Create a mapping from company to cluster
    company_cluster_map = company_features['cluster'].to_dict()
    
    # Add cluster to original dataframe
    df_clustered = df.copy()
    df_clustered['cluster'] = df_clustered['company_id'].map(company_cluster_map)
    
    # Create cluster summary
    cluster_summary = company_features.groupby('cluster').agg({
        'loan_utilization_mean': 'mean',
        'deposit_balance_mean': 'mean',
        'used_loan_mean': 'mean',
        'deposit_to_loan_ratio_mean': 'mean',
        'risk_flag_mean': 'mean'
    }).reset_index()
    
    # Add cluster names based on characteristics
    cluster_profiles = []
    for _, row in cluster_summary.iterrows():
        profile = ""
        if row['loan_utilization_mean'] > cluster_summary['loan_utilization_mean'].median():
            profile += "High Utilization, "
        else:
            profile += "Low Utilization, "
            
        if row['deposit_balance_mean'] > cluster_summary['deposit_balance_mean'].median():
            profile += "High Deposits, "
        else:
            profile += "Low Deposits, "
            
        if row['risk_flag_mean'] > cluster_summary['risk_flag_mean'].median():
            profile += "High Risk"
        else:
            profile += "Low Risk"
            
        cluster_profiles.append(profile)
        
    cluster_summary['profile'] = cluster_profiles
    
    # Return company features with cluster assignments and the updated original dataframe
    return company_features.reset_index(), df_clustered, cluster_summary

def perform_cohort_analysis(df, time_period='M'):
    """
    Perform cohort analysis based on specified time period.
    Tracks how clusters evolve over time.
    
    Args:
        df (pd.DataFrame): Input financial dataframe with cluster assignments
        time_period (str): Pandas time frequency string ('M' for month, 'Q' for quarter)
        
    Returns:
        pd.DataFrame: Cohort analysis results
        pd.DataFrame: Transition data between clusters
        pd.DataFrame: Cluster sizes over time
        pd.DataFrame: Transition counts between clusters
    """
    # Ensure date is datetime
    df_cohort = df.copy()
    df_cohort['date'] = pd.to_datetime(df_cohort['date'])
    
    # Create a period column based on the specified time period
    df_cohort['period'] = df_cohort['date'].dt.to_period(time_period)
    
    # Group by period and company to get the most common cluster for each company in each period
    cohort_data = []
    
    for company in df_cohort['company_id'].unique():
        company_data = df_cohort[df_cohort['company_id'] == company]
        
        # Group by period
        for period, period_data in company_data.groupby('period'):
            # Get the most common cluster for this company in this period
            cluster = period_data['cluster'].mode().iloc[0]
            
                            # Calculate average metrics for this company in this period
            avg_metrics = period_data.agg({
                'loan_utilization': 'mean',
                'deposit_balance': 'mean',
                'used_loan': 'mean',
                'deposit_to_loan_ratio': 'mean',
                'risk_flag': 'mean'
            })
            
            # Add to cohort data
            cohort_data.append({
                'company_id': company,
                'period': period,
                'cluster': cluster,
                'loan_utilization': avg_metrics['loan_utilization'],
                'deposit_balance': avg_metrics['deposit_balance'],
                'used_loan': avg_metrics['used_loan'],
                'deposit_to_loan_ratio': avg_metrics['deposit_to_loan_ratio'],
                'risk_percentage': avg_metrics['risk_flag'] * 100
            })
    
    # Create cohort dataframe
    cohort_df = pd.DataFrame(cohort_data)
    
    # Create transition matrix to see how companies move between clusters over time
    transitions = []
    
    for company in df_cohort['company_id'].unique():
        # Get the clusters for this company over time
        company_clusters = cohort_df[cohort_df['company_id'] == company].sort_values('period')
        
        if len(company_clusters) < 2:
            continue
            
        # Extract the sequence of clusters
        cluster_sequence = company_clusters['cluster'].tolist()
        
        # Create transitions (from one period to the next)
        for i in range(len(cluster_sequence) - 1):
            from_cluster = cluster_sequence[i]
            to_cluster = cluster_sequence[i + 1]
            period = company_clusters.iloc[i]['period']
            next_period = company_clusters.iloc[i + 1]['period']
            
            transitions.append({
                'company_id': company,
                'from_period': period,
                'to_period': next_period,
                'from_cluster': from_cluster,
                'to_cluster': to_cluster
            })
    
    # Create transitions dataframe
    transitions_df = pd.DataFrame(transitions)
    
    # Calculate cluster sizes over time
    cluster_sizes = cohort_df.groupby(['period', 'cluster']).size().unstack().fillna(0)
    
    # Calculate transition counts
    if len(transitions_df) > 0:
        transition_counts = transitions_df.groupby(['from_cluster', 'to_cluster']).size().unstack().fillna(0)
    else:
        transition_counts = pd.DataFrame()
    
    return cohort_df, transitions_df, cluster_sizes, transition_counts

def plot_waterfall_chart(df):
    """
    Create a waterfall chart showing client segmentation.
    
    Args:
        df (pd.DataFrame): Input financial dataframe with segments
        
    Returns:
        matplotlib.figure.Figure: The created waterfall chart figure
    """
    # Count total companies
    total_companies = df['company_id'].nunique()
    
    # Count companies with both loans and deposits
    companies_with_both = df.groupby('company_id').agg({
        'deposit_balance': lambda x: (x > 0).any(),
        'used_loan': lambda x: (x > 0).any()
    })
    
    companies_with_both = companies_with_both[(companies_with_both['deposit_balance']) & 
                                             (companies_with_both['used_loan'])]
    
    companies_with_both_count = len(companies_with_both)
    
    # Get segments
    latest_data = df.sort_values('date').groupby('company_id').last().reset_index()
    
    # Count companies in each loan utilization segment
    util_segments = latest_data.groupby('loan_utilization_segment').size()
    
    # Count companies in each used loan segment
    loan_segments = latest_data.groupby('used_loan_segment').size()
    
    # Count companies in each deposit segment
    deposit_segments = latest_data.groupby('deposit_balance_segment').size()
    
    # Create plot
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Define width and spacing
    width = 0.7
    
    # Waterfall chart data
    labels = ['Total Companies', 'With Both\nLoans & Deposits']
    values = [total_companies, companies_with_both_count]
    
    # Add segment labels and values
    for segment, count in util_segments.items():
        labels.append(f'Utilization\n{segment}')
        values.append(count)
        
    for segment, count in loan_segments.items():
        labels.append(f'Loan Amount\n{segment}')
        values.append(count)
        
    for segment, count in deposit_segments.items():
        labels.append(f'Deposit\n{segment}')
        values.append(count)
    
    # Create the bars
    bars = ax.bar(labels, values, width=width, edgecolor='black')
    
    # Set colors
    colors = ['#5DA5DA', '#60BD68']
    util_colors = {'High': '#F15854', 'Medium': '#FAA43A', 'Low': '#DECF3F'}
    loan_colors = {'High': '#B276B2', 'Medium': '#8CD17D', 'Low': '#4D4D4D'}
    deposit_colors = {'High': '#1E88E5', 'Medium': '#FFC107', 'Low': '#D81B60'}
    
    # Apply colors
    for i, bar in enumerate(bars):
        if i < 2:
            bar.set_color(colors[i])
        elif 'Utilization' in labels[i]:
            segment = labels[i].split('\n')[1]
            bar.set_color(util_colors.get(segment, '#CCCCCC'))
        elif 'Loan Amount' in labels[i]:
            segment = labels[i].split('\n')[1]
            bar.set_color(loan_colors.get(segment, '#CCCCCC'))
        elif 'Deposit' in labels[i]:
            segment = labels[i].split('\n')[1]
            bar.set_color(deposit_colors.get(segment, '#CCCCCC'))
    
    # Add values on top of bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{int(height)}', ha='center', va='bottom')
    
    # Customize plot
    ax.set_title('Client Segmentation Analysis', fontsize=16)
    ax.set_ylabel('Number of Companies', fontsize=14)
    
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    return fig

def plot_risk_company(df, company_id, window_size=180):
    """
    Plot loan utilization and deposit trends for a risk-flagged company.
    
    Args:
        df (pd.DataFrame): Input financial dataframe with risk flags
        company_id (str): ID of the company to plot
        window_size (int): Number of days to include in the plot
        
    Returns:
        matplotlib.figure.Figure: The created figure with dual-axis plot
    """
    # Filter data for the specified company
    company_data = df[df['company_id'] == company_id].copy()
    
    # Sort by date
    company_data = company_data.sort_values('date')
    
    # Get the most recent data
    if len(company_data) > window_size:
        company_data = company_data.iloc[-window_size:]
    
    # Create figure with two y-axes
    fig, ax1 = plt.subplots(figsize=(14, 8))
    
    # Format dates for x-axis
    dates = company_data['date']
    
    # First y-axis: Loan Utilization
    color1 = 'tab:red'
    ax1.set_xlabel('Date', fontsize=12)
    ax1.set_ylabel('Loan Utilization (%)', color=color1, fontsize=12)
    ax1.plot(dates, company_data['loan_utilization'] * 100, color=color1, marker='o', markersize=3, label='Loan Utilization')
    ax1.tick_params(axis='y', labelcolor=color1)
    ax1.set_ylim(0, max(company_data['loan_utilization'] * 100) * 1.1)
    
    # Add second y-axis: Deposit Balance
    ax2 = ax1.twinx()
    color2 = 'tab:blue'
    ax2.set_ylabel('Deposit Balance', color=color2, fontsize=12)
    ax2.plot(dates, company_data['deposit_balance'], color=color2, marker='s', markersize=3, label='Deposit Balance')
    ax2.tick_params(axis='y', labelcolor=color2)
    
    # Format currency for second y-axis
    fmt = FuncFormatter(lambda x, _: f'${x:,.0f}')
    ax2.yaxis.set_major_formatter(fmt)
    
    # Add risk flags as vertical lines or markers
    risk_days = company_data[company_data['risk_flag'] == True]
    
    # Shade risk periods
    for idx, risk_day in risk_days.iterrows():
        ax1.axvline(x=risk_day['date'], color='red', alpha=0.1, linewidth=2)
    
    # Add labels for risk reasons
    # Group consecutive risk days with the same reason
    current_reason = None
    reason_start_date = None
    last_date = None
    
    risk_annotations = []
    
    # Sort risk days by date for proper grouping
    risk_days = risk_days.sort_values('date')
    
    for idx, risk_day in risk_days.iterrows():
        # If this is a new reason or not consecutive with the last date
        if (current_reason != risk_day['risk_reasons'] or 
            (last_date and (risk_day['date'] - last_date).days > 5)):
            # Store the previous reason if it exists
            if current_reason and reason_start_date:
                risk_annotations.append((reason_start_date, current_reason))
            
            # Start a new reason group
            current_reason = risk_day['risk_reasons']
            reason_start_date = risk_day['date']
        
        last_date = risk_day['date']
    
    # Add the last reason if it exists
    if current_reason and reason_start_date:
        risk_annotations.append((reason_start_date, current_reason))
    
    # Plot risk annotations (shortened and with line breaks)
    y_positions = np.linspace(0.1, 0.9, len(risk_annotations))
    
    for i, (date, reason) in enumerate(risk_annotations):
        # Shorten and format reasons
        reason_parts = reason.split(';')
        shortened_reasons = []
        
        for part in reason_parts[:3]:  # Limit to 3 reasons
            # Extract the key part from each reason
            if 'RISK:' in part:
                key_part = part.split('RISK:')[1].strip()
                shortened_reasons.append(key_part)
        
        if reason_parts and len(reason_parts) > 3:
            shortened_reasons.append(f"+ {len(reason_parts) - 3} more")
            
        formatted_reason = '\n'.join(shortened_reasons)
        
        # Calculate y position in axis coordinates
        y_pos = y_positions[i % len(y_positions)]
        
        # Draw an arrow from the date to the text
        ax1.annotate(
            formatted_reason,
            xy=(date, 0),  # Base of the arrow on the date
            xytext=(date, y_pos * max(company_data['loan_utilization'] * 100)),  # Text position
            arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=.2", color='darkred'),
            bbox=dict(boxstyle="round,pad=0.3", fc="lightyellow", ec="darkred", alpha=0.8),
            ha='left', va='center',
            fontsize=8
        )
    
    # Add title and legend
    title = f"Risk Analysis for {company_id} - "
    if len(risk_days) > 0:
        risk_level = risk_days['risk_level'].iloc[-1]
        title += f"{risk_level} Risk Level"
    else:
        title += "No Risk Detected"
        
    plt.title(title, fontsize=14)
    
    # Add custom legend with both axes
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
    
    # Format x-axis dates
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.xticks(rotation=45)
    
    # Add grid
    ax1.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    
    return fig

def plot_cluster_analysis(df_clusters, cluster_summary):
    """
    Create a cluster analysis visualization showing the key characteristics
    of each identified cluster.
    
    Args:
        df_clusters (pd.DataFrame): Dataframe with company-level features and cluster assignments
        cluster_summary (pd.DataFrame): Summary statistics for each cluster
        
    Returns:
        matplotlib.figure.Figure: The created cluster analysis figure
    """
    # Create a figure with subplots
    fig, axs = plt.subplots(2, 2, figsize=(16, 12))
    
    # Flatten for easier indexing
    axs = axs.flatten()
    
    # 1. Plot average loan utilization vs. deposit balance by cluster
    scatter = axs[0].scatter(
        df_clusters['loan_utilization_mean'],
        df_clusters['deposit_balance_mean'],
        c=df_clusters['cluster'],
        cmap='viridis',
        alpha=0.7,
        s=100
    )
    
    # Add cluster centroids
    for cluster in cluster_summary['cluster']:
        cluster_data = cluster_summary[cluster_summary['cluster'] == cluster]
        axs[0].scatter(
            cluster_data['loan_utilization_mean'],
            cluster_data['deposit_balance_mean'],
            s=300,
            marker='*',
            c='red',
            label=f"Cluster {cluster}: {cluster_data['profile'].values[0]}"
        )
    
    axs[0].set_xlabel('Average Loan Utilization Rate', fontsize=12)
    axs[0].set_ylabel('Average Deposit Balance', fontsize=12)
    axs[0].set_title('Clusters by Loan Utilization and Deposit Balance', fontsize=14)
    axs[0].legend()
    axs[0].grid(True, linestyle='--', alpha=0.7)
    
    # Format y-axis as currency
    fmt = FuncFormatter(lambda x, _: f'${x:,.0f}')
    axs[0].yaxis.set_major_formatter(fmt)
    
    # 2. Plot loan volatility vs. deposit volatility by cluster
    scatter = axs[1].scatter(
        df_clusters['loan_volatility'],
        df_clusters['deposit_volatility'],
        c=df_clusters['cluster'],
        cmap='viridis',
        alpha=0.7,
        s=100
    )
    
    axs[1].set_xlabel('Loan Amount Volatility', fontsize=12)
    axs[1].set_ylabel('Deposit Balance Volatility', fontsize=12)
    axs[1].set_title('Clusters by Financial Volatility Patterns', fontsize=14)
    axs[1].grid(True, linestyle='--', alpha=0.7)
    
    # Add colorbar for cluster identification
    cbar = plt.colorbar(scatter, ax=axs[1])
    cbar.set_label('Cluster')
    
    # 3. Bar chart of risk percentage by cluster
    cluster_summary.sort_values('risk_flag_mean', ascending=False).plot(
        x='cluster', 
        y='risk_flag_mean', 
        kind='bar',
        ax=axs[2],
        color='salmon'
    )
    
    axs[2].set_xlabel('Cluster', fontsize=12)
    axs[2].set_ylabel('Risk Percentage', fontsize=12)
    axs[2].set_title('Risk Percentage by Cluster', fontsize=14)
    
    # Format y-axis as percentage
    axs[2].yaxis.set_major_formatter(FuncFormatter(lambda y, _: f'{y:.1%}'))
    
    # Add labels on bars
    for p in axs[2].patches:
        axs[2].annotate(
            f'{p.get_height():.1%}',
            (p.get_x() + p.get_width() / 2., p.get_height()),
            ha='center', va='bottom',
            fontsize=10
        )
    
    # 4. Radar chart showing cluster profiles
    # Prepare the data for the radar chart
    categories = ['Loan Utilization', 'Deposit Balance', 'Used Loan', 'Deposit/Loan Ratio', 'Risk']
    
    # Normalize the values for the radar chart
    radar_data = cluster_summary[['loan_utilization_mean', 'deposit_balance_mean', 
                                'used_loan_mean', 'deposit_to_loan_ratio_mean', 
                                'risk_flag_mean']].copy()
    
    # Normalize each column to 0-1 scale
    for col in radar_data.columns:
        min_val = radar_data[col].min()
        max_val = radar_data[col].max()
        if max_val > min_val:
            radar_data[col] = (radar_data[col] - min_val) / (max_val - min_val)
        else:
            radar_data[col] = 0.5  # Default if all values are the same
    
    # Set up the radar chart
    angles = np.linspace(0, 2*np.pi, len(categories), endpoint=False)
    angles = np.concatenate((angles, [angles[0]]))  # Close the loop
    
    # Create subplot with polar projection
    ax_radar = plt.subplot(2, 2, 4, polar=True)
    
    # Plot each cluster
    for i, row in radar_data.iterrows():
        values = row.values.flatten().tolist()
        values += [values[0]]  # Close the loop
        ax_radar.plot(angles, values, linewidth=2, label=f"Cluster {cluster_summary.iloc[i]['cluster']}")
        ax_radar.fill(angles, values, alpha=0.1)
    
    # Set category labels
    ax_radar.set_xticks(angles[:-1])
    ax_radar.set_xticklabels(categories)
    
    # Add legend
    ax_radar.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
    ax_radar.set_title('Cluster Profiles (Normalized)', fontsize=14)
    
    plt.tight_layout()
    
    return fig

def plot_cohort_transitions(cluster_sizes, transition_counts):
    """
    Create visualizations for cohort analysis showing how companies 
    transition between clusters over time.
    
    Args:
        cluster_sizes (pd.DataFrame): Cluster sizes over time
        transition_counts (pd.DataFrame): Counts of transitions between clusters
        
    Returns:
        matplotlib.figure.Figure: The created transition analysis figure
    """
    fig, axs = plt.subplots(1, 2, figsize=(18, 8))
    
    # 1. Plot cluster sizes over time
    cluster_sizes.plot(
        kind='bar', 
        stacked=True,
        ax=axs[0],
        colormap='viridis'
    )
    
    axs[0].set_xlabel('Time Period', fontsize=12)
    axs[0].set_ylabel('Number of Companies', fontsize=12)
    axs[0].set_title('Cluster Sizes Over Time', fontsize=14)
    axs[0].legend(title='Cluster')
    
    # Rotate x-axis labels for better readability
    axs[0].tick_params(axis='x', rotation=45)
    
    # 2. Plot transition heatmap
    if not transition_counts.empty:
        # Create heatmap
        sns.heatmap(
            transition_counts,
            annot=True,
            fmt='.0f',
            cmap='YlGnBu',
            ax=axs[1]
        )
        
        axs[1].set_xlabel('To Cluster', fontsize=12)
        axs[1].set_ylabel('From Cluster', fontsize=12)
        axs[1].set_title('Cluster Transitions Heatmap', fontsize=14)
    else:
        axs[1].text(0.5, 0.5, 'Insufficient transition data', 
                   horizontalalignment='center', verticalalignment='center',
                   transform=axs[1].transAxes, fontsize=14)
        axs[1].set_title('Cluster Transitions', fontsize=14)
    
    plt.tight_layout()
    
    return fig

def analyze_company_risks(df):
    """
    Analyze risk distribution across companies and provide summary statistics.
    
    Args:
        df (pd.DataFrame): Input financial dataframe with risk flags
        
    Returns:
        pd.DataFrame: Risk analysis summary
        matplotlib.figure.Figure: A visualization of risk distribution
    """
    # Calculate risk statistics by company
    risk_stats = df.groupby('company_id').agg({
        'risk_flag': 'mean',
        'risk_level': lambda x: x.mode().iloc[0] if not x.mode().empty else 'Low',
        'risk_count': 'mean'
    }).reset_index()
    
    # Rename columns for clarity
    risk_stats.columns = ['company_id', 'risk_percentage', 'dominant_risk_level', 'avg_risk_count']
    
    # Calculate additional risk metrics
    risk_stats['is_high_risk'] = risk_stats['dominant_risk_level'] == 'High'
    
    # Create summary visualization
    fig, axs = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Risk level distribution
    risk_level_counts = risk_stats['dominant_risk_level'].value_counts()
    risk_level_counts.plot(
        kind='pie',
        ax=axs[0, 0],
        autopct='%1.1f%%',
        colors=['red', 'orange', 'green'],
        explode=[0.05, 0.02, 0]
    )
    axs[0, 0].set_title('Distribution of Risk Levels', fontsize=14)
    axs[0, 0].set_ylabel('')
    
    # 2. Risk percentage distribution
    axs[0, 1].hist(risk_stats['risk_percentage'], bins=20, color='orange', edgecolor='black')
    axs[0, 1].set_xlabel('Risk Percentage', fontsize=12)
    axs[0, 1].set_ylabel('Number of Companies', fontsize=12)
    axs[0, 1].set_title('Distribution of Risk Percentages', fontsize=14)
    
    # Format x-axis as percentage
    axs[0, 1].xaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x:.0%}'))
    
    # 3. Average risk count distribution
    axs[1, 0].hist(risk_stats['avg_risk_count'], bins=15, color='salmon', edgecolor='black')
    axs[1, 0].set_xlabel('Average Risk Count', fontsize=12)
    axs[1, 0].set_ylabel('Number of Companies', fontsize=12)
    axs[1, 0].set_title('Distribution of Average Risk Counts', fontsize=14)
    
    # 4. Top 10 highest risk companies
    top_risk_companies = risk_stats.sort_values('risk_percentage', ascending=False).head(10)
    bars = top_risk_companies.plot(
        x='company_id',
        y='risk_percentage',
        kind='bar',
        color='red',
        ax=axs[1, 1]
    )
    
    axs[1, 1].set_xlabel('Company ID', fontsize=12)
    axs[1, 1].set_ylabel('Risk Percentage', fontsize=12)
    axs[1, 1].set_title('Top 10 Highest Risk Companies', fontsize=14)
    
    # Format y-axis as percentage
    axs[1, 1].yaxis.set_major_formatter(FuncFormatter(lambda y, _: f'{y:.0%}'))
    
    # Add risk level as text on bars
    for i, bar in enumerate(bars.patches):
        risk_level = top_risk_companies.iloc[i]['dominant_risk_level']
        axs[1, 1].text(
            bar.get_x() + bar.get_width()/2.,
            bar.get_height() + 0.01,
            risk_level,
            ha='center', va='bottom',
            color='darkred',
            fontsize=9
        )
    
    plt.tight_layout()
    
    return risk_stats, fig

def main():
    """
    Main function to execute the entire analysis workflow.
    """
    # For reproducibility
    np.random.seed(42)
    
    # Step 1: Generate synthetic financial data
    print("Generating synthetic financial data...")
    df = generate_financial_data(n_companies=100, years=4, seed=42)
    print(f"Generated data with {df.shape[0]} rows and {df.shape[1]} columns")
    
    # Step 2: Clean the data
    print("\nCleaning financial data...")
    df_clean = clean_financial_data(df, non_zero_threshold=0.8)
    
    # Step 3: Calculate key features
    print("\nCalculating financial features...")
    df_features = calculate_features(df_clean)
    
    # Step 4: Segment companies
    print("\nSegmenting companies...")
    df_segmented = segment_companies(df_features)
    
    # Step 5: Analyze correlations
    print("\nAnalyzing correlations between loan utilization and deposits...")
    corr_df, df_corr = analyze_correlations(df_segmented, corr_threshold=0.7)
    print(f"Found {(corr_df['correlation_flag'] != 'Not Significantly Correlated').sum()} companies with significant correlations.")
    
    # Step 6: Detect risks
    print("\nDetecting risk patterns and creating early warning signals...")
    df_risk = detect_risks(df_corr)
    print(f"Identified {df_risk['risk_flag'].sum()} days with risk flags across all companies.")
    
    # Step 7: Cluster companies
    print("\nClustering companies based on behavior patterns...")
    company_clusters, df_clustered, cluster_summary = cluster_companies(df_risk, n_clusters=4)
    print(f"Created {len(cluster_summary)} distinct company clusters.")
    
    # Step 8: Perform cohort analysis
    print("\nPerforming cohort analysis...")
    cohort_df, transitions_df, cluster_sizes, transition_counts = perform_cohort_analysis(df_clustered, time_period='M')
    
    # Step 9: Create visualizations
    print("\nCreating visualizations...")
    
    # 9.1: Waterfall chart
    print("  Creating waterfall chart...")
    fig_waterfall = plot_waterfall_chart(df_segmented)
    plt.savefig('waterfall_chart.png', dpi=300, bbox_inches='tight')
    plt.close(fig_waterfall)
    
    # 9.2: Risk analysis summary
    print("  Creating risk analysis summary...")
    risk_stats, fig_risk_summary = analyze_company_risks(df_risk)
    plt.savefig('risk_summary.png', dpi=300, bbox_inches='tight')
    plt.close(fig_risk_summary)
    
    # 9.3: Plot individual risk companies
    print("  Creating individual risk company plots...")
    # Find top 3 riskiest companies
    top_risk_companies = risk_stats.sort_values('risk_percentage', ascending=False).head(3)['company_id'].tolist()
    
    for company_id in top_risk_companies:
        fig_risk_company = plot_risk_company(df_risk, company_id, window_size=180)
        plt.savefig(f'risk_company_{company_id}.png', dpi=300, bbox_inches='tight')
        plt.close(fig_risk_company)
    
    # 9.4: Cluster analysis visualization
    print("  Creating cluster analysis visualization...")
    fig_clusters = plot_cluster_analysis(company_clusters, cluster_summary)
    plt.savefig('cluster_analysis.png', dpi=300, bbox_inches='tight')
    plt.close(fig_clusters)
    
    # 9.5: Cohort transitions visualization
    print("  Creating cohort transitions visualization...")
    fig_cohort = plot_cohort_transitions(cluster_sizes, transition_counts)
    plt.savefig('cohort_transitions.png', dpi=300, bbox_inches='tight')
    plt.close(fig_cohort)
    
    print("\nAnalysis complete! All visualizations have been saved.")
    
    # Return main dataframes for further analysis
    return {
        'raw_data': df,
        'cleaned_data': df_clean,
        'risk_data': df_risk,
        'cluster_data': df_clustered,
        'company_clusters': company_clusters,
        'cluster_summary': cluster_summary,
        'cohort_data': cohort_df,
        'risk_stats': risk_stats
    }

# Execute the analysis when the script is run
if __name__ == "__main__":
    results = main()
