import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from datetime import datetime, timedelta
import random
from scipy import stats
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings('ignore')

# Set styles
plt.style.use('ggplot')
sns.set_palette("Set2")

def generate_realistic_banking_data(num_companies=100, years=4):
    """
    Generate realistic banking data for the specified number of companies over a period of years.
    Returns a DataFrame with company data at daily level.
    """
    # Create date range for 4 years at daily level
    end_date = datetime(2024, 12, 31)
    start_date = end_date - timedelta(days=365 * years)
    dates = pd.date_range(start=start_date, end=end_date, freq='D')
    
    # Create company IDs
    company_ids = [f'COMP{str(i).zfill(3)}' for i in range(1, num_companies + 1)]
    
    # Initialize empty dataframe
    records = []
    
    # Company profiles - different behaviors
    company_profiles = {
        'stable': 0.4,  # Stable companies
        'growing': 0.2,  # Growing companies
        'declining': 0.15,  # Declining companies
        'seasonal': 0.15,  # Seasonal businesses
        'volatile': 0.1  # Volatile businesses
    }
    
    # Assign profiles to companies
    company_to_profile = {}
    profile_types = list(company_profiles.keys())
    profile_weights = list(company_profiles.values())
    
    for company in company_ids:
        profile = random.choices(profile_types, weights=profile_weights, k=1)[0]
        company_to_profile[company] = profile
    
    # Generate initial values for each company
    company_initial_values = {}
    for company in company_ids:
        profile = company_to_profile[company]
        
        # Determine if company has deposits (80% do)
        has_deposits = random.random() < 0.8
        
        # Initial values based on profile
        if profile == 'stable':
            deposit_base = random.uniform(500000, 5000000) if has_deposits else 0
            loan_limit = random.uniform(200000, 2000000)
            util_pct = random.uniform(0.5, 0.7)  # Initial utilization percentage
        elif profile == 'growing':
            deposit_base = random.uniform(300000, 3000000) if has_deposits else 0
            loan_limit = random.uniform(400000, 4000000)
            util_pct = random.uniform(0.6, 0.8)
        elif profile == 'declining':
            deposit_base = random.uniform(1000000, 10000000) if has_deposits else 0
            loan_limit = random.uniform(800000, 8000000)
            util_pct = random.uniform(0.7, 0.9)
        elif profile == 'seasonal':
            deposit_base = random.uniform(200000, 2000000) if has_deposits else 0
            loan_limit = random.uniform(150000, 1500000)
            util_pct = random.uniform(0.4, 0.6)
        else:  # volatile
            deposit_base = random.uniform(100000, 1000000) if has_deposits else 0
            loan_limit = random.uniform(100000, 1000000)
            util_pct = random.uniform(0.3, 0.9)
        
        # Store initial values
        used_loan = loan_limit * util_pct
        unused_loan = loan_limit - used_loan
        
        company_initial_values[company] = {
            'profile': profile,
            'has_deposits': has_deposits,
            'deposit_balance': deposit_base,
            'used_loan': used_loan,
            'unused_loan': unused_loan,
            'seasonality_factor': random.uniform(0.8, 1.2) if profile == 'seasonal' else 1.0
        }
    
    # Create time series data for each company
    for company in company_ids:
        profile = company_to_profile[company]
        init_values = company_initial_values[company]
        
        # Parameters for this company
        deposit_balance = init_values['deposit_balance']
        used_loan = init_values['used_loan']
        unused_loan = init_values['unused_loan']
        has_deposits = init_values['has_deposits']
        seasonality_factor = init_values['seasonality_factor']
        
        # Risk patterns (randomly assign to some companies)
        is_risky_pattern = random.random() < 0.2  # 20% of companies show risk patterns
        risk_start_date = random.choice(dates[int(len(dates) * 0.7):]) if is_risky_pattern else None
        
        for date in dates:
            # Apply profile-specific trends
            if profile == 'growing':
                growth_factor = 1 + random.uniform(0.0001, 0.0005)  # Small daily growth
                used_loan *= growth_factor
                if has_deposits:
                    deposit_growth = random.uniform(0.0001, 0.0003)
                    deposit_balance *= (1 + deposit_growth)
                
            elif profile == 'declining':
                decline_factor = 1 - random.uniform(0.0001, 0.0003)
                used_loan *= decline_factor
                if has_deposits:
                    deposit_decline = random.uniform(0.0001, 0.0004)
                    deposit_balance *= (1 - deposit_decline)
                
            elif profile == 'seasonal':
                # Seasonal variations (quarterly patterns)
                day_of_year = date.dayofyear
                seasonal_effect = np.sin(day_of_year / 365 * 2 * np.pi) * 0.1 * seasonality_factor
                used_loan *= (1 + seasonal_effect/100)
                if has_deposits:
                    deposit_balance *= (1 + seasonal_effect/150)
                
            elif profile == 'volatile':
                # Random variations
                volatility = random.uniform(-0.01, 0.01)
                used_loan *= (1 + volatility)
                if has_deposits:
                    deposit_volatility = random.uniform(-0.008, 0.008)
                    deposit_balance *= (1 + deposit_volatility)
            
            # Stable profile has minor random fluctuations
            if profile == 'stable':
                used_loan *= (1 + random.uniform(-0.0005, 0.0005))
                if has_deposits:
                    deposit_balance *= (1 + random.uniform(-0.0003, 0.0003))
            
            # Apply risk patterns if applicable
            if is_risky_pattern and date >= risk_start_date:
                # Risk pattern: increasing loan utilization with decreasing deposits
                used_loan *= (1 + random.uniform(0.001, 0.003))
                if has_deposits:
                    deposit_balance *= (1 - random.uniform(0.001, 0.002))
            
            # Recalculate unused loan based on original limit
            total_loan = used_loan + unused_loan
            unused_loan = max(0, total_loan - used_loan)
            
            # Add some randomness to create missing/zero values
            if random.random() < 0.05:  # 5% chance of missing value
                if random.random() < 0.5 and has_deposits:
                    deposit_balance = 0
                else:
                    used_loan = 0
                    unused_loan = 0
            
            # Append record
            records.append({
                'company_id': company,
                'date': date,
                'deposit_balance': max(0, round(deposit_balance, 2)),
                'used_loan': max(0, round(used_loan, 2)),
                'unused_loan': max(0, round(unused_loan, 2)),
                'profile': profile  # Store profile for validation
            })
    
    # Convert to DataFrame
    df = pd.DataFrame(records)
    
    return df

def clean_data(df, min_nonzero_pct=0.8):
    """
    Clean the data by:
    1. Removing companies where less than min_nonzero_pct of their values are non-zero
    2. Removing NaN and infinite values
    """
    print(f"Original data shape: {df.shape}")
    
    # Calculate percentage of non-zero values for each company
    company_stats = {}
    for company in df['company_id'].unique():
        company_data = df[df['company_id'] == company]
        
        deposit_nonzero = (company_data['deposit_balance'] > 0).mean()
        used_loan_nonzero = (company_data['used_loan'] > 0).mean()
        unused_loan_nonzero = (company_data['unused_loan'] > 0).mean()
        
        company_stats[company] = {
            'deposit_nonzero': deposit_nonzero,
            'used_loan_nonzero': used_loan_nonzero,
            'unused_loan_nonzero': unused_loan_nonzero
        }
    
    # Filter companies based on the minimum percentage requirement
    valid_companies = []
    for company, stats in company_stats.items():
        if (stats['deposit_nonzero'] >= min_nonzero_pct or 
            stats['used_loan_nonzero'] >= min_nonzero_pct and 
            stats['unused_loan_nonzero'] >= min_nonzero_pct):
            valid_companies.append(company)
    
    # Filter dataframe
    df_clean = df[df['company_id'].isin(valid_companies)].copy()
    
    # Replace any remaining zeros with NaN for calculations
    # (keeping zeros intact in the original data)
    df_calc = df_clean.copy()
    df_calc.loc[df_calc['deposit_balance'] == 0, 'deposit_balance'] = np.nan
    df_calc.loc[df_calc['used_loan'] == 0, 'used_loan'] = np.nan
    df_calc.loc[df_calc['unused_loan'] == 0, 'unused_loan'] = np.nan
    
    # Remove infinite values
    df_calc.replace([np.inf, -np.inf], np.nan, inplace=True)
    
    print(f"Cleaned data shape: {df_clean.shape}")
    print(f"Removed {len(df['company_id'].unique()) - len(valid_companies)} companies")
    
    return df_clean, df_calc

def add_derived_metrics(df):
    """
    Add derived metrics to the dataframe:
    - loan_utilization: used_loan / (used_loan + unused_loan)
    - total_loan: used_loan + unused_loan
    """
    df['total_loan'] = df['used_loan'] + df['unused_loan']
    df['loan_utilization'] = df['used_loan'] / df['total_loan']
    
    # Handle NaN values for loan_utilization
    df['loan_utilization'].fillna(0, inplace=True)
    df.loc[df['total_loan'] == 0, 'loan_utilization'] = 0
    
    return df

def create_segments(df):
    """
    Create segments/categories for:
    - loan_utilization: high, medium, low
    - used_loan: high, medium, low
    - deposit_balance: high, medium, low
    
    Uses percentiles to create even segments.
    """
    # Create a copy to avoid modifying the original
    df_seg = df.copy()
    
    # Function to categorize values
    def categorize(series, name):
        # Filter out NaNs for quantile calculation
        valid_series = series.dropna()
        
        if len(valid_series) < 3:
            # If not enough data, assign all to 'medium'
            return pd.Series('medium', index=series.index)
        
        lower_third = valid_series.quantile(0.33)
        upper_third = valid_series.quantile(0.67)
        
        # Make sure bin edges are unique
        if lower_third == upper_third:
            # If we have the same value, create arbitrary bins
            lower_third = valid_series.min() + (valid_series.max() - valid_series.min()) / 3
            upper_third = valid_series.min() + 2 * (valid_series.max() - valid_series.min()) / 3
        
        # If still not unique, just divide the range into three equal parts
        if lower_third >= upper_third:
            min_val = valid_series.min()
            max_val = valid_series.max()
            lower_third = min_val + (max_val - min_val) / 3
            upper_third = min_val + 2 * (max_val - min_val) / 3
        
        # Create categories
        try:
            categories = pd.cut(
                series, 
                bins=[float('-inf'), lower_third, upper_third, float('inf')],
                labels=['low', 'medium', 'high']
            )
            return categories
        except ValueError:
            # Fallback in case of error - assign all to 'medium'
            print(f"Warning: Could not create categories for {name}. Assigning all to 'medium'.")
            return pd.Series('medium', index=series.index)
    
    # Create segments
    # Handle loan utilization
    df_seg['loan_util_segment'] = categorize(df_seg['loan_utilization'], 'loan_utilization')
    
    # Initialize the columns before trying to fill them
    df_seg['used_loan_segment'] = 'none'
    df_seg['deposit_segment'] = 'none'
    
    # Handle used loan - only categorize non-zero values
    non_zero_loans = df_seg[df_seg['used_loan'] > 0]
    if not non_zero_loans.empty:
        df_seg.loc[non_zero_loans.index, 'used_loan_segment'] = categorize(
            non_zero_loans['used_loan'], 'used_loan'
        )
    
    # Only create deposit segment for non-zero deposits
    deposit_df = df_seg[df_seg['deposit_balance'] > 0]
    if not deposit_df.empty:
        df_seg.loc[deposit_df.index, 'deposit_segment'] = categorize(
            deposit_df['deposit_balance'], 'deposit_balance'
        )
    
    return df_seg

def analyze_correlations(df):
    """
    Analyze correlations between loan utilization and deposits for each company.
    Returns DataFrame with correlation information.
    """
    correlations = []
    
    for company in df['company_id'].unique():
        company_data = df[df['company_id'] == company]
        
        # Only calculate if company has deposits
        if (company_data['deposit_balance'] > 0).any():
            # Filter out rows where either value is zero
            valid_data = company_data[(company_data['deposit_balance'] > 0) & 
                                      (company_data['loan_utilization'] > 0)]
            
            # Need enough data points
            if len(valid_data) > 5:  # At least 5 data points for meaningful correlation
                # Calculate correlation
                corr = valid_data['loan_utilization'].corr(valid_data['deposit_balance'])
                p_value = 0  # Placeholder for now
                
                if not pd.isna(corr):
                    # Determine correlation type
                    if abs(corr) >= 0.7:
                        if corr > 0:
                            corr_type = 'highly_correlated'
                        else:
                            corr_type = 'highly_anti_correlated'
                    elif abs(corr) >= 0.4:
                        if corr > 0:
                            corr_type = 'moderately_correlated'
                        else:
                            corr_type = 'moderately_anti_correlated'
                    else:
                        corr_type = 'weakly_correlated'
                    
                    correlations.append({
                        'company_id': company,
                        'correlation': corr,
                        'p_value': p_value,
                        'correlation_type': corr_type
                    })
    
    # Convert to DataFrame
    if correlations:
        corr_df = pd.DataFrame(correlations)
        return corr_df
    else:
        # Return empty DataFrame with correct columns
        return pd.DataFrame(columns=['company_id', 'correlation', 'p_value', 'correlation_type'])


def detect_risk_patterns(df, time_windows=[3, 6]):
    """
    Detect risk patterns based on specified rules:
    1. Loan utilization increasing but deposits decreasing over time windows
    2. Loan utilization steady but deposits decreasing
    3. Loan decreasing but deposits decreasing faster
    4. Dramatic increase in loan utilization
    5. Consistent decline in deposits with stable or increasing loans
    6. Sudden drop in deposits with high loan utilization
    7. Increasing volatility in both metrics
    
    Returns DataFrame with risk flags and detailed descriptions.
    """
    risk_records = []
    
    for company in df['company_id'].unique():
        company_data = df[df['company_id'] == company].sort_values('date')
        
        # Skip if not enough data points
        if len(company_data) < max(time_windows) * 30:  # Need enough days for largest window
            continue
        
        # Only analyze if company has deposits
        if not (company_data['deposit_balance'] > 0).any():
            continue
        
        # Process each date to look for risk patterns
        for i in range(max(time_windows) * 30, len(company_data)):
            current_date = company_data.iloc[i]['date']
            
            # Initialize risk flags and descriptions
            risk_flags = []
            risk_descriptions = []
            risk_levels = []
            
            # Analyze for each time window
            for window in time_windows:
                window_days = 30 * window  # Convert months to days approximately
                
                # Get data for the window period
                window_data = company_data.iloc[i-window_days:i+1]
                
                # Calculate trends using linear regression for more accurate detection
                dates_numeric = (window_data['date'] - window_data['date'].min()).dt.days.values
                
                # Prepare data for regression (handle NaN values)
                valid_deposit = ~np.isnan(window_data['deposit_balance'].values)
                valid_util = ~np.isnan(window_data['loan_utilization'].values)
                
                if sum(valid_deposit) > window_days/2 and sum(valid_util) > window_days/2:
                    deposit_slope = 0
                    util_slope = 0
                    loan_slope = 0
                    
                    # Calculate deposit trend
                    if sum(valid_deposit) >= 2:
                        deposit_values = window_data.loc[valid_deposit, 'deposit_balance'].values
                        deposit_dates = dates_numeric[valid_deposit]
                        deposit_slope, _, _, _, _ = stats.linregress(deposit_dates, deposit_values)
                    
                    # Calculate utilization trend
                    if sum(valid_util) >= 2:
                        util_values = window_data.loc[valid_util, 'loan_utilization'].values
                        util_dates = dates_numeric[valid_util]
                        util_slope, _, _, _, _ = stats.linregress(util_dates, util_values)
                    
                    # Calculate loan amount trend
                    valid_loan = ~np.isnan(window_data['used_loan'].values)
                    if sum(valid_loan) >= 2:
                        loan_values = window_data.loc[valid_loan, 'used_loan'].values
                        loan_dates = dates_numeric[valid_loan]
                        loan_slope, _, _, _, _ = stats.linregress(loan_dates, loan_values)
                    
                    # Calculate percentage changes for better interpretability
                    start_deposit = window_data['deposit_balance'].iloc[0]
                    end_deposit = window_data['deposit_balance'].iloc[-1]
                    deposit_pct_change = ((end_deposit - start_deposit) / start_deposit * 100) if start_deposit > 0 else 0
                    
                    start_util = window_data['loan_utilization'].iloc[0]
                    end_util = window_data['loan_utilization'].iloc[-1]
                    util_pct_change = end_util - start_util  # Already a percentage
                    
                    # Current values
                    current_util = company_data.iloc[i]['loan_utilization']
                    
                    # Volatility measures
                    deposit_volatility = window_data['deposit_balance'].pct_change().std()
                    util_volatility = window_data['loan_utilization'].diff().std()
                    
                    # Check for specific risk patterns
                    
                    # Pattern 1: Increasing loan utilization with decreasing deposits
                    if util_slope > 0.0001 and deposit_slope < -0.0001:
                        risk_level = 'high' if abs(util_pct_change) > 0.1 and abs(deposit_pct_change) > 5 else 'medium'
                        risk_flags.append(f'pattern1_{window}m')
                        risk_descriptions.append(f"[{risk_level.upper()}] {window}m: Loan utilization increasing ({util_pct_change:.1f}%) while deposits decreasing ({deposit_pct_change:.1f}%)")
                        risk_levels.append(risk_level)
                    
                    # Pattern 2: Stable loan utilization with decreasing deposits
                    elif abs(util_slope) < 0.0001 and deposit_slope < -0.0001:
                        risk_level = 'medium' if abs(deposit_pct_change) > 10 else 'low'
                        risk_flags.append(f'pattern2_{window}m')
                        risk_descriptions.append(f"[{risk_level.upper()}] {window}m: Stable loan utilization with declining deposits ({deposit_pct_change:.1f}%)")
                        risk_levels.append(risk_level)
                    
                    # Pattern 3: Decreasing loan but deposits decreasing faster
                    elif loan_slope < 0 and deposit_slope < loan_slope * 1.5:
                        risk_level = 'medium'
                        risk_flags.append(f'pattern3_{window}m')
                        risk_descriptions.append(f"[{risk_level.upper()}] {window}m: Deposits declining faster ({deposit_pct_change:.1f}%) than loan amount")
                        risk_levels.append(risk_level)
                    
                    # Pattern 4: Dramatic increase in loan utilization
                    elif util_pct_change > 0.15:  # 15 percentage points increase
                        risk_level = 'high' if current_util > 0.8 else 'medium'
                        risk_flags.append(f'pattern4_{window}m')
                        risk_descriptions.append(f"[{risk_level.upper()}] {window}m: Sharp increase in loan utilization ({util_pct_change:.1f}%)")
                        risk_levels.append(risk_level)
                    
                    # Pattern 5: Consistent decline in deposits with stable/increasing loans
                    elif deposit_pct_change < -15 and loan_slope >= 0:
                        risk_level = 'high' if deposit_pct_change < -25 else 'medium'
                        risk_flags.append(f'pattern5_{window}m')
                        risk_descriptions.append(f"[{risk_level.upper()}] {window}m: Significant deposit decline ({deposit_pct_change:.1f}%) with stable/increasing loans")
                        risk_levels.append(risk_level)
                    
                    # Pattern 6: Sudden drop in deposits with high loan utilization
                    recent_deposit_change = window_data['deposit_balance'].pct_change(periods=10).iloc[-1] * 100
                    if recent_deposit_change < -10 and current_util > 0.75:
                        risk_level = 'high'
                        risk_flags.append(f'pattern6_{window}m')
                        risk_descriptions.append(f"[{risk_level.upper()}] {window}m: Recent sharp drop in deposits ({recent_deposit_change:.1f}%) with high loan utilization ({current_util:.1%})")
                        risk_levels.append(risk_level)
                    
                    # Pattern 7: Increasing volatility in both metrics
                    prev_window_data = company_data.iloc[i-2*window_days:i-window_days+1]
                    if len(prev_window_data) > 0:
                        prev_deposit_volatility = prev_window_data['deposit_balance'].pct_change().std()
                        prev_util_volatility = prev_window_data['loan_utilization'].diff().std()
                        
                        if (deposit_volatility > prev_deposit_volatility * 1.5 and 
                            util_volatility > prev_util_volatility * 1.5):
                            risk_level = 'medium'
                            risk_flags.append(f'pattern7_{window}m')
                            risk_descriptions.append(f"[{risk_level.upper()}] {window}m: Increasing volatility in both deposits and loan utilization")
                            risk_levels.append(risk_level)
            
            # If risks are detected for current date, add to records
            if risk_flags:
                # Get highest risk level
                highest_risk = 'low'
                if 'high' in risk_levels:
                    highest_risk = 'high'
                elif 'medium' in risk_levels:
                    highest_risk = 'medium'
                
                risk_records.append({
                    'company_id': company,
                    'date': current_date,
                    'risk_flags': ', '.join(risk_flags),
                    'risk_description': ' | '.join(risk_descriptions),
                    'risk_level': highest_risk
                })
    
    # Convert to DataFrame if we have records
    if risk_records:
        risk_df = pd.DataFrame(risk_records)
        return risk_df
    else:
        return pd.DataFrame(columns=['company_id', 'date', 'risk_flags', 'risk_description', 'risk_level'])

def cluster_companies(df, risk_df, n_clusters=4):
    """
    Cluster companies based on their behavior patterns and risk profiles.
    Returns DataFrame with cluster assignments.
    """
    # Prepare features for clustering
    company_features = []
    
    for company in df['company_id'].unique():
        company_data = df[df['company_id'] == company]
        
        # Calculate features
        avg_loan_util = company_data['loan_utilization'].mean()
        util_volatility = company_data['loan_utilization'].std()
        
        # Deposit features (if applicable)
        has_deposits = (company_data['deposit_balance'] > 0).any()
        avg_deposit = company_data['deposit_balance'].mean() if has_deposits else 0
        deposit_volatility = company_data['deposit_balance'].pct_change().std() if has_deposits else 0
        
        # Risk features
        company_risks = risk_df[risk_df['company_id'] == company]
        risk_count = len(company_risks)
        has_high_risk = (company_risks['risk_level'] == 'high').any() if not company_risks.empty else False
        has_medium_risk = (company_risks['risk_level'] == 'medium').any() if not company_risks.empty else False
        
        # Store features
        company_features.append({
            'company_id': company,
            'avg_loan_util': avg_loan_util,
            'util_volatility': util_volatility,
            'has_deposits': has_deposits,
            'avg_deposit': avg_deposit,
            'deposit_volatility': deposit_volatility,
            'risk_count': risk_count,
            'has_high_risk': has_high_risk,
            'has_medium_risk': has_medium_risk
        })
    
    # Convert to DataFrame
    features_df = pd.DataFrame(company_features)
    
    # Prepare numerical features for clustering
    num_features = ['avg_loan_util', 'util_volatility', 'avg_deposit', 
                   'deposit_volatility', 'risk_count']
    X = features_df[num_features].fillna(0)
    
    # Normalize features
    X_norm = (X - X.mean()) / X.std()
    
    # Perform KMeans clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X_norm)
    
    # Add cluster assignments to features DataFrame
    features_df['cluster'] = clusters
    
    # Analyze clusters
    cluster_profiles = []
    for cluster in range(n_clusters):
        cluster_data = features_df[features_df['cluster'] == cluster]
        
        profile = {
            'cluster': cluster,
            'size': len(cluster_data),
            'avg_loan_util': cluster_data['avg_loan_util'].mean(),
            'util_volatility': cluster_data['util_volatility'].mean(),
            'pct_with_deposits': cluster_data['has_deposits'].mean() * 100,
            'avg_deposit': cluster_data['avg_deposit'].mean(),
            'deposit_volatility': cluster_data['deposit_volatility'].mean(),
            'avg_risk_count': cluster_data['risk_count'].mean(),
            'pct_high_risk': cluster_data['has_high_risk'].mean() * 100,
            'pct_medium_risk': cluster_data['has_medium_risk'].mean() * 100
        }
        
        cluster_profiles.append(profile)
    
    # Create cluster profile DataFrame
    profile_df = pd.DataFrame(cluster_profiles)
    
    # Assign descriptive names to clusters
    cluster_names = []
    for _, profile in profile_df.iterrows():
        cluster = profile['cluster']
        
        if profile['pct_high_risk'] > 30:
            name = "High Risk Clients"
        elif profile['avg_loan_util'] > 0.7 and profile['pct_with_deposits'] < 50:
            name = "High Utilization, Low Deposit"
        elif profile['avg_loan_util'] < 0.4 and profile['pct_with_deposits'] > 70:
            name = "Low Utilization, High Deposit"
        elif profile['util_volatility'] > profile_df['util_volatility'].median() * 1.5:
            name = "Volatile Utilization"
        elif profile['deposit_volatility'] > profile_df['deposit_volatility'].median() * 1.5:
            name = "Volatile Deposits"
        elif profile['avg_risk_count'] > profile_df['avg_risk_count'].median() * 1.5:
            name = "Moderate Risk Clients"
        else:
            name = "Stable Clients"
        
        cluster_names.append({'cluster': cluster, 'cluster_name': name})
    
    # Create cluster names DataFrame and merge with clusters
    names_df = pd.DataFrame(cluster_names)
    profile_df = profile_df.merge(names_df, on='cluster')
    
    # Add cluster names to features DataFrame
    features_df = features_df.merge(names_df, on='cluster')
    
    return features_df, profile_df

def perform_cohort_analysis(df, cluster_df, time_periods=['quarter']):
    """
    Perform cohort analysis for different clusters over time.
    Analyzes changes in behavior across time periods (quarterly, monthly).
    """
    cohort_results = []
    
    # Create time period columns
    df['year'] = df['date'].dt.year
    df['quarter'] = df['date'].dt.quarter
    df['month'] = df['date'].dt.month
    
    # Merge cluster information
    df_with_clusters = df.merge(
        cluster_df[['company_id', 'cluster', 'cluster_name']], 
        on='company_id'
    )
    
    # Analyze for each time period
    for period in time_periods:
        if period == 'quarter':
            df_with_clusters['time_period'] = df_with_clusters['year'].astype(str) + "-Q" + df_with_clusters['quarter'].astype(str)
            group_by = ['cluster', 'cluster_name', 'time_period']
        elif period == 'month':
            df_with_clusters['time_period'] = df_with_clusters['year'].astype(str) + "-" + df_with_clusters['month'].astype(str).str.zfill(2)
            group_by = ['cluster', 'cluster_name', 'time_period']
        else:
            df_with_clusters['time_period'] = df_with_clusters['year'].astype(str)
            group_by = ['cluster', 'cluster_name', 'time_period']
        
        # Group by cluster and time period
        cohort_stats = df_with_clusters.groupby(group_by).agg({
            'loan_utilization': ['mean', 'std'],
            'deposit_balance': ['mean', 'std'],
            'used_loan': ['mean', 'sum'],
            'company_id': 'nunique'
        }).reset_index()
        
        # Rename columns
        cohort_stats.columns = ['_'.join(col).strip('_') for col in cohort_stats.columns.values]
        
        # Add to results
        cohort_results.append(cohort_stats)
    
    return cohort_results

def plot_risk_company(company_id, df, risk_df):
    """
    Create a detailed plot of a company's risk patterns.
    Shows loan utilization and deposit balance on two axes with risk reasons annotated.
    """
    # Filter data for company
    company_data = df[df['company_id'] == company_id].sort_values('date')
    company_risks = risk_df[risk_df['company_id'] == company_id].sort_values('date')
    
    if company_data.empty:
        print(f"No data found for company {company_id}")
        return
    
    # Create figure with two y-axes
    fig, ax1 = plt.subplots(figsize=(15, 8))
    
    # First axis - Loan Utilization
    color = 'tab:blue'
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Loan Utilization (%)', color=color)
    ax1.plot(company_data['date'], company_data['loan_utilization'] * 100, color=color, linewidth=2)
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.set_ylim(0, 100)
    
    # Second axis - Deposits
    color = 'tab:red'
    ax2 = ax1.twinx()
    ax2.set_ylabel('Deposit Balance', color=color)
    ax2.plot(company_data['date'], company_data['deposit_balance'], color=color, linewidth=2)
    ax2.tick_params(axis='y', labelcolor=color)
    
    # Format dates on x-axis
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.xticks(rotation=45)
    
    # Add risk markers
    if not company_risks.empty:
        risk_dates = company_risks['date'].tolist()
        risk_descriptions = company_risks['risk_description'].tolist()
        risk_levels = company_risks['risk_level'].tolist()
        
        # Create color map for risk levels
        risk_colors = {'high': 'red', 'medium': 'orange', 'low': 'yellow'}
        
        # Add vertical lines for each risk event
        for i, (date, desc, level) in enumerate(zip(risk_dates, risk_descriptions, risk_levels)):
            # Limit to plotting only a reasonable number of risk markers
            if i < 10:  # Only show first 10 risk events to avoid cluttering
                ax1.axvline(x=date, color=risk_colors[level], linestyle='--', alpha=0.7)
                
                # Add descriptions (shortened if too long)
                short_desc = desc.split('|')[0] if len(desc) > 50 else desc
                y_pos = 90 - (i % 5) * 15  # Stagger text vertically
                
                ax1.annotate(short_desc, xy=(date, y_pos), xytext=(10, 0), 
                            textcoords="offset points", color=risk_colors[level],
                            rotation=0, ha='left', va='center', fontsize=8,
                            bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8))
    
    # Add title and company details
    plt.title(f"Risk Analysis for {company_id}", fontsize=16)
    
    # Add key metrics in text box
    avg_util = company_data['loan_utilization'].mean() * 100
    avg_deposit = company_data['deposit_balance'].mean()
    risk_count = len(company_risks)
    
    metrics_text = (
        f"Average Utilization: {avg_util:.1f}%\n"
        f"Average Deposit: ${avg_deposit:,.2f}\n"
        f"Risk Events: {risk_count}"
    )
    
    # Add text box with metrics
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax1.text(0.02, 0.05, metrics_text, transform=ax1.transAxes, fontsize=10,
            verticalalignment='bottom', bbox=props)
    
    plt.tight_layout()
    return fig

def plot_cluster_trends(cohort_results, cluster_name=None):
    """
    Plot trends for clusters over time.
    If cluster_name is provided, only that cluster is plotted.
    """
    # Use quarterly cohort data for trend analysis
    quarterly_data = [df for df in cohort_results if 'Q' in df['time_period'].iloc[0]]
    
    if not quarterly_data:
        print("No quarterly data available for trend analysis")
        return
    
    df = quarterly_data[0]
    
    # Filter for specific cluster if provided
    if cluster_name:
        df = df[df['cluster_name'] == cluster_name]
        if df.empty:
            print(f"No data found for cluster '{cluster_name}'")
            return
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
    
    # Get unique clusters for plotting
    clusters = df['cluster_name'].unique()
    
    # Plot loan utilization trends
    for cluster in clusters:
        cluster_data = df[df['cluster_name'] == cluster]
        ax1.plot(cluster_data['time_period'], cluster_data['loan_utilization_mean'] * 100,
                marker='o', linewidth=2, label=cluster)
    
    ax1.set_ylabel('Average Loan Utilization (%)')
    ax1.set_title('Loan Utilization Trends by Cluster Over Time')
    ax1.grid(True, linestyle='--', alpha=0.7)
    ax1.legend()
    
    # Plot deposit balance trends
    for cluster in clusters:
        cluster_data = df[df['cluster_name'] == cluster]
        ax2.plot(cluster_data['time_period'], cluster_data['deposit_balance_mean'],
                marker='o', linewidth=2, label=cluster)
    
    ax2.set_xlabel('Time Period')
    ax2.set_ylabel('Average Deposit Balance')
    ax2.set_title('Deposit Balance Trends by Cluster Over Time')
    ax2.grid(True, linestyle='--', alpha=0.7)
    
    # Format x-axis labels
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    return fig

def plot_waterfall_chart(df):
    """
    Create a waterfall chart showing:
    1. All companies with loans
    2. Companies with both loans and deposits
    3. Breakdown by loan utilization segments
    """
    # Get total number of companies with loans
    total_companies = df['company_id'].nunique()
    
    # Get companies with both loans and deposits
    companies_with_deposits = df[df['deposit_balance'] > 0]['company_id'].nunique()
    
    # Get breakdown by loan utilization segment
    segment_counts = df.groupby('company_id')['loan_util_segment'].first().value_counts()
    
    # Prepare data for waterfall chart
    stages = ['All Companies', 'With Deposits', 'High Util', 'Medium Util', 'Low Util']
    values = [
        total_companies,
        companies_with_deposits - total_companies,  # Change from all to with deposits
        segment_counts.get('high', 0),
        segment_counts.get('medium', 0),
        segment_counts.get('low', 0)
    ]
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Plot waterfall
    cumulative = 0
    for i, (stage, value) in enumerate(zip(stages, values)):
        if i == 0:
            # First bar (total)
            ax.bar(i, value, bottom=0, color='tab:blue', label='Total')
            cumulative = value
        else:
            # Other bars
            ax.bar(i, value, bottom=cumulative, color='tab:green' if value >= 0 else 'tab:red',
                  label='Increase' if i == 1 else 'Segment')
            cumulative += value
        
        # Add value labels
        if i == 0:
            ax.text(i, value/2, f"{value}", ha='center', va='center', color='white', fontweight='bold')
        else:
            if value >= 0:
                ax.text(i, cumulative - value/2, f"+{value}", ha='center', va='center', color='white', fontweight='bold')
            else:
                ax.text(i, cumulative - value/2, f"{value}", ha='center', va='center', color='white', fontweight='bold')
    
    # Add connecting lines
    for i in range(1, len(stages)):
        ax.plot([i-1, i], [cumulative - values[i], cumulative - values[i]], 'k--', alpha=0.3)
    
    # Set labels and title
    ax.set_ylabel('Number of Companies')
    ax.set_title('Waterfall Chart: Company Breakdown by Deposits and Utilization', fontsize=14)
    ax.set_xticks(range(len(stages)))
    ax.set_xticklabels(stages, rotation=45)
    
    plt.tight_layout()
    return fig

def pivot_analysis(df):
    """
    Create pivot tables to analyze patterns across segments.
    """
    # Create a pivot for loan utilization segment vs deposit segment
    util_deposit_pivot = pd.pivot_table(
        df,
        values='company_id',
        index='loan_util_segment',
        columns='deposit_segment',
        aggfunc='nunique',
        fill_value=0
    )
    
    # Create a pivot for loan utilization vs used loan amount
    util_amount_pivot = pd.pivot_table(
        df,
        values='company_id',
        index='loan_util_segment',
        columns='used_loan_segment',
        aggfunc='nunique',
        fill_value=0
    )
    
    return util_deposit_pivot, util_amount_pivot

def visualize_pivot_tables(util_deposit_pivot, util_amount_pivot):
    """
    Create heatmap visualizations for pivot tables.
    """
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))
    
    # Plot first heatmap - Loan Utilization vs Deposit Segments
    sns.heatmap(util_deposit_pivot, annot=True, cmap='Blues', fmt='d', ax=ax1)
    ax1.set_title('Company Count: Loan Utilization vs Deposit Segments', fontsize=14)
    
    # Plot second heatmap - Loan Utilization vs Used Loan Amount
    sns.heatmap(util_amount_pivot, annot=True, cmap='Greens', fmt='d', ax=ax2)
    ax2.set_title('Company Count: Loan Utilization vs Used Loan Amount', fontsize=14)
    
    plt.tight_layout()
    return fig

def main():
    """
    Main function to execute the entire analysis workflow.
    """
    print("Starting bank client risk analysis...")
    
    # 1. Generate realistic data
    print("\nGenerating data...")
    df = generate_realistic_banking_data(num_companies=100, years=4)
    
    # 2. Clean data
    print("\nCleaning data...")
    df_clean, df_calc = clean_data(df, min_nonzero_pct=0.8)
    
    # 3. Add derived metrics
    print("\nAdding derived metrics...")
    df_with_metrics = add_derived_metrics(df_clean)
    
    # 4. Create segments
    print("\nCreating segments...")
    df_segmented = create_segments(df_with_metrics)
    
    # 5. Analyze correlations
    print("\nAnalyzing correlations...")
    corr_df = analyze_correlations(df_segmented)
    print(f"Found {len(corr_df)} companies with correlation data")
    
    # Handle empty correlation dataframe
    if 'correlation_type' in corr_df.columns:
        print(f"Highly correlated: {len(corr_df[corr_df['correlation_type'] == 'highly_correlated'])}")
        print(f"Highly anti-correlated: {len(corr_df[corr_df['correlation_type'] == 'highly_anti_correlated'])}")
    else:
        print("No significant correlations found")
    
    # 6. Detect risk patterns
    print("\nDetecting risk patterns...")
    risk_df = detect_risk_patterns(df_segmented, time_windows=[3, 6])
    
    # Check if risk dataframe is empty
    if risk_df.empty:
        print("No risk events detected")
        # Create an empty risk dataframe with necessary columns for downstream processing
        risk_df = pd.DataFrame(columns=['company_id', 'date', 'risk_flags', 'risk_description', 'risk_level'])
    else:
        print(f"Found {len(risk_df)} risk events across {risk_df['company_id'].nunique()} companies")
    
    # 7. Cluster companies
    print("\nClustering companies...")
    cluster_df, cluster_profiles = cluster_companies(df_segmented, risk_df, n_clusters=4)
    print("\nCluster profiles:")
    print(cluster_profiles[['cluster', 'cluster_name', 'size', 'avg_loan_util', 'pct_with_deposits', 'avg_risk_count']])
    
    # 8. Perform cohort analysis
    print("\nPerforming cohort analysis...")
    cohort_results = perform_cohort_analysis(df_segmented, cluster_df, time_periods=['quarter'])
    
    # 9. Create pivot tables
    print("\nCreating pivot tables...")
    util_deposit_pivot, util_amount_pivot = pivot_analysis(df_segmented)
    
    # 10. Visualize waterfall chart
    print("\nCreating waterfall chart...")
    waterfall_fig = plot_waterfall_chart(df_segmented)
    waterfall_fig.savefig('waterfall_chart.png')
    
    # 11. Visualize pivot tables
    print("\nVisualizing pivot tables...")
    pivot_fig = visualize_pivot_tables(util_deposit_pivot, util_amount_pivot)
    pivot_fig.savefig('pivot_analysis.png')
    
    # 12. Visualize cluster trends
    print("\nVisualizing cluster trends...")
    trends_fig = plot_cluster_trends(cohort_results)
    trends_fig.savefig('cluster_trends.png')
    
    # 13. Plot sample risky companies
    print("\nPlotting sample risky companies...")
    # Get top 3 companies with most risk events (or all if fewer than 3)
    if not risk_df.empty:
        top_risky_companies = risk_df['company_id'].value_counts().head(3).index.tolist()
        
        for company_id in top_risky_companies:
            print(f"Plotting risk analysis for {company_id}...")
            company_fig = plot_risk_company(company_id, df_segmented, risk_df)
            company_fig.savefig(f'risk_analysis_{company_id}.png')
    else:
        print("No risky companies to plot")
    
    print("\nAnalysis complete! Visualization files saved.")
    
    return {
        'data': df_segmented,
        'risk_df': risk_df,
        'cluster_df': cluster_df,
        'cluster_profiles': cluster_profiles,
        'cohort_results': cohort_results
    }

# If script is run directly
if __name__ == "__main__":
    results = main()
