import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from datetime import datetime, timedelta
from scipy import stats, signal
import statsmodels.api as sm
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.decomposition import PCA
from sklearn.impute import KNNImputer
from tqdm import tqdm
import matplotlib.ticker as mtick
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import warnings
from statsmodels.tsa.seasonal import seasonal_decompose
from scipy.signal import find_peaks
warnings.filterwarnings('ignore')

# Set styles for better visualizations
plt.style.use('seaborn-v0_8-whitegrid')
custom_params = {"axes.spines.right": False, "axes.spines.top": False}
sns.set_theme(style="ticks", rc=custom_params)
colors = sns.color_palette("viridis", 10)

#######################################################
# PART 1: BANK CLIENT RISK ANALYSIS FRAMEWORK
#######################################################

# Configuration dictionary for easy parameter tuning
CONFIG = {
    'data': {
        'min_nonzero_pct': 0.8,
        'min_continuous_days': 365,  # Minimum continuous days of data required
        'recent_window': 30  # Recent risk window in days
    },
    'risk': {
        'trend_windows': [30, 90, 180],  # days
        'change_thresholds': {
            'sharp': 0.2,    # 20% change
            'moderate': 0.1, # 10% change
            'gradual': 0.05  # 5% change
        },
        'persona_patterns': {
            # Existing personas
            'cautious_borrower': 'Low utilization (<40%), stable deposits',
            'aggressive_expansion': 'Rising utilization (>10% increase), volatile deposits',
            'distressed_client': 'High utilization (>80%), declining deposits (>5% decrease)',
            'seasonal_loan_user': 'Cyclical utilization with >15% amplitude',
            'seasonal_deposit_pattern': 'Cyclical deposits with >20% amplitude',
            'deteriorating_health': 'Rising utilization (>15% increase), declining deposits (>10% decrease)',
            'cash_constrained': 'Stable utilization, rapidly declining deposits (>15% decrease)',
            'credit_dependent': 'High utilization (>75%), low deposit ratio (<0.8)',
            
            # New personas
            'stagnant_growth': 'Loan utilization increasing (>8%) with flat deposits (<2% change)',
            'utilization_spikes': 'Showing sudden large increases (>15%) in loan utilization within short periods',
            'seasonal_pattern_breaking': 'Historical seasonality exists but recent data shows deviation from expected patterns',
            'approaching_limit': 'Utilization nearing credit limit (>90%) with increased usage velocity',
            'withdrawal_intensive': 'Unusual increase in deposit withdrawal frequency or size',
            'deposit_concentration': 'Deposits heavily concentrated in timing, showing potential liquidity planning issues',
            'historical_low_deposits': 'Deposit balance below historical low point while maintaining high utilization'
        },
        'risk_levels': {
            'high': 3,
            'medium': 2,
            'low': 1,
            'none': 0
        }
    },
    'clustering': {
        'n_clusters': 5,
        'random_state': 42
    }
}

def clean_data(df, min_nonzero_pct=0.8):
    """
    Clean the data by:
    1. Removing companies where less than min_nonzero_pct of their values are non-zero
    2. Removing companies without the minimum continuous days of data
    3. Applying sophisticated imputation for missing values
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
    
    # Check for continuous data requirement
    continuous_companies = []
    for company in valid_companies:
        company_data = df_clean[df_clean['company_id'] == company].sort_values('date')
        
        # Check if company has data for the most recent period
        max_date = df_clean['date'].max()
        min_required_date = max_date - pd.Timedelta(days=CONFIG['data']['min_continuous_days'])
        
        # Get recent data
        recent_data = company_data[company_data['date'] >= min_required_date]
        
        # Ensure both deposit and loan data are available
        if (recent_data['deposit_balance'] > 0).sum() >= CONFIG['data']['min_continuous_days'] * 0.8 and \
           (recent_data['used_loan'] > 0).sum() >= CONFIG['data']['min_continuous_days'] * 0.8:
            continuous_companies.append(company)
    
    # Filter for companies with continuous data
    df_clean = df_clean[df_clean['company_id'].isin(continuous_companies)].copy()
    print(f"After continuous data filter: {df_clean.shape}")
    
    # Apply advanced imputation using KNN imputer for each company
    df_imputed = df_clean.copy()
    
    for company in tqdm(continuous_companies, desc="Applying KNN imputation"):
        company_data = df_clean[df_clean['company_id'] == company].sort_values('date')
        
        # Only impute if there are missing values
        if company_data[['deposit_balance', 'used_loan', 'unused_loan']].isna().any().any() or \
           (company_data['deposit_balance'] == 0).any() or \
           (company_data['used_loan'] == 0).any() or \
           (company_data['unused_loan'] == 0).any():
            
            # Prepare data for imputation
            impute_data = company_data[['deposit_balance', 'used_loan', 'unused_loan']].copy()
            
            # Convert zeros to NaN for imputation
            impute_data.loc[impute_data['deposit_balance'] == 0, 'deposit_balance'] = np.nan
            impute_data.loc[impute_data['used_loan'] == 0, 'used_loan'] = np.nan
            impute_data.loc[impute_data['unused_loan'] == 0, 'unused_loan'] = np.nan
            
            # Apply KNN imputation
            imputer = KNNImputer(n_neighbors=5, weights='distance')
            imputed_values = imputer.fit_transform(impute_data)
            
            # Update dataframe with imputed values
            df_imputed.loc[company_data.index, 'deposit_balance'] = imputed_values[:, 0]
            df_imputed.loc[company_data.index, 'used_loan'] = imputed_values[:, 1]
            df_imputed.loc[company_data.index, 'unused_loan'] = imputed_values[:, 2]
    
    # Create a copy for calculations that preserves NaN values
    df_calc = df_imputed.copy()
    
    print(f"Cleaned and imputed data shape: {df_imputed.shape}")
    print(f"Retained {len(continuous_companies)}/{len(df['company_id'].unique())} companies")
    
    return df_imputed, df_calc

def add_derived_metrics(df):
    """
    Add derived metrics to the dataframe including loan utilization, deposit to loan ratio,
    and rolling metrics for trend analysis.
    """
    df = df.copy()
    
    # Basic metrics
    df['total_loan'] = df['used_loan'] + df['unused_loan']
    df['loan_utilization'] = df['used_loan'] / df['total_loan']
    
    # Handle NaN values for loan_utilization
    df.loc[df['total_loan'] == 0, 'loan_utilization'] = np.nan
    
    # Calculate deposit to loan ratio
    df['deposit_loan_ratio'] = df['deposit_balance'] / df['used_loan']
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    
    # Calculate rolling metrics for each company
    for company in tqdm(df['company_id'].unique(), desc="Calculating rolling metrics"):
        company_data = df[df['company_id'] == company].sort_values('date')
        
        # Skip if too few data points
        if len(company_data) < 30:
            continue
            
        # Calculate rolling averages for different windows
        for window in [7, 30, 90]:
            df.loc[company_data.index, f'util_ma_{window}d'] = company_data['loan_utilization'].rolling(
                window, min_periods=max(3, window//4)).mean()
            df.loc[company_data.index, f'deposit_ma_{window}d'] = company_data['deposit_balance'].rolling(
                window, min_periods=max(3, window//4)).mean()
            
        # Calculate rates of change
        for window in [30, 90]:
            df.loc[company_data.index, f'util_change_{window}d'] = (company_data['loan_utilization'].pct_change(periods=window)
                                                                  .rolling(window=7, min_periods=3).mean())  # Smooth changes
            df.loc[company_data.index, f'deposit_change_{window}d'] = (company_data['deposit_balance'].pct_change(periods=window)
                                                                     .rolling(window=7, min_periods=3).mean())
            
        # Calculate volatility measures
        df.loc[company_data.index, 'util_volatility_30d'] = company_data['loan_utilization'].rolling(
            30, min_periods=10).std()
        df.loc[company_data.index, 'deposit_volatility_30d'] = company_data['deposit_balance'].pct_change().rolling(
            30, min_periods=10).std()
        
        # Detect seasonality
        detect_seasonality(df, company_data)
        
        # Add new metrics for enhanced risk detection
        
        # 1. Calculate utilization acceleration (2nd derivative)
        df.loc[company_data.index, 'util_acceleration_30d'] = df.loc[company_data.index, 'util_change_30d'].pct_change(periods=30)
        
        # 2. Calculate deposit concentration metrics (for detecting lumpy deposits)
        if len(company_data) >= 90:
            # Calculate Gini coefficient for deposits over rolling 90-day windows
            for i in range(90, len(company_data)):
                window_data = company_data.iloc[i-90:i]
                deposit_changes = window_data['deposit_balance'].diff().fillna(0)
                positive_changes = deposit_changes[deposit_changes > 0].values
                
                if len(positive_changes) > 5:  # Need sufficient positive changes
                    # Calculate Gini coefficient
                    sorted_changes = np.sort(positive_changes)
                    n = len(sorted_changes)
                    index = np.arange(1, n+1)
                    gini = (np.sum((2 * index - n - 1) * sorted_changes)) / (n * np.sum(sorted_changes))
                    
                    df.loc[company_data.index[i], 'deposit_concentration_gini'] = gini
        
        # 3. Calculate withdrawal frequency and size metrics
        if len(company_data) >= 60:
            for i in range(60, len(company_data)):
                # For the current 30-day window
                current_window = company_data.iloc[i-30:i]
                deposit_changes = current_window['deposit_balance'].diff().fillna(0)
                withdrawals = deposit_changes[deposit_changes < 0].abs()
                
                # For the previous 30-day window
                prev_window = company_data.iloc[i-60:i-30]
                prev_deposit_changes = prev_window['deposit_balance'].diff().fillna(0)
                prev_withdrawals = prev_deposit_changes[prev_deposit_changes < 0].abs()
                
                # Calculate metrics
                withdrawal_count = len(withdrawals)
                withdrawal_avg = withdrawals.mean() if len(withdrawals) > 0 else 0
                
                prev_withdrawal_count = len(prev_withdrawals)
                prev_withdrawal_avg = prev_withdrawals.mean() if len(prev_withdrawals) > 0 else 0
                
                # Store metrics
                df.loc[company_data.index[i], 'withdrawal_count_30d'] = withdrawal_count
                df.loc[company_data.index[i], 'withdrawal_avg_30d'] = withdrawal_avg
                df.loc[company_data.index[i], 'withdrawal_count_change'] = (
                    (withdrawal_count - prev_withdrawal_count) / max(1, prev_withdrawal_count)
                )
                df.loc[company_data.index[i], 'withdrawal_avg_change'] = (
                    (withdrawal_avg - prev_withdrawal_avg) / max(1, prev_withdrawal_avg)
                )
        
        # 4. Calculate historical minimums and current proximity
        if len(company_data) >= 180:
            # Calculate rolling 180-day minimum
            min_deposits = company_data['deposit_balance'].rolling(180, min_periods=90).min()
            df.loc[company_data.index, 'deposit_min_180d'] = min_deposits
            
            # Calculate ratio only where we have valid minimums
            valid_indices = min_deposits.notna()
            if valid_indices.any():
                df.loc[company_data.index[valid_indices], 'deposit_to_min_ratio'] = (
                    company_data.loc[valid_indices, 'deposit_balance'] / min_deposits[valid_indices]
                )
    
    return df

def detect_seasonality(df, company_data, min_periods=365):
    """
    Detect seasonality in both loan utilization and deposits.
    Adds seasonal metrics to the dataframe.
    """
    if len(company_data) < min_periods:
        return
    
    try:
        # For loan utilization
        if company_data['loan_utilization'].notna().sum() >= min_periods * 0.8:
            util_series = company_data['loan_utilization'].interpolate(method='linear')
            
            # Use FFT to detect seasonality
            fft_values = np.fft.fft(util_series.values - util_series.mean())
            fft_freq = np.fft.fftfreq(len(util_series))
            
            # Find peaks excluding the DC component
            peaks = find_peaks(np.abs(fft_values[1:len(fft_values)//2]), height=np.std(np.abs(fft_values))*2)[0] + 1
            
            if len(peaks) > 0:
                # Calculate period of strongest seasonal component
                strongest_idx = peaks[np.argmax(np.abs(fft_values)[peaks])]
                period = int(1 / np.abs(fft_freq[strongest_idx]))
                
                # Calculate seasonal amplitude
                seasonal_amplitude = np.abs(fft_values[strongest_idx]) * 2 / len(util_series)
                normalized_amplitude = seasonal_amplitude / util_series.mean() if util_series.mean() != 0 else 0
                
                # Record seasonality metrics
                if 350 <= period <= 380 or 170 <= period <= 190 or 80 <= period <= 100:  # Annual, semi-annual, quarterly
                    df.loc[company_data.index, 'util_seasonal_period'] = period
                    df.loc[company_data.index, 'util_seasonal_amplitude'] = normalized_amplitude
                    
                    # Flag as seasonal if amplitude is significant
                    df.loc[company_data.index, 'util_is_seasonal'] = normalized_amplitude > 0.15
        
        # For deposits
        if company_data['deposit_balance'].notna().sum() >= min_periods * 0.8:
            deposit_series = company_data['deposit_balance'].interpolate(method='linear')
            
            # Use FFT to detect seasonality
            fft_values = np.fft.fft(deposit_series.values - deposit_series.mean())
            fft_freq = np.fft.fftfreq(len(deposit_series))
            
            # Find peaks excluding the DC component
            peaks = find_peaks(np.abs(fft_values[1:len(fft_values)//2]), height=np.std(np.abs(fft_values))*2)[0] + 1
            
            if len(peaks) > 0:
                # Calculate period of strongest seasonal component
                strongest_idx = peaks[np.argmax(np.abs(fft_values)[peaks])]
                period = int(1 / np.abs(fft_freq[strongest_idx]))
                
                # Calculate seasonal amplitude
                seasonal_amplitude = np.abs(fft_values[strongest_idx]) * 2 / len(deposit_series)
                normalized_amplitude = seasonal_amplitude / deposit_series.mean() if deposit_series.mean() != 0 else 0
                
                # Record seasonality metrics
                if 350 <= period <= 380 or 170 <= period <= 190 or 80 <= period <= 100:  # Annual, semi-annual, quarterly
                    df.loc[company_data.index, 'deposit_seasonal_period'] = period
                    df.loc[company_data.index, 'deposit_seasonal_amplitude'] = normalized_amplitude
                    
                    # Flag as seasonal if amplitude is significant
                    df.loc[company_data.index, 'deposit_is_seasonal'] = normalized_amplitude > 0.20
                    
                    # Add new metric: seasonal expectation
                    # Calculate expected deposit values based on seasonal pattern
                    if normalized_amplitude > 0.15:
                        # Simple seasonal expectation based on day of year
                        days_in_year = 365
                        day_of_year = company_data['date'].dt.dayofyear.values
                        seasonal_component = np.sin(2 * np.pi * day_of_year / days_in_year)
                        expected_seasonal = company_data['deposit_balance'].mean() * (
                            1 + normalized_amplitude * seasonal_component
                        )
                        
                        # Calculate deviation from seasonal expectation
                        actual_values = company_data['deposit_balance'].values
                        seasonal_deviation = np.abs(actual_values - expected_seasonal) / expected_seasonal
                        
                        # Store deviation metric
                        df.loc[company_data.index, 'deposit_seasonal_deviation'] = seasonal_deviation
    except:
        # If seasonality detection fails, continue without it
        pass

def filter_stable_companies(df_with_metrics, recent_risk_df, min_util_change=0.02, months=12):
    """
    Filters out companies whose loan utilization has remained relatively unchanged.
    
    Parameters:
    -----------
    df_with_metrics : pandas.DataFrame
        DataFrame containing metrics including loan utilization
    recent_risk_df : pandas.DataFrame
        DataFrame containing recent risk assessments
    min_util_change : float
        Minimum change in utilization required to include company (default: 0.02 or 2%)
    months : int
        Number of months to look back for utilization changes (default: 12)
        
    Returns:
    --------
    pandas.DataFrame
        Filtered DataFrame containing only companies with significant utilization changes
    """
    if df_with_metrics is None or recent_risk_df is None or recent_risk_df.empty:
        return recent_risk_df
    
    # Get the most recent date in the data
    max_date = df_with_metrics['date'].max()
    
    # Define the lookback period
    lookback_date = max_date - pd.DateOffset(months=months)
    
    # Filter companies based on utilization change
    dynamic_companies = []
    
    for company_id in recent_risk_df['company_id'].unique():
        # Get company metrics during lookback period
        company_data = df_with_metrics[
            (df_with_metrics['company_id'] == company_id) & 
            (df_with_metrics['date'] >= lookback_date)
        ].sort_values('date')
        
        # Skip if not enough data
        if len(company_data) < 2:
            continue
            
        # Calculate max utilization change
        util_max = company_data['loan_utilization'].max()
        util_min = company_data['loan_utilization'].min()
        util_change = util_max - util_min
        
        # Include company if utilization changed more than threshold
        if util_change >= min_util_change:
            dynamic_companies.append(company_id)
    
    # Filter the risk DataFrame to include only dynamic companies
    filtered_risk_df = recent_risk_df[recent_risk_df['company_id'].isin(dynamic_companies)]
    
    return filtered_risk_df

def detect_risk_patterns_efficient(df):
    """
    Efficient implementation of risk pattern detection based on rolling metrics
    and predefined risk rules. Assign each company to a persona based on their pattern.
    """
    risk_records = []
    persona_assignments = []
    
    # Time windows to analyze
    windows = CONFIG['risk']['trend_windows']
    
    # Get the latest date for recent risk calculation
    max_date = df['date'].max()
    recent_cutoff = max_date - pd.Timedelta(days=CONFIG['data']['recent_window'])
    
    # Process each company
    for company in tqdm(df['company_id'].unique(), desc="Detecting risk patterns"):
        company_data = df[df['company_id'] == company].sort_values('date')
        
        # Skip if not enough data points or no deposits
        if len(company_data) < max(windows) or not (company_data['deposit_balance'] > 0).any():
            continue
        
        # Process each date after we have enough history
        for i in range(max(windows), len(company_data), 15):  # Process every 15 days for efficiency
            current_row = company_data.iloc[i]
            current_date = current_row['date']
            
            # Skip older dates for efficiency if not the most recent month
            if i < len(company_data) - 1 and current_date < recent_cutoff:
                continue
            
            # Extract current metrics
            current_util = current_row['loan_utilization']
            current_deposit = current_row['deposit_balance']
            
            # Skip if key metrics are missing
            if pd.isna(current_util) or pd.isna(current_deposit):
                continue
            
            # Initialize risk data
            risk_flags = []
            risk_levels = []
            risk_descriptions = []
            persona = None
            persona_confidence = 0.0
            
            # ---- RISK PATTERN 1: Rising utilization with declining deposits ----
            if not pd.isna(current_row.get('util_change_90d')) and not pd.isna(current_row.get('deposit_change_90d')):
                util_change_90d = current_row['util_change_90d']
                deposit_change_90d = current_row['deposit_change_90d']
                
                if util_change_90d > 0.1 and deposit_change_90d < -0.1:
                    severity = "high" if (util_change_90d > 0.2 and deposit_change_90d < -0.2) else "medium"
                    risk_flags.append('deteriorating_90d')
                    risk_descriptions.append(
                        f"[{severity.upper()}] 90d: Rising utilization (+{util_change_90d:.1%}) "
                        f"with declining deposits ({deposit_change_90d:.1%})"
                    )
                    risk_levels.append(severity)
                    
                    if persona_confidence < 0.8:
                        persona = "deteriorating_health"
                        persona_confidence = 0.8
            
            # ---- RISK PATTERN 2: High utilization with low deposit ratio ----
            if current_util > 0.75 and current_row.get('deposit_loan_ratio', float('inf')) < 0.8:
                severity = "high" if current_util > 0.9 else "medium"
                risk_flags.append('credit_dependent')
                risk_descriptions.append(
                    f"[{severity.upper()}] Current: High loan utilization ({current_util:.1%}) "
                    f"with low deposit coverage (ratio: {current_row.get('deposit_loan_ratio', 0):.2f})"
                )
                risk_levels.append(severity)
                
                if persona_confidence < 0.7:
                    persona = "credit_dependent"
                    persona_confidence = 0.7
            
            # ---- RISK PATTERN 3: Rapid deposit decline with stable utilization ----
            if not pd.isna(current_row.get('deposit_change_30d')) and not pd.isna(current_row.get('util_change_30d')):
                deposit_change_30d = current_row['deposit_change_30d']
                util_change_30d = current_row['util_change_30d']
                
                if deposit_change_30d < -0.15 and abs(util_change_30d) < 0.05:
                    severity = "high" if deposit_change_30d < -0.25 else "medium"
                    risk_flags.append('cash_drain_30d')
                    risk_descriptions.append(
                        f"[{severity.upper()}] 30d: Rapid deposit decline ({deposit_change_30d:.1%}) "
                        f"with stable utilization (change: {util_change_30d:.1%})"
                    )
                    risk_levels.append(severity)
                    
                    if persona_confidence < 0.75:
                        persona = "cash_constrained"
                        persona_confidence = 0.75
            
            # ---- RISK PATTERN 4: Increasing volatility in both metrics ----
            if not pd.isna(current_row.get('util_volatility_30d')) and not pd.isna(current_row.get('deposit_volatility_30d')):
                # Compare current volatility to historical volatility
                current_vol_u = current_row['util_volatility_30d']
                current_vol_d = current_row['deposit_volatility_30d']
                
                # Get historical volatility (from earlier period)
                if i > 90:
                    past_vol_u = company_data.iloc[i-90]['util_volatility_30d']
                    past_vol_d = company_data.iloc[i-90]['deposit_volatility_30d']
                    
                    if not pd.isna(past_vol_u) and not pd.isna(past_vol_d):
                        if current_vol_u > past_vol_u * 1.5 and current_vol_d > past_vol_d * 1.5:
                            risk_flags.append('volatility_increase')
                            risk_descriptions.append(
                                f"[MEDIUM] Significant increase in volatility for both metrics "
                                f"(util: {past_vol_u:.4f}→{current_vol_u:.4f}, deposit: {past_vol_d:.4f}→{current_vol_d:.4f})"
                            )
                            risk_levels.append("medium")
                            
                            if persona_confidence < 0.6:
                                persona = "aggressive_expansion"
                                persona_confidence = 0.6
            
            # ---- RISK PATTERN 5: Loan Utilization Seasonality ----
            if current_row.get('util_is_seasonal') == True:
                amplitude = current_row.get('util_seasonal_amplitude', 0)
                if amplitude > 0.2:  # More than 20% seasonal variation
                    risk_flags.append('seasonal_util')
                    risk_descriptions.append(
                        f"[LOW] Seasonal loan utilization with {amplitude:.1%} amplitude "
                        f"(period: {current_row.get('util_seasonal_period', 0):.0f} days)"
                    )
                    risk_levels.append("low")
                    
                    if persona_confidence < 0.65:
                        persona = "seasonal_loan_user"
                        persona_confidence = 0.65
            
            # ---- RISK PATTERN 6: Deposit Seasonality ----
            if current_row.get('deposit_is_seasonal') == True:
                amplitude = current_row.get('deposit_seasonal_amplitude', 0)
                if amplitude > 0.25:  # More than 25% seasonal variation
                    risk_flags.append('seasonal_deposit')
                    risk_descriptions.append(
                        f"[LOW] Seasonal deposit pattern with {amplitude:.1%} amplitude "
                        f"(period: {current_row.get('deposit_seasonal_period', 0):.0f} days)"
                    )
                    risk_levels.append("low")
                    
                    if persona_confidence < 0.65 and persona != "seasonal_loan_user":
                        persona = "seasonal_deposit_pattern"
                        persona_confidence = 0.65
            
            # ---- RISK PATTERN 7: Combined Seasonal Risk ----
            if (current_row.get('util_is_seasonal') == True and 
                current_row.get('deposit_is_seasonal') == True):
                util_amplitude = current_row.get('util_seasonal_amplitude', 0)
                deposit_amplitude = current_row.get('deposit_seasonal_amplitude', 0)
                
                # If loan volatility is higher than deposit volatility, potential risk
                if util_amplitude > deposit_amplitude * 1.5 and util_amplitude > 0.25:
                    risk_flags.append('seasonal_imbalance')
                    risk_descriptions.append(
                        f"[MEDIUM] Seasonal imbalance: Loan utilization amplitude ({util_amplitude:.1%}) "
                        f"exceeds deposit amplitude ({deposit_amplitude:.1%})"
                    )
                    risk_levels.append("medium")
            
            # ---- NEW RISK PATTERN 8: Loan utilization increasing but deposits stagnant ----
            if not pd.isna(current_row.get('util_change_90d')) and not pd.isna(current_row.get('deposit_change_90d')):
                util_change_90d = current_row['util_change_90d']
                deposit_change_90d = current_row['deposit_change_90d']
                
                if util_change_90d > 0.08 and abs(deposit_change_90d) < 0.02:
                    severity = "medium" if util_change_90d > 0.15 else "low"
                    risk_flags.append('stagnant_growth')
                    risk_descriptions.append(
                        f"[{severity.upper()}] 90d: Increasing utilization (+{util_change_90d:.1%}) "
                        f"with stagnant deposits (change: {deposit_change_90d:.1%})"
                    )
                    risk_levels.append(severity)
                    
                    if persona_confidence < 0.7:
                        persona = "stagnant_growth"
                        persona_confidence = 0.7
            
            # ---- NEW RISK PATTERN 9: Sudden spikes in loan utilization ----
            if i >= 7:  # Need at least 7 days of history for short-term spike detection
                # Get short-term utilization history (7 days)
                recent_utils = company_data.iloc[i-7:i+1]['loan_utilization'].values
                if len(recent_utils) >= 2:
                    # Calculate maximum day-to-day change
                    day_changes = np.diff(recent_utils)
                    max_day_change = np.max(day_changes) if len(day_changes) > 0 else 0
                    
                    if max_day_change > 0.15:  # 15% spike in a single day
                        severity = "high" if max_day_change > 0.25 else "medium"
                        risk_flags.append('utilization_spike')
                        risk_descriptions.append(
                            f"[{severity.upper()}] Recent: Sudden utilization spike detected "
                            f"(+{max_day_change:.1%} in a single day)"
                        )
                        risk_levels.append(severity)
                        
                        if persona_confidence < 0.75:
                            persona = "utilization_spikes"
                            persona_confidence = 0.75
            
            # ---- NEW RISK PATTERN 10: Seasonal pattern breaking ----
            if 'deposit_seasonal_deviation' in current_row and not pd.isna(current_row['deposit_seasonal_deviation']):
                seasonal_deviation = current_row['deposit_seasonal_deviation']
                
                if seasonal_deviation > 0.3:  # 30% deviation from expected seasonal pattern
                    severity = "medium" if seasonal_deviation > 0.5 else "low"
                    risk_flags.append('seasonal_break')
                    risk_descriptions.append(
                        f"[{severity.upper()}] Seasonal pattern break: "
                        f"Deposit deviation {seasonal_deviation:.1%} from expected seasonal pattern"
                    )
                    risk_levels.append(severity)
                    
                    if persona_confidence < 0.75:
                        persona = "seasonal_pattern_breaking"
                        persona_confidence = 0.75
            
            # ---- NEW RISK PATTERN 11: Approaching credit limit ----
            if current_util > 0.9:
                # Look at rate of approach to limit
                if i >= 30 and 'loan_utilization' in company_data.columns:
                    past_util = company_data.iloc[i-30]['loan_utilization']
                    util_velocity = (current_util - past_util) / 30  # Daily increase
                    
                    if util_velocity > 0.002:  # More than 0.2% per day increase
                        severity = "high" if current_util > 0.95 else "medium"
                        risk_flags.append('approaching_limit')
                        risk_descriptions.append(
                            f"[{severity.upper()}] Current utilization near limit ({current_util:.1%}) "
                            f"with velocity of +{util_velocity*100:.2f}% per day"
                        )
                        risk_levels.append(severity)
                        
                        if persona_confidence < 0.85:
                            persona = "approaching_limit"
                            persona_confidence = 0.85
            
            # ---- NEW RISK PATTERN 12: Withdrawal intensity ----
            if 'withdrawal_count_change' in current_row and not pd.isna(current_row['withdrawal_count_change']):
                withdrawal_count_change = current_row['withdrawal_count_change']
                withdrawal_avg_change = current_row.get('withdrawal_avg_change', 0)
                
                if withdrawal_count_change > 0.5 or withdrawal_avg_change > 0.3:
                    severity = "medium" if (withdrawal_count_change > 1 or withdrawal_avg_change > 0.5) else "low"
                    risk_flags.append('withdrawal_intensive')
                    risk_descriptions.append(
                        f"[{severity.upper()}] Increased withdrawal activity: "
                        f"Count change +{withdrawal_count_change:.1%}, "
                        f"Average size change +{withdrawal_avg_change:.1%}"
                    )
                    risk_levels.append(severity)
                    
                    if persona_confidence < 0.7:
                        persona = "withdrawal_intensive"
                        persona_confidence = 0.7
            
            # ---- NEW RISK PATTERN 13: Deposit concentration risk ----
            if 'deposit_concentration_gini' in current_row and not pd.isna(current_row['deposit_concentration_gini']):
                gini = current_row['deposit_concentration_gini']
                
                if gini > 0.6:
                    severity = "medium" if gini > 0.75 else "low"
                    risk_flags.append('deposit_concentration')
                    risk_descriptions.append(
                        f"[{severity.upper()}] Deposit concentration detected: "
                        f"Concentration index {gini:.2f}"
                    )
                    risk_levels.append(severity)
                    
                    if persona_confidence < 0.65:
                        persona = "deposit_concentration"
                        persona_confidence = 0.65
            
            # ---- NEW RISK PATTERN 14: Deposit balance below historical low with high utilization ----
            if 'deposit_to_min_ratio' in current_row and not pd.isna(current_row['deposit_to_min_ratio']):
                min_ratio = current_row['deposit_to_min_ratio']
                
                if min_ratio < 1.1 and current_util > 0.7:
                    severity = "high" if min_ratio <= 1.0 else "medium"
                    risk_flags.append('historical_low_deposits')
                    risk_descriptions.append(
                        f"[{severity.upper()}] Deposits near historical low "
                        f"({min_ratio:.2f}x minimum) with high utilization ({current_util:.1%})"
                    )
                    risk_levels.append(severity)
                    
                    if persona_confidence < 0.8:
                        persona = "historical_low_deposits"
                        persona_confidence = 0.8
            
            # If any risks were detected, record them
            if risk_flags:
                # Determine overall risk level
                overall_risk = "low"
                if "high" in risk_levels:
                    overall_risk = "high"
                elif "medium" in risk_levels:
                    overall_risk = "medium"
                
                # If no persona was assigned, use default based on utilization
                if persona is None:
                    if current_util < 0.4:
                        persona = "cautious_borrower"
                    elif current_util > 0.8:
                        persona = "distressed_client"
                    else:
                        persona = "credit_dependent"
                
                risk_records.append({
                    'company_id': company,
                    'date': current_date,
                    'risk_flags': '|'.join(risk_flags),
                    'risk_description': ' | '.join(risk_descriptions),
                    'risk_level': overall_risk,
                    'persona': persona,
                    'confidence': persona_confidence,
                    'current_util': current_util,
                    'current_deposit': current_deposit,
                    'is_recent': current_date >= recent_cutoff
                })
                
                # Record persona assignment for cohort analysis
                persona_assignments.append({
                    'company_id': company,
                    'date': current_date,
                    'persona': persona,
                    'confidence': persona_confidence,
                    'risk_level': overall_risk,
                    'is_recent': current_date >= recent_cutoff
                })
    
    # Create risk dataframe
    if risk_records:
        risk_df = pd.DataFrame(risk_records)
        persona_df = pd.DataFrame(persona_assignments)
        
        # Create a dataframe of recent risks (last 30 days)
        recent_risks = risk_df[risk_df['is_recent'] == True].copy()
        
        # For each company, find the most frequent risk flag in the recent period
        recent_company_risks = []
        for company in recent_risks['company_id'].unique():
            company_recent = recent_risks[recent_risks['company_id'] == company]
            
            # Get all risk flags
            all_flags = []
            for flags in company_recent['risk_flags']:
                all_flags.extend(flags.split('|'))
            
            if not all_flags:
                continue
                
            # Count flag occurrences
            flag_counts = pd.Series(all_flags).value_counts()
            most_common_flag = flag_counts.index[0]
            
            # Get the most recent risk entry for this company
            latest_entry = company_recent.sort_values('date').iloc[-1]
            
            recent_company_risks.append({
                'company_id': company,
                'latest_date': latest_entry['date'],
                'most_common_flag': most_common_flag,
                'risk_level': latest_entry['risk_level'],
                'persona': latest_entry['persona'],
                'current_util': latest_entry['current_util'],
                'current_deposit': latest_entry['current_deposit']
            })
        
        recent_risk_summary = pd.DataFrame(recent_company_risks)
        
        return risk_df, persona_df, recent_risk_summary
    else:
        # Return empty dataframes with correct columns
        risk_df = pd.DataFrame(columns=['company_id', 'date', 'risk_flags', 'risk_description', 
                                        'risk_level', 'persona', 'confidence', 'current_util', 
                                        'current_deposit', 'is_recent'])
        persona_df = pd.DataFrame(columns=['company_id', 'date', 'persona', 'confidence', 
                                          'risk_level', 'is_recent'])
        recent_risk_summary = pd.DataFrame(columns=['company_id', 'latest_date', 'most_common_flag',
                                                   'risk_level', 'persona', 'current_util', 'current_deposit'])
        return risk_df, persona_df, recent_risk_summary

def filter_high_risk_companies(risk_df, persona_df, min_confidence=0.7):
    """
    Filters persona data to focus only on high-risk companies with high confidence.
    
    Parameters:
    -----------
    risk_df : pandas.DataFrame
        DataFrame containing risk assessments
    persona_df : pandas.DataFrame
        DataFrame containing persona assignments
    min_confidence : float
        Minimum confidence threshold for persona assignment (default: 0.7)
        
    Returns:
    --------
    pandas.DataFrame
        Filtered persona DataFrame with only high-risk, high-confidence entries
    """
    if risk_df is None or persona_df is None or risk_df.empty or persona_df.empty:
        return None
    
    # Get high-risk company IDs
    high_risk_companies = risk_df[risk_df['risk_level'] == 'high']['company_id'].unique()
    
    if len(high_risk_companies) == 0:
        print("No high-risk companies found. Falling back to medium risk.")
        high_risk_companies = risk_df[risk_df['risk_level'] == 'medium']['company_id'].unique()
    
    # Filter persona DataFrame for high-risk companies and high confidence
    high_risk_persona_df = persona_df[
        (persona_df['company_id'].isin(high_risk_companies)) &
        (persona_df['confidence'] >= min_confidence)
    ].copy()
    
    return high_risk_persona_df

def calculate_persona_affinity(persona_df):
    """
    Calculate affinity scores for each company-persona combination.
    This measures how strongly a company is associated with each persona over time.
    """
    if persona_df.empty:
        return pd.DataFrame()
    
    # Calculate persona frequency for each company
    affinity_scores = []
    
    for company in persona_df['company_id'].unique():
        company_data = persona_df[persona_df['company_id'] == company]
        
        # Count occurrences of each persona
        persona_counts = company_data['persona'].value_counts()
        total_records = len(company_data)
        
        # Calculate affinity for each persona
        for persona, count in persona_counts.items():
            # Calculate frequency
            frequency = count / total_records
            
            # Calculate weighted confidence (average confidence for this persona)
            avg_confidence = company_data[company_data['persona'] == persona]['confidence'].mean()
            
            # Calculate recency factor (more weight to recent occurrences)
            recent_data = company_data[company_data['is_recent'] == True]
            recent_count = len(recent_data[recent_data['persona'] == persona])
            total_recent = len(recent_data) if len(recent_data) > 0 else 1
            recency_factor = recent_count / total_recent if total_recent > 0 else 0
            
            # Calculate combined affinity score
            affinity = (frequency * 0.5) + (avg_confidence * 0.3) + (recency_factor * 0.2)
            
            # Get the most recent risk level for this persona
            if len(company_data[company_data['persona'] == persona]) > 0:
                latest_entry = company_data[company_data['persona'] == persona].sort_values('date').iloc[-1]
                risk_level = latest_entry['risk_level']
            else:
                risk_level = 'low'
            
            affinity_scores.append({
                'company_id': company,
                'persona': persona,
                'frequency': frequency,
                'avg_confidence': avg_confidence,
                'recency_factor': recency_factor,
                'affinity_score': affinity,
                'risk_level': risk_level
            })
    
    affinity_df = pd.DataFrame(affinity_scores)
    return affinity_df

def track_persona_transitions(persona_df):
    """
    Track transitions between personas with focus on movements to riskier personas.
    """
    if persona_df.empty:
        return None, None
    
    # Create risk score mapping
    risk_scores = CONFIG['risk']['risk_levels']
    
    # Create a risk score for each persona based on common risk levels
    persona_risk_scores = {}
    for persona in persona_df['persona'].unique():
        risk_levels = persona_df[persona_df['persona'] == persona]['risk_level'].value_counts()
        if not risk_levels.empty:
            most_common_risk = risk_levels.index[0]
            persona_risk_scores[persona] = risk_scores.get(most_common_risk, 0)
        else:
            persona_risk_scores[persona] = 0
    
    # Track transitions between personas
    transitions = []
    risk_increases = []
    
    for company_id, company_data in persona_df.groupby('company_id'):
        company_data = company_data.sort_values('date')
        
        # Skip if only one persona record
        if len(company_data) < 2:
            continue
            
        # Track transitions between consecutive records
        for i in range(len(company_data) - 1):
            from_persona = company_data.iloc[i]['persona']
            to_persona = company_data.iloc[i+1]['persona']
            from_date = company_data.iloc[i]['date']
            to_date = company_data.iloc[i+1]['date']
            
            # Get risk scores
            from_risk = persona_risk_scores.get(from_persona, 0)
            to_risk = persona_risk_scores.get(to_persona, 0)
            
            # Record transition
            transitions.append({
                'company_id': company_id,
                'from_date': from_date,
                'to_date': to_date,
                'from_persona': from_persona,
                'to_persona': to_persona,
                'risk_change': to_risk - from_risk
            })
            
            # Record significant risk increases
            if to_risk > from_risk and (to_date - from_date).days <= 90:
                risk_increases.append({
                    'company_id': company_id,
                    'from_date': from_date,
                    'to_date': to_date,
                    'from_persona': from_persona,
                    'to_persona': to_persona,
                    'risk_change': to_risk - from_risk,
                    'days_between': (to_date - from_date).days
                })
    
    transitions_df = pd.DataFrame(transitions)
    risk_increase_df = pd.DataFrame(risk_increases)
    
    return transitions_df, risk_increase_df

def create_personas_cohort_analysis(persona_df):
    """
    Create cohort analysis based on personas over time.
    Track how personas evolve over different time periods.
    """
    if persona_df.empty:
        print("No persona data available for cohort analysis.")
        return None, None, None
    
    # Convert date to period (month, quarter)
    persona_df['month'] = persona_df['date'].dt.to_period('M')
    persona_df['quarter'] = persona_df['date'].dt.to_period('Q')
    
    # For each company, get the dominant persona per quarter
    quarterly_personas = []
    
    for company_id, company_data in persona_df.groupby('company_id'):
        for quarter, quarter_data in company_data.groupby('quarter'):
            # Get most frequent persona with highest confidence
            persona_counts = quarter_data.groupby('persona')['confidence'].mean().reset_index()
            if not persona_counts.empty:
                dominant_persona = persona_counts.loc[persona_counts['confidence'].idxmax()]['persona']
                
                # Get most severe risk level
                if 'high' in quarter_data['risk_level'].values:
                    risk_level = 'high'
                elif 'medium' in quarter_data['risk_level'].values:
                    risk_level = 'medium'
                else:
                    risk_level = 'low'
                
                quarterly_personas.append({
                    'company_id': company_id,
                    'quarter': quarter,
                    'persona': dominant_persona,
                    'risk_level': risk_level
                })
    
    quarterly_persona_df = pd.DataFrame(quarterly_personas)
    
    if quarterly_persona_df.empty:
        return None, None, None
    
    # Create cohort analysis - count companies per persona per quarter
    cohort_data = quarterly_persona_df.pivot_table(
        index='quarter', 
        columns='persona', 
        values='company_id',
        aggfunc='count',
        fill_value=0
    )
    
    # Also create risk-level cohort analysis
    risk_cohort_data = quarterly_persona_df.pivot_table(
        index='quarter', 
        columns='risk_level', 
        values='company_id',
        aggfunc='count',
        fill_value=0
    )
    
    return cohort_data, risk_cohort_data, quarterly_persona_df

def plot_persona_cohort_enhanced(cohort_data):
    """
    Create an enhanced visualization of persona-based cohort analysis with improved aesthetics.
    """
    if cohort_data is None or cohort_data.empty:
        print("No cohort data available for plotting.")
        return None
    
    # Prepare data
    cohort_pct = cohort_data.div(cohort_data.sum(axis=1), axis=0)
    
    # Create figure with subplots - absolute counts and percentages
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 12), sharex=True)
    
    # Define a visually appealing color palette
    palette = sns.color_palette("viridis", len(cohort_data.columns))
    
    # Plot absolute values as stacked area chart
    cohort_data.plot(kind='area', stacked=True, alpha=0.8, ax=ax1, color=palette)
    
    # Plot percentages as stacked area chart
    cohort_pct.plot(kind='area', stacked=True, alpha=0.8, ax=ax2, color=palette)
    
    # Format the percentage y-axis
    ax2.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    
    # Add titles and labels
    ax1.set_title('Client Personas Over Time (Absolute Count)', fontsize=16, fontweight='bold')
    ax2.set_title('Client Personas Over Time (Percentage)', fontsize=16, fontweight='bold')
    ax1.set_ylabel('Number of Companies', fontsize=14)
    ax2.set_ylabel('Percentage of Companies', fontsize=14)
    ax2.set_xlabel('Quarter', fontsize=14)
    
    # Improve grid appearance
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax2.grid(True, alpha=0.3, linestyle='--')
    
    # Add annotations for absolute values
    for i, quarter in enumerate(cohort_data.index):
        y_pos = 0
        for persona in cohort_data.columns:
            count = cohort_data.loc[quarter, persona]
            if count > 0 and count > cohort_data.sum(axis=1).max() * 0.05:  # Only annotate significant values
                ax1.annotate(f"{int(count)}", 
                            xy=(i, y_pos + count/2), 
                            ha='center', 
                            va='center',
                            fontsize=9,
                            fontweight='bold',
                            color='white',
                            bbox=dict(boxstyle="round,pad=0.2", fc='black', alpha=0.6))
            y_pos += count
    
    # Add annotations for percentages
    for i, quarter in enumerate(cohort_pct.index):
        y_pos = 0
        for persona in cohort_pct.columns:
            pct = cohort_pct.loc[quarter, persona]
            if pct > 0.05:  # Only annotate if > 5%
                ax2.annotate(f"{pct:.1%}", 
                            xy=(i, y_pos + pct/2), 
                            ha='center', 
                            va='center',
                            fontsize=9,
                            fontweight='bold',
                            color='white',
                            bbox=dict(boxstyle="round,pad=0.2", fc='black', alpha=0.6))
            y_pos += pct
    
    # Add a legend with persona descriptions
    handles, labels = ax2.get_legend_handles_labels()
    
    # Create a custom legend with descriptions
    legend_entries = []
    for label in labels:
        description = CONFIG['risk']['persona_patterns'].get(label, "")
        legend_entries.append(f"{label}: {description}")
    
    # Place the legend outside the plot
    fig.legend(handles, legend_entries, loc='upper center', bbox_to_anchor=(0.5, 0), 
              fontsize=11, ncol=2, frameon=True, fancybox=True, shadow=True)
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.25)  # Make room for the legend
    
    return fig

def plot_high_risk_persona_flow(persona_df, df_with_metrics, risk_df, months=6):
    """
    Creates a visualization that tracks high-risk clients as they move between personas
    over a specified time period with monthly granularity.
    
    Parameters:
    -----------
    persona_df : pandas.DataFrame
        DataFrame containing persona assignments over time
    df_with_metrics : pandas.DataFrame
        DataFrame with all metrics including loan utilization data
    risk_df : pandas.DataFrame
        DataFrame containing risk assessments
    months : int
        Number of months to include in the visualization (default: 6)
    
    Returns:
    --------
    matplotlib.figure.Figure
        The figure object containing the visualization
    """
    if persona_df is None or persona_df.empty:
        print("No persona data available for flow analysis.")
        return None
    
    # Get the latest date in the data
    max_date = persona_df['date'].max()
    
    # Calculate the start date for our analysis period
    start_date = max_date - pd.DateOffset(months=months)
    
    # Filter for only the data within our time range
    recent_persona_df = persona_df[persona_df['date'] >= start_date].copy()
    
    # Filter for only high-risk clients
    high_risk_companies = risk_df[
        (risk_df['risk_level'] == 'high') & 
        (risk_df['date'] >= start_date)
    ]['company_id'].unique()
    
    if len(high_risk_companies) == 0:
        print("No high-risk companies found in the selected time period.")
        return None
    
    # Filter for only high-risk companies
    high_risk_persona_df = recent_persona_df[recent_persona_df['company_id'].isin(high_risk_companies)].copy()
    
    # Check if we have enough data
    if high_risk_persona_df.empty:
        print("No persona data available for high-risk companies.")
        return None
    
    # Add month as a period column for easier grouping
    high_risk_persona_df['month'] = high_risk_persona_df['date'].dt.to_period('M')
    
    # For each month, get the dominant persona for each company
    monthly_personas = []
    
    for company_id, company_data in high_risk_persona_df.groupby('company_id'):
        # Check if loan utilization has remained unchanged (within 2%)
        if df_with_metrics is not None:
            company_metrics = df_with_metrics[df_with_metrics['company_id'] == company_id].copy()
            if not company_metrics.empty:
                company_metrics = company_metrics.sort_values('date')
                
                # Get utilization data for the last 12 months
                yearly_cutoff = max_date - pd.DateOffset(months=12)
                recent_utilization = company_metrics[company_metrics['date'] >= yearly_cutoff]['loan_utilization']
                
                if len(recent_utilization) >= 2:
                    # Calculate max change in utilization
                    max_util = recent_utilization.max()
                    min_util = recent_utilization.min()
                    
                    # Skip if utilization change is less than 2%
                    if max_util - min_util < 0.02:
                        continue
        
        # For each month, find the dominant persona with higher confidence threshold
        for month, month_data in company_data.groupby('month'):
            # Only include if confidence is high (>0.7) - stricter classification
            high_conf_data = month_data[month_data['confidence'] > 0.7]
                
            if not high_conf_data.empty:
                # Get the persona with highest confidence
                best_persona = high_conf_data.loc[high_conf_data['confidence'].idxmax()]
                
                monthly_personas.append({
                    'company_id': company_id,
                    'month': month,
                    'persona': best_persona['persona'],
                    'confidence': best_persona['confidence'],
                    'risk_level': best_persona['risk_level']
                })
    
    monthly_persona_df = pd.DataFrame(monthly_personas)
    
    if monthly_persona_df.empty:
        print("No high-confidence persona data available for high-risk companies.")
        return None
    
    # Convert the month period back to datetime for plotting
    monthly_persona_df['month_dt'] = monthly_persona_df['month'].dt.to_timestamp()
    
    # Sort the months in chronological order
    all_months = sorted(monthly_persona_df['month'].unique())
    
    # Get unique personas after filtering
    unique_personas = monthly_persona_df['persona'].unique()
    
    if len(unique_personas) < 2:
        print("Not enough different personas to visualize transitions.")
        return None
    
    # Create figure with appropriate size based on number of personas and months
    fig_height = max(8, len(unique_personas) * 1.5)
    fig_width = max(12, months * 2)
    
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    
    # Create a colormap for personas
    persona_colors = dict(zip(unique_personas, 
                              sns.color_palette("viridis", len(unique_personas))))
    
    # Track company paths between personas
    company_paths = {}
    
    # For each company, track their persona path across months
    for company_id, company_data in monthly_persona_df.groupby('company_id'):
        company_data = company_data.sort_values('month')
        
        # Need at least two months of data to show transition
        if len(company_data) >= 2:
            path = []
            
            # Create path with (month_index, persona) tuples
            for _, row in company_data.iterrows():
                month_idx = all_months.index(row['month'])
                persona = row['persona']
                path.append((month_idx, persona))
            
            company_paths[company_id] = path
    
    # Count how many companies in each persona for each month
    persona_counts = {}
    
    for month in all_months:
        month_data = monthly_persona_df[monthly_persona_df['month'] == month]
        counts = month_data['persona'].value_counts().to_dict()
        persona_counts[month] = counts
    
    # Create month positions on x-axis
    month_positions = np.arange(len(all_months))
    
    # Create persona positions on y-axis (with space between personas)
    persona_spacing = 2
    persona_positions = {}
    for i, persona in enumerate(unique_personas):
        persona_positions[persona] = i * persona_spacing
    
    # Plot company movements between personas
    for company_id, path in company_paths.items():
        # Skip if the path is too short
        if len(path) < 2:
            continue
        
        for i in range(len(path) - 1):
            start_month, start_persona = path[i]
            end_month, end_persona = path[i+1]
            
            # Skip if same persona (no movement)
            if start_persona == end_persona:
                continue
            
            # Get coordinates
            x1 = month_positions[start_month]
            y1 = persona_positions[start_persona]
            x2 = month_positions[end_month]
            y2 = persona_positions[end_persona]
            
            # Calculate line width based on confidence
            company_month_start = all_months[start_month]
            company_month_end = all_months[end_month]
            
            confidence_start = monthly_persona_df[
                (monthly_persona_df['company_id'] == company_id) & 
                (monthly_persona_df['month'] == company_month_start)
            ]['confidence'].values[0]
            
            confidence_end = monthly_persona_df[
                (monthly_persona_df['company_id'] == company_id) & 
                (monthly_persona_df['month'] == company_month_end)
            ]['confidence'].values[0]
            
            # Average confidence affects line transparency
            avg_confidence = (confidence_start + confidence_end) / 2
            
            # Plot connecting line - more confident transitions are more visible
            ax.plot([x1, x2], [y1, y2], 'k-', alpha=min(0.7, avg_confidence), 
                    linewidth=1.5, zorder=1)
    
    # Plot persona nodes for each month
    node_radius_scale = 30  # Scale for node size
    max_radius = 25  # Maximum node radius
    min_radius = 8   # Minimum node radius
    
    for month_idx, month in enumerate(all_months):
        month_data = monthly_persona_df[monthly_persona_df['month'] == month]
        
        for persona in unique_personas:
            persona_data = month_data[month_data['persona'] == persona]
            
            if not persona_data.empty:
                count = len(persona_data)
                
                # Calculate node size based on count
                radius = min(max_radius, max(min_radius, np.sqrt(count) * node_radius_scale / np.sqrt(len(high_risk_companies))))
                
                # Get coordinates
                x = month_positions[month_idx]
                y = persona_positions[persona]
                
                # Only plot if there's at least one company
                if count > 0:
                    ax.scatter(x, y, s=radius**2, color=persona_colors[persona], 
                               edgecolors='black', linewidth=1.5, zorder=2)
                    
                    # Add count label
                    ax.text(x, y, str(count), ha='center', va='center', 
                            fontsize=9, fontweight='bold', color='white', zorder=3)
    
    # Add text labels for personas on the right side with detailed descriptions
    for persona, y_pos in persona_positions.items():
        # Get description from CONFIG
        description = CONFIG['risk']['persona_patterns'].get(persona, "")
        shortened_desc = description.split(':')[0] if ':' in description else description
        
        ax.text(month_positions[-1] + 0.5, y_pos, f"{persona}: {shortened_desc}", 
                va='center', ha='left', fontsize=10, bbox=dict(
                    boxstyle="round,pad=0.3", fc=persona_colors[persona], alpha=0.7, 
                    ec="black", lw=1))
    
    # Add month labels
    month_labels = [m.strftime('%b %Y') for m in [p.to_timestamp() for p in all_months]]
    ax.set_xticks(month_positions)
    ax.set_xticklabels(month_labels, rotation=45, ha='right')
    
    # Remove y-axis ticks
    ax.set_yticks([])
    
    # Set axis limits with some padding
    ax.set_xlim(month_positions[0] - 0.5, month_positions[-1] + 3)
    
    if persona_positions:
        y_min = min(persona_positions.values()) - 2
        y_max = max(persona_positions.values()) + 2
        ax.set_ylim(y_min, y_max)
    
    # Add grid for x-axis only
    ax.grid(axis='x', linestyle='--', alpha=0.7)
    
    # Add title and labels
    title = f"High-Risk Client Movement Between Personas\n(Last {months} Months, {len(company_paths)} Companies)"
    ax.set_title(title, fontsize=16, fontweight='bold')
    ax.set_xlabel('Month', fontsize=14)
    
    # Add annotation about persona filter criteria
    filter_text = (
        "Note: Only showing high-risk clients with >2% loan utilization change\n"
        "and high persona confidence (>0.7). Line transparency shows confidence strength."
    )
    plt.figtext(0.5, 0.01, filter_text, ha='center', fontsize=10, 
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15, right=0.85)  # Make room for the annotation and labels
    
    return fig

def generate_bank_client_data(num_companies=50, days=1460):  # 4 years of data
    """
    Simple data generator to test the bank client risk analysis algorithm.
    
    Parameters:
    -----------
    num_companies : int
        Number of companies to generate data for
    days : int
        Number of days of historical data to generate
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame containing synthetic bank client data
    """
    print("Generating test bank client data...")
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Create date range
    end_date = pd.Timestamp('2022-12-31')
    start_date = end_date - pd.Timedelta(days=days)
    date_range = pd.date_range(start=start_date, end=end_date)
    
    # Create company IDs
    company_ids = [f'COMP{str(i).zfill(4)}' for i in range(num_companies)]
    
    # Create dataframe
    data = []
    
    for company_id in tqdm(company_ids, desc="Generating company data"):
        # Generate random parameters for this company
        base_deposit = np.random.lognormal(10, 1)  # Random base deposit amount
        base_loan = np.random.lognormal(9, 1.2)    # Random base loan amount
        util_rate = np.random.uniform(0.3, 0.8)    # Initial utilization rate
        
        # Trend parameters
        deposit_trend = np.random.normal(0, 0.001)  # Slight upward or downward trend
        util_trend = np.random.normal(0, 0.0005)    # Utilization trend
        
        # Seasonal parameters
        has_seasonality = np.random.random() < 0.3  # 30% of companies have seasonality
        seasonal_amp = np.random.uniform(0.05, 0.2) if has_seasonality else 0
        
        # Volatility parameters
        deposit_vol = np.random.uniform(0.01, 0.1)  # Deposit volatility
        util_vol = np.random.uniform(0.01, 0.05)    # Utilization volatility
        
        # Risk pattern (25% of companies develop risk pattern)
        has_risk = np.random.random() < 0.25
        risk_start = int(len(date_range) * 0.7) if has_risk else len(date_range)
        
        # Enhanced risk patterns (15% of companies show withdrawal patterns)
        has_withdrawal_pattern = np.random.random() < 0.15
        withdrawal_start = int(len(date_range) * 0.8) if has_withdrawal_pattern else len(date_range)
        
        # Companies with deposit concentration risk (10%)
        has_concentration_risk = np.random.random() < 0.1
        
        # Companies approaching credit limit (8%)
        approaching_limit = np.random.random() < 0.08
        limit_start = int(len(date_range) * 0.85) if approaching_limit else len(date_range)
        
        # Generate time series
        for i, date in enumerate(date_range):
            # Create time-dependent components
            t = i / len(date_range)  # Normalized time
            
            # Trends
            deposit_trend_component = 1 + deposit_trend * i
            util_trend_component = util_rate + util_trend * i
            
            # Seasonality (if applicable)
            day_of_year = date.dayofyear
            seasonal_component = 1 + seasonal_amp * np.sin(2 * np.pi * day_of_year / 365)
            
            # Volatility (random variation)
            deposit_random = np.random.normal(1, deposit_vol)
            util_random = np.random.normal(0, util_vol)
            
            # Risk pattern after risk_start
            if i > risk_start:
                # Deteriorating pattern: deposit down, utilization up
                risk_factor_deposit = 1 - 0.001 * (i - risk_start)
                risk_factor_util = 0.0005 * (i - risk_start)
            else:
                risk_factor_deposit = 1
                risk_factor_util = 0
            
            # Withdrawal pattern after withdrawal_start
            if has_withdrawal_pattern and i > withdrawal_start:
                # More frequent withdrawals
                if np.random.random() < 0.2:  # 20% chance of withdrawal event
                    withdrawal_size = np.random.uniform(0.05, 0.15)
                    deposit_random *= (1 - withdrawal_size)
            
            # Companies approaching limit
            if approaching_limit and i > limit_start:
                util_trend_component = min(0.95, util_trend_component + 0.002 * (i - limit_start))
            
            # Calculate final values
            deposit = base_deposit * deposit_trend_component * seasonal_component * deposit_random * risk_factor_deposit
            
            # Add deposit concentration for some companies
            if has_concentration_risk and np.random.random() < 0.05:  # 5% chance of large deposit
                deposit *= np.random.uniform(1.5, 3.0)
            
            utilization = min(0.95, max(0.1, util_trend_component + util_random + risk_factor_util))
            used_loan = base_loan * utilization
            unused_loan = base_loan - used_loan
            
            # Add some missing values (5% probability)
            if np.random.random() < 0.05:
                if np.random.random() < 0.5:
                    deposit = 0
                else:
                    used_loan = 0
                    unused_loan = 0
            
            data.append({
                'company_id': company_id,
                'date': date,
                'deposit_balance': max(0, deposit),
                'used_loan': max(0, used_loan),
                'unused_loan': max(0, unused_loan)
            })
    
    return pd.DataFrame(data)

def bank_client_risk_analysis(df):
    """
    Main function to execute the entire bank client risk analysis workflow with improved risk detection,
    persona-based cohort analysis, and enhanced visualizations.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing bank client data with columns:
        - company_id: Identifier for the company
        - date: Date of the data point
        - deposit_balance: Balance of deposits
        - used_loan: Amount of loan used
        - unused_loan: Amount of loan unused
        
    Returns:
    --------
    dict
        Dictionary containing all the analysis results
    """
    print("Starting enhanced bank client risk analysis...")
    
    # 1. Clean data with enhanced imputation techniques
    print("\nCleaning data and applying advanced imputation...")
    df_clean, df_calc = clean_data(df, min_nonzero_pct=CONFIG['data']['min_nonzero_pct'])
    
    # 2. Add derived metrics with enhanced features, including seasonality detection
    print("\nAdding derived metrics and detecting seasonality...")
    df_with_metrics = add_derived_metrics(df_clean)
    
    # 3. Detect risk patterns and assign personas
    print("\nDetecting risk patterns and assigning personas...")
    risk_df, persona_df, recent_risk_df = detect_risk_patterns_efficient(df_with_metrics)
    print(f"Found {len(risk_df)} risk events across {risk_df['company_id'].nunique()} companies")
    print(f"Identified {len(recent_risk_df)} companies with recent risk events")
    print(f"Assigned {persona_df['persona'].nunique() if not persona_df.empty else 0} different personas")
    
    # 4. Filter out stable companies (less than 2% loan utilization change)
    print("\nFiltering out companies with stable loan utilization...")
    dynamic_risk_df = filter_stable_companies(df_with_metrics, recent_risk_df, min_util_change=0.02, months=12)
    print(f"Filtered to {len(dynamic_risk_df)} companies with significant utilization changes")
    
    # 5. Filter for high-risk companies with high confidence
    print("\nFiltering for high-risk companies with high confidence personas...")
    high_risk_persona_df = filter_high_risk_companies(risk_df, persona_df, min_confidence=0.7)
    if high_risk_persona_df is not None:
        print(f"Found {high_risk_persona_df['company_id'].nunique()} high-risk companies with confident persona assignments")
    else:
        print("No high-risk companies found with confident persona assignments")
    
    # 6. Calculate persona affinity scores
    print("\nCalculating persona affinity scores...")
    affinity_df = calculate_persona_affinity(persona_df)
    print(f"Calculated affinity scores for {len(affinity_df['company_id'].unique()) if not affinity_df.empty else 0} companies")
    
    # 7. Track persona transitions with focus on risk increases
    print("\nTracking persona transitions...")
    transitions_df, risk_increase_df = track_persona_transitions(persona_df)
    print(f"Identified {len(risk_increase_df) if risk_increase_df is not None else 0} risk-increasing transitions")
    
    # 8. Create persona-based cohort analysis
    print("\nCreating persona-based cohort analysis...")
    cohort_results = create_personas_cohort_analysis(persona_df)
    if cohort_results[0] is not None:
        cohort_data, risk_cohort_data, quarterly_persona_df = cohort_results
        print(f"Created cohort analysis across {len(cohort_data)} time periods and {len(cohort_data.columns)} personas")
    else:
        cohort_data, risk_cohort_data, quarterly_persona_df = None, None, None
        print("No cohort analysis created due to insufficient data")
    
    # 9. Plot high-risk persona flow
    print("\nPlotting high-risk client persona flow...")
    high_risk_flow_fig = plot_high_risk_persona_flow(
        persona_df, df_with_metrics, risk_df, months=6
    )
    
    if high_risk_flow_fig:
        plt.figure(high_risk_flow_fig.number)
        plt.savefig('high_risk_client_flow.png')
        print("Saved high-risk client persona flow to high_risk_client_flow.png")
    
    # 10. Plot persona cohort analysis
    if cohort_data is not None:
        print("\nPlotting persona cohort analysis...")
        cohort_fig = plot_persona_cohort_enhanced(cohort_data)
        if cohort_fig:
            plt.figure(cohort_fig.number)
            plt.savefig('persona_cohort_analysis.png')
            print("Saved persona cohort analysis to persona_cohort_analysis.png")
    
    return {
        'data': df_with_metrics,
        'risk_df': risk_df,
        'persona_df': persona_df,
        'recent_risk_df': recent_risk_df,
        'dynamic_risk_df': dynamic_risk_df,
        'high_risk_persona_df': high_risk_persona_df,       
        'affinity_df': affinity_df,
        'transitions_df': transitions_df,
        'risk_increase_df': risk_increase_df,
        'quarterly_persona_df': quarterly_persona_df
    }


#######################################################
# PART 2: CREDIT SCORE BACKTEST FRAMEWORK
#######################################################

def convert_credit_score_to_numeric(score):
    """
    Convert credit score to numeric value for comparison.
    Higher numeric value = better credit score.
    """
    # Define mapping of scores to numeric values
    score_mapping = {
        '1+': 23, '1': 22, 
        '2+': 21, '2': 20, '2-': 19,
        '3+': 18, '3': 17, '3-': 16,
        '4+': 15, '4': 14, '4-': 13,
        '5+': 12, '5': 11, '5-': 10,
        '6+': 9, '6': 8, '6-': 7,
        '7': 6, '8': 5, '9': 4, '10': 3,
        'NC': 2, 'NR': 1
    }
    return score_mapping.get(str(score), 0)

def detect_credit_downgrades(credit_score_df, lookback_years=1):
    """
    Detect credit score downgrades within the specified lookback period.
    
    Parameters:
    -----------
    credit_score_df : pandas.DataFrame
        DataFrame containing credit score data with columns:
        - company_id: Identifier for the company
        - date: Date of the credit score
        - credit_score: Credit score value
        - industry: Industry of the company
    lookback_years : int
        Number of years to look back for downgrades (default: 1)
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame containing downgrade events
    """
    # Calculate cutoff date for lookback period
    end_date = credit_score_df['date'].max()
    start_date = end_date - pd.Timedelta(days=365*lookback_years)
    
    # Filter for data within lookback period
    recent_data = credit_score_df[credit_score_df['date'] >= start_date].copy()
    
    downgrades = []
    
    # Process each company
    for company_id, company_data in tqdm(recent_data.groupby('company_id'), desc="Detecting downgrades"):
        # Sort by date
        company_data = company_data.sort_values('date')
        
        # Convert scores to numeric values
        company_data['numeric_score'] = company_data['credit_score'].apply(convert_credit_score_to_numeric)
        
        # Skip if less than 2 records
        if len(company_data) < 2:
            continue
            
        # Check for downgrades
        for i in range(1, len(company_data)):
            prev_score = company_data.iloc[i-1]['credit_score']
            curr_score = company_data.iloc[i]['credit_score']
            prev_numeric = company_data.iloc[i-1]['numeric_score']
            curr_numeric = company_data.iloc[i]['numeric_score']
            
            # If numeric value decreased, it's a downgrade
            if curr_numeric < prev_numeric:
                downgrades.append({
                    'company_id': company_id,
                    'downgrade_date': company_data.iloc[i]['date'],
                    'from_score': prev_score,
                    'to_score': curr_score,
                    'from_numeric': prev_numeric,
                    'to_numeric': curr_numeric,
                    'downgrade_severity': prev_numeric - curr_numeric,
                    'industry': company_data.iloc[i]['industry']
                })
    
    return pd.DataFrame(downgrades)

def analyze_downgrade_distribution_by_industry(downgrade_df, lookback_years=2):
    """
    Analyze the distribution of downgrades by industry over the specified lookback period.
    
    Parameters:
    -----------
    downgrade_df : pandas.DataFrame
        DataFrame containing downgrade events
    lookback_years : int
        Number of years to look back (default: 2)
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame with downgrade statistics by industry
    """
    # Calculate cutoff date
    end_date = downgrade_df['downgrade_date'].max()
    start_date = end_date - pd.Timedelta(days=365*lookback_years)
    
    # Filter for downgrades within the lookback period
    recent_downgrades = downgrade_df[downgrade_df['downgrade_date'] >= start_date].copy()
    
    # Group by industry and calculate statistics
    industry_stats = recent_downgrades.groupby('industry').agg({
        'company_id': 'nunique',
        'downgrade_severity': ['count', 'mean', 'median', 'min', 'max', 'sum']
    }).reset_index()
    
    # Rename columns for clarity
    industry_stats.columns = [
        'industry', 'unique_companies', 'total_downgrades', 
        'avg_severity', 'median_severity', 'min_severity', 
        'max_severity', 'sum_severity'
    ]
    
    # Add percentage column
    total_downgrades = industry_stats['total_downgrades'].sum()
    industry_stats['percentage'] = industry_stats['total_downgrades'] / total_downgrades * 100
    
    # Sort by total downgrades (descending)
    industry_stats = industry_stats.sort_values('total_downgrades', ascending=False)
    
    return industry_stats

def backtest_personas_vs_downgrades(persona_df, risk_df, downgrade_df, lookback_months=6):
    """
    Back-test personas against credit score downgrades to evaluate prediction accuracy.
    
    This function analyzes how well the persona assignments predict actual credit downgrades.
    It identifies cases where a risky persona was assigned before a downgrade occurred,
    calculates the lead time, and evaluates the predictive performance of different personas.
    
    Parameters:
    -----------
    persona_df : pandas.DataFrame
        DataFrame containing persona assignments
    risk_df : pandas.DataFrame
        DataFrame containing risk assessments
    downgrade_df : pandas.DataFrame
        DataFrame containing downgrade events
    lookback_months : int
        Number of months to look back (default: 6)
        
    Returns:
    --------
    tuple
        (performance_df, timing_df)
        - performance_df: DataFrame with performance metrics by persona
        - timing_df: DataFrame with timing analysis
    """
    # Calculate cutoff date
    end_date = max(
        persona_df['date'].max() if not persona_df.empty else datetime.now(),
        downgrade_df['downgrade_date'].max() if not downgrade_df.empty else datetime.now()
    )
    start_date = end_date - pd.Timedelta(days=30*lookback_months)
    
    # Filter for recent data
    recent_persona_df = persona_df[persona_df['date'] >= start_date].copy()
    recent_risk_df = risk_df[risk_df['date'] >= start_date].copy()
    recent_downgrade_df = downgrade_df[downgrade_df['downgrade_date'] >= start_date].copy()
    
    # Get common companies
    common_companies = set(recent_persona_df['company_id'].unique()).intersection(
        set(recent_downgrade_df['company_id'].unique())
    )
    
    print(f"Found {len(common_companies)} companies with both persona data and downgrades")
    
    # If no common companies, return empty results
    if not common_companies:
        return pd.DataFrame(), pd.DataFrame()
    
    # Prepare results containers
    performance_data = []
    timing_data = []
    
    # Define risky personas (these are considered predictive of downgrades)
    risky_personas = [
        'deteriorating_health', 'distressed_client', 'credit_dependent',
        'cash_constrained', 'stagnant_growth', 'utilization_spikes',
        'approaching_limit', 'withdrawal_intensive', 'historical_low_deposits'
    ]
    
    # For each company with downgrades, analyze persona assignments
    for company_id in tqdm(common_companies, desc="Back-testing personas"):
        # Get company downgrades
        company_downgrades = recent_downgrade_df[recent_downgrade_df['company_id'] == company_id].sort_values('downgrade_date')
        
        # Get company persona history
        company_personas = recent_persona_df[recent_persona_df['company_id'] == company_id].sort_values('date')
        
        # Get company risk events
        company_risks = recent_risk_df[recent_risk_df['company_id'] == company_id].sort_values('date')
        
        # For each downgrade, find the persona before and after
        for _, downgrade in company_downgrades.iterrows():
            downgrade_date = downgrade['downgrade_date']
            
            # Find the most recent persona before the downgrade
            before_personas = company_personas[company_personas['date'] < downgrade_date]
            after_personas = company_personas[company_personas['date'] >= downgrade_date]
            
            before_persona = before_personas['persona'].iloc[-1] if not before_personas.empty else None
            before_date = before_personas['date'].iloc[-1] if not before_personas.empty else None
            before_confidence = before_personas['confidence'].iloc[-1] if not before_personas.empty else None
            
            # Find the first persona after the downgrade
            after_persona = after_personas['persona'].iloc[0] if not after_personas.empty else None
            after_date = after_personas['date'].iloc[0] if not after_personas.empty else None
            
            # Calculate days before/after
            days_before = (downgrade_date - before_date).days if before_date is not None else None
            days_after = (after_date - downgrade_date).days if after_date is not None else None
            
            # Find risk level before downgrade
            risk_before = None
            if not company_risks.empty:
                risk_before_data = company_risks[company_risks['date'] < downgrade_date]
                if not risk_before_data.empty:
                    risk_before = risk_before_data['risk_level'].iloc[-1]
            
            # Check if the persona before the downgrade was a risky persona
            correctly_flagged = before_persona in risky_personas if before_persona is not None else False
            
            # Record timing data
            timing_data.append({
                'company_id': company_id,
                'downgrade_date': downgrade_date,
                'from_score': downgrade['from_score'],
                'to_score': downgrade['to_score'],
                'downgrade_severity': downgrade['downgrade_severity'],
                'before_persona': before_persona,
                'before_date': before_date,
                'before_confidence': before_confidence,
                'days_before': days_before,
                'after_persona': after_persona,
                'after_date': after_date,
                'days_after': days_after,
                'correctly_flagged': correctly_flagged,
                'risk_level_before': risk_before,
                'industry': downgrade['industry']
            })
    
    timing_df = pd.DataFrame(timing_data)
    
    # Calculate performance metrics by persona
    if not timing_df.empty:
        # Calculate statistics by persona
        persona_performance = []
        
        for persona in recent_persona_df['persona'].unique():
            # Get all downgrades where this was the persona before
            persona_downgrades = timing_df[timing_df['before_persona'] == persona]
            
            # Calculate metrics
            total_instances = len(persona_downgrades)
            
            if total_instances == 0:
                continue
                
            true_positives = persona_downgrades['correctly_flagged'].sum()
            false_negatives = total_instances - true_positives
            
            # Is this a risky persona?
            is_risky_persona = persona in risky_personas
            
            # Calculate average days before downgrade
            avg_days_before = persona_downgrades['days_before'].mean()
            
            # Calculate average severity for this persona
            avg_severity = persona_downgrades['downgrade_severity'].mean()
            
            # Calculate reliability by confidence level
            high_conf_downgrades = persona_downgrades[persona_downgrades['before_confidence'] >= 0.7]
            high_conf_correctly = high_conf_downgrades['correctly_flagged'].sum() if not high_conf_downgrades.empty else 0
            high_conf_total = len(high_conf_downgrades)
            high_conf_precision = high_conf_correctly / high_conf_total if high_conf_total > 0 else 0
            
            persona_performance.append({
                'persona': persona,
                'total_downgrades': total_instances,
                'correctly_flagged': true_positives,
                'false_negatives': false_negatives,
                'precision': true_positives / total_instances if total_instances > 0 else 0,
                'is_risky_persona': is_risky_persona,
                'avg_days_before_downgrade': avg_days_before,
                'min_days_before': persona_downgrades['days_before'].min(),
                'max_days_before': persona_downgrades['days_before'].max(),
                'avg_severity': avg_severity,
                'high_conf_precision': high_conf_precision
            })
        
        performance_df = pd.DataFrame(persona_performance)
        
        # Add global statistics
        all_downgrades = len(timing_df)
        correctly_flagged = timing_df['correctly_flagged'].sum()
        
        global_stats = {
            'persona': 'OVERALL',
            'total_downgrades': all_downgrades,
            'correctly_flagged': correctly_flagged,
            'false_negatives': all_downgrades - correctly_flagged,
            'precision': correctly_flagged / all_downgrades if all_downgrades > 0 else 0,
            'is_risky_persona': None,
            'avg_days_before_downgrade': timing_df['days_before'].mean(),
            'min_days_before': timing_df['days_before'].min(),
            'max_days_before': timing_df['days_before'].max(),
            'avg_severity': timing_df['downgrade_severity'].mean(),
            'high_conf_precision': timing_df[timing_df['before_confidence'] >= 0.7]['correctly_flagged'].mean() 
                                    if not timing_df[timing_df['before_confidence'] >= 0.7].empty else 0
        }
        
        performance_df = pd.concat([performance_df, pd.DataFrame([global_stats])], ignore_index=True)
        
        # Sort by total downgrades (descending)
        performance_df = performance_df.sort_values('total_downgrades', ascending=False)
    else:
        performance_df = pd.DataFrame()
    
    return performance_df, timing_df

def analyze_lead_time_by_severity(timing_df):
    """
    Analyze the relationship between downgrade severity and lead time.
    
    Parameters:
    -----------
    timing_df : pandas.DataFrame
        DataFrame containing timing analysis
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame with lead time statistics by severity
    """
    if timing_df.empty:
        print("No timing data available for analysis.")
        return pd.DataFrame()
    
    # Group by severity
    severity_stats = timing_df.groupby('downgrade_severity').agg({
        'days_before': ['count', 'mean', 'median', 'std', 'min', 'max'],
        'correctly_flagged': 'mean'  # Proportion correctly flagged
    }).reset_index()
    
    # Rename columns for clarity
    severity_stats.columns = [
        'downgrade_severity', 'count', 'avg_days_before', 'median_days_before',
        'std_days_before', 'min_days_before', 'max_days_before', 'detection_rate'
    ]
    
    # Sort by severity (ascending)
    severity_stats = severity_stats.sort_values('downgrade_severity')
    
    return severity_stats

def analyze_industry_performance(timing_df):
    """
    Analyze performance of the persona model across different industries.
    
    Parameters:
    -----------
    timing_df : pandas.DataFrame
        DataFrame containing timing analysis with industry information
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame with performance metrics by industry
    """
    if timing_df.empty or 'industry' not in timing_df.columns:
        print("No industry data available for performance analysis.")
        return pd.DataFrame()
    
    # Group by industry
    industry_performance = timing_df.groupby('industry').agg({
        'company_id': 'nunique',
        'correctly_flagged': ['count', 'sum', 'mean'],
        'days_before': ['mean', 'median', 'min', 'max'],
        'downgrade_severity': 'mean'
    }).reset_index()
    
    # Rename columns for clarity
    industry_performance.columns = [
        'industry', 'unique_companies', 'total_downgrades', 'correctly_flagged',
        'detection_rate', 'avg_days_before', 'median_days_before',
        'min_days_before', 'max_days_before', 'avg_severity'
    ]
    
    # Calculate additional metrics
    industry_performance['false_negatives'] = industry_performance['total_downgrades'] - industry_performance['correctly_flagged']
    
    # Sort by detection rate (descending)
    industry_performance = industry_performance.sort_values('detection_rate', ascending=False)
    
    return industry_performance

def plot_downgrade_distribution(industry_stats, title="Credit Downgrade Distribution by Industry (Last 2 Years)"):
    """
    Plot the distribution of downgrades by industry.
    
    Parameters:
    -----------
    industry_stats : pandas.DataFrame
        DataFrame with downgrade statistics by industry
        
    Returns:
    --------
    matplotlib.figure.Figure
        The figure object containing the visualization
    """
    if industry_stats.empty:
        print("No industry statistics available for plotting.")
        return None
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
    
    # Top industries by total downgrades (limit to top 10 for readability)
    top_industries = industry_stats.head(10).copy()
    
    # Plot total downgrades by industry
    bars1 = ax1.barh(top_industries['industry'], top_industries['total_downgrades'], color='skyblue')
    ax1.set_title('Total Downgrades by Industry', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Number of Downgrades', fontsize=12)
    ax1.set_ylabel('Industry', fontsize=12)
    
    # Add data labels
    for bar in bars1:
        width = bar.get_width()
        label_x_pos = width if width > 0 else 0
        ax1.text(label_x_pos + 0.5, bar.get_y() + bar.get_height()/2, f'{int(width)}',
                 va='center', fontweight='bold')
    
    # Plot average severity by industry
    bars2 = ax2.barh(top_industries['industry'], top_industries['avg_severity'], color='salmon')
    ax2.set_title('Average Downgrade Severity by Industry', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Average Severity', fontsize=12)
    
    # Add data labels
    for bar in bars2:
        width = bar.get_width()
        label_x_pos = width if width > 0 else 0
        ax2.text(label_x_pos + 0.05, bar.get_y() + bar.get_height()/2, f'{width:.2f}',
                 va='center', fontweight='bold')
    
    # Add overall title
    plt.suptitle(title, fontsize=16, fontweight='bold')
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    
    return fig

def plot_persona_backtest_performance(performance_df, title="Persona Performance in Predicting Credit Downgrades"):
    """
    Plot the performance of personas in predicting credit downgrades.
    
    Parameters:
    -----------
    performance_df : pandas.DataFrame
        DataFrame with performance metrics by persona
        
    Returns:
    --------
    matplotlib.figure.Figure
        The figure object containing the visualization
    """
    if performance_df.empty:
        print("No performance data available for plotting.")
        return None
    
    # Create figure with multiple subplots
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 16))
    
    # Filter out the OVERALL row for the top plots
    performance_df_no_overall = performance_df[performance_df['persona'] != 'OVERALL'].copy()
    
    # Sort by precision for the first plot
    precision_df = performance_df_no_overall.sort_values('precision', ascending=False).head(10)
    
    # Plot precision by persona
    bars1 = ax1.barh(precision_df['persona'], precision_df['precision']*100, color='skyblue')
    ax1.set_title('Precision by Persona (Top 10)', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Precision (% of Correctly Flagged Downgrades)', fontsize=12)
    ax1.set_ylabel('Persona', fontsize=12)
    
    # Add data labels with percentage
    for bar in bars1:
        width = bar.get_width()
        label_x_pos = width if width > 0 else 0
        ax1.text(label_x_pos + 1, bar.get_y() + bar.get_height()/2, f'{width:.1f}%',
                 va='center', fontweight='bold')
    
    # Add color coding for risky personas
    for i, (_, row) in enumerate(precision_df.iterrows()):
        if row['is_risky_persona']:
            bars1[i].set_color('salmon')
    
    # Sort by total downgrades for the second plot
    volume_df = performance_df_no_overall.sort_values('total_downgrades', ascending=False).head(10)
    
    # Plot total downgrades by persona
    bars2 = ax2.barh(volume_df['persona'], volume_df['total_downgrades'], color='lightgreen')
    ax2.set_title('Total Downgrades by Persona (Top 10)', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Number of Downgrades', fontsize=12)
    ax2.set_ylabel('Persona', fontsize=12)
    
    # Add data labels
    for bar in bars2:
        width = bar.get_width()
        label_x_pos = width if width > 0 else 0
        ax2.text(label_x_pos + 0.5, bar.get_y() + bar.get_height()/2, f'{int(width)}',
                 va='center', fontweight='bold')
    
    # Add color coding for risky personas
    for i, (_, row) in enumerate(volume_df.iterrows()):
        if row['is_risky_persona']:
            bars2[i].set_color('salmon')
    
    # Sort by average days before downgrade for the third plot
    timing_df = performance_df_no_overall.sort_values('avg_days_before_downgrade').head(10)
    
    # Plot average days before downgrade by persona
    bars3 = ax3.barh(timing_df['persona'], timing_df['avg_days_before_downgrade'], color='lightsalmon')
    ax3.set_title('Avg. Days Before Downgrade by Persona', fontsize=14, fontweight='bold')
    ax3.set_xlabel('Average Days Before Downgrade', fontsize=12)
    ax3.set_ylabel('Persona', fontsize=12)
    
    # Add data labels
    for bar in bars3:
        width = bar.get_width()
        label_x_pos = width if width > 0 else 0
        ax3.text(label_x_pos + 1, bar.get_y() + bar.get_height()/2, f'{width:.1f} days',
                 va='center', fontweight='bold')
    
    # Add color coding for risky personas
    for i, (_, row) in enumerate(timing_df.iterrows()):
        if row['is_risky_persona']:
            bars3[i].set_color('darkred')
    
    # Add a legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='salmon', edgecolor='black', label='Risky Persona'),
        Patch(facecolor='skyblue', edgecolor='black', label='Non-Risky Persona')
    ]
    ax1.legend(handles=legend_elements, loc='lower right')
    
    # Add overall statistics in a text box
    if 'OVERALL' in performance_df['persona'].values:
        overall_stats = performance_df[performance_df['persona'] == 'OVERALL'].iloc[0]
        
        stats_text = (
            f"Overall Performance:\n"
            f"Total Downgrades: {int(overall_stats['total_downgrades'])}\n"
            f"Correctly Flagged: {int(overall_stats['correctly_flagged'])} ({overall_stats['precision']*100:.1f}%)\n"
            f"Avg. Lead Time: {overall_stats['avg_days_before_downgrade']:.1f} days\n"
            f"Min Lead Time: {overall_stats['min_days_before']:.1f} days\n"
            f"Max Lead Time: {overall_stats['max_days_before']:.1f} days"
        )
        
        # Add stats text box to the first plot
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        ax1.text(0.98, 0.02, stats_text, transform=ax1.transAxes, fontsize=10,
                 verticalalignment='bottom', horizontalalignment='right', bbox=props)
    
    # Add overall title
    plt.suptitle(title, fontsize=16, fontweight='bold')
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.95)
    
    return fig

def plot_timing_distribution(timing_df, title="Timing Analysis: Persona Assignment vs Credit Downgrade"):
    """
    Plot the distribution of days between persona assignment and credit downgrade.
    
    Parameters:
    -----------
    timing_df : pandas.DataFrame
        DataFrame containing timing analysis
        
    Returns:
    --------
    matplotlib.figure.Figure
        The figure object containing the visualization
    """
    if timing_df.empty:
        print("No timing data available for plotting.")
        return None
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 12))
    
    # 1. Distribution of days before downgrade (histogram)
    sns.histplot(timing_df['days_before'].dropna(), bins=30, kde=True, ax=ax1, color='skyblue')
    ax1.set_title('Distribution of Days Before Downgrade', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Days Before Downgrade', fontsize=12)
    ax1.set_ylabel('Frequency', fontsize=12)
    
    # Add vertical line at mean
    mean_days = timing_df['days_before'].mean()
    median_days = timing_df['days_before'].median()
    ax1.axvline(x=mean_days, color='red', linestyle='--', label=f'Mean: {mean_days:.1f} days')
    ax1.axvline(x=median_days, color='green', linestyle=':', label=f'Median: {median_days:.1f} days')
    ax1.legend()
    
    # 2. Scatter plot of downgrade severity vs days before
    sns.scatterplot(x='days_before', y='downgrade_severity', 
                   data=timing_df.dropna(subset=['days_before']), 
                   hue='correctly_flagged', size='downgrade_severity',
                   sizes=(20, 200), palette=['red', 'green'], ax=ax2)
    
    ax2.set_title('Downgrade Severity vs Days Before Downgrade', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Days Before Downgrade', fontsize=12)
    ax2.set_ylabel('Downgrade Severity', fontsize=12)
    
    # Change legend labels
    handles, labels = ax2.get_legend_handles_labels()
    for i, label in enumerate(labels):
        if label == 'True':
            labels[i] = 'Correctly Flagged'
        elif label == 'False':
            labels[i] = 'Missed'
    ax2.legend(handles, labels, title='Prediction', loc='upper right')
    
    # Add trend line
    valid_data = timing_df.dropna(subset=['days_before', 'downgrade_severity'])
    if len(valid_data) > 1:
        slope, intercept, r_value, p_value, std_err = stats.linregress(
            valid_data['days_before'], valid_data['downgrade_severity']
        )
        
        x = np.array([valid_data['days_before'].min(), valid_data['days_before'].max()])
        y = intercept + slope * x
        
        ax2.plot(x, y, 'k--', alpha=0.7)
        
        # Add correlation coefficient
        ax2.text(0.02, 0.98, f'Correlation: {r_value:.2f}', transform=ax2.transAxes, 
                fontsize=10, verticalalignment='top', horizontalalignment='left',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
    
    # Add overall title
    plt.suptitle(title, fontsize=16, fontweight='bold')
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.95)
    
    return fig

def plot_heatmap_persona_risk_timing(timing_df, title="Heatmap: Persona-Risk Relationship to Downgrade Timing"):
    """
    Create a heatmap showing the relationship between personas, risk levels, and downgrade timing.
    
    Parameters:
    -----------
    timing_df : pandas.DataFrame
        DataFrame containing timing analysis with personas, risk levels, and days before downgrade
        
    Returns:
    --------
    matplotlib.figure.Figure
        The figure object containing the visualization
    """
    if timing_df.empty:
        print("No timing data available for plotting.")
        return None
    
    # Create pivot table for heatmap
    # Rows: personas, Columns: risk levels, Values: average days before downgrade
    heatmap_data = timing_df.pivot_table(
        index='before_persona',
        columns='risk_level_before',
        values='days_before',
        aggfunc='mean'
    )
    
    # Sort rows by average value
    heatmap_data = heatmap_data.reindex(heatmap_data.mean(axis=1).sort_values().index)
    
    # Create figure
    plt.figure(figsize=(14, 10))
    
    # Create heatmap
    mask = heatmap_data.isna()
    ax = sns.heatmap(heatmap_data, annot=True, fmt='.1f', cmap='YlOrRd_r', 
                    mask=mask, cbar_kws={'label': 'Avg. Days Before Downgrade'})
    
    # Set titles and labels
    plt.title(title, fontsize=16, fontweight='bold')
    plt.xlabel('Risk Level', fontsize=12)
    plt.ylabel('Persona', fontsize=12)
    
    # Add count of samples in each cell as text
    count_data = timing_df.pivot_table(
        index='before_persona',
        columns='risk_level_before',
        values='days_before',
        aggfunc='count'
    ).reindex(heatmap_data.index)
    
    # Add count as second line in each cell
    for i, idx in enumerate(heatmap_data.index):
        for j, col in enumerate(heatmap_data.columns):
            if not np.isnan(heatmap_data.iloc[i, j]):
                count = count_data.iloc[i, j]
                ax.text(j + 0.5, i + 0.7, f'n={int(count)}', 
                       ha="center", va="center", color="black", fontsize=8)
    
    plt.tight_layout()
    
    return plt.gcf()

def plot_lead_time_by_severity(severity_stats, title="Lead Time by Downgrade Severity"):
    """
    Plot the relationship between downgrade severity and lead time.
    
    Parameters:
    -----------
    severity_stats : pandas.DataFrame
        DataFrame with lead time statistics by severity
        
    Returns:
    --------
    matplotlib.figure.Figure
        The figure object containing the visualization
    """
    if severity_stats.empty:
        print("No severity statistics available for plotting.")
        return None
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))
    
    # Plot average days before by severity
    ax1.scatter(severity_stats['downgrade_severity'], severity_stats['avg_days_before'],
               s=severity_stats['count']*5, alpha=0.7, color='blue', edgecolor='black')
    
    # Add error bars
    ax1.errorbar(severity_stats['downgrade_severity'], severity_stats['avg_days_before'],
                yerr=severity_stats['std_days_before'], fmt='none', ecolor='gray', alpha=0.5)
    
    # Add trend line
    if len(severity_stats) > 1:
        slope, intercept, r_value, p_value, std_err = stats.linregress(
            severity_stats['downgrade_severity'], severity_stats['avg_days_before']
        )
        
        x = np.array([severity_stats['downgrade_severity'].min(), severity_stats['downgrade_severity'].max()])
        y = intercept + slope * x
        
        ax1.plot(x, y, 'r--', alpha=0.7)
        
        # Add correlation text
        ax1.text(0.05, 0.95, f'Correlation: {r_value:.2f}', transform=ax1.transAxes,
                fontsize=10, verticalalignment='top', horizontalalignment='left',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
    
    # Add labels for each point
    for i, row in severity_stats.iterrows():
        ax1.annotate(f"n={int(row['count'])}", 
                    (row['downgrade_severity'], row['avg_days_before']),
                    textcoords="offset points", xytext=(0, 10),
                    ha='center', fontsize=8)
    
    ax1.set_title('Average Lead Time by Downgrade Severity', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Downgrade Severity', fontsize=12)
    ax1.set_ylabel('Average Days Before Downgrade', fontsize=12)
    ax1.grid(True, alpha=0.3)
    
    # Plot detection rate by severity
    bars = ax2.bar(severity_stats['downgrade_severity'], severity_stats['detection_rate']*100,
                  color='skyblue', alpha=0.7, edgecolor='black')
    
    # Add data labels
    for bar in bars:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 1, f'{height:.1f}%',
                ha='center', va='bottom', fontsize=8, fontweight='bold')
    
    ax2.set_title('Detection Rate by Downgrade Severity', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Downgrade Severity', fontsize=12)
    ax2.set_ylabel('Detection Rate (%)', fontsize=12)
    ax2.set_ylim(0, 105)  # Add space for labels
    ax2.grid(True, alpha=0.3)
    
    # Add overall title
    plt.suptitle(title, fontsize=16, fontweight='bold')
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    
    return fig

def plot_industry_performance(industry_performance, title="Persona Model Performance by Industry"):
    """
    Plot the performance of the persona model across different industries.
    
    Parameters:
    -----------
    industry_performance : pandas.DataFrame
        DataFrame with performance metrics by industry
        
    Returns:
    --------
    matplotlib.figure.Figure
        The figure object containing the visualization
    """
    if industry_performance.empty:
        print("No industry performance data available for plotting.")
        return None
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 12))
    
    # 1. Plot detection rate by industry (sorted)
    industry_performance_sorted = industry_performance.sort_values('detection_rate', ascending=True)
    
    # Filter to top 10 industries by total downgrades for readability
    top_industries = industry_performance_sorted.nlargest(10, 'total_downgrades')
    
    # Create horizontal bar chart of detection rates
    bars1 = ax1.barh(top_industries['industry'], top_industries['detection_rate']*100, color='skyblue')
    
    # Add data labels
    for bar in bars1:
        width = bar.get_width()
        ax1.text(width + 1, bar.get_y() + bar.get_height()/2, f'{width:.1f}%',
                va='center', fontsize=9, fontweight='bold')
    
    ax1.set_title('Detection Rate by Industry (Top 10 by Volume)', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Detection Rate (%)', fontsize=12)
    ax1.set_ylabel('Industry', fontsize=12)
    ax1.set_xlim(0, 105)  # Add space for labels
    ax1.grid(True, alpha=0.3)
    
    # Add sample count labels
    for i, (_, row) in enumerate(top_industries.iterrows()):
        ax1.text(5, i, f"n={int(row['total_downgrades'])}", 
                va='center', ha='left', fontsize=8, color='white', fontweight='bold')
    
    # 2. Plot lead time by industry
    bars2 = ax2.barh(top_industries['industry'], top_industries['avg_days_before'], color='salmon')
    
    # Add data labels
    for bar in bars2:
        width = bar.get_width()
        ax2.text(width + 1, bar.get_y() + bar.get_height()/2, f'{width:.1f} days',
                va='center', fontsize=9, fontweight='bold')
    
    ax2.set_title('Average Lead Time by Industry (Top 10 by Volume)', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Average Days Before Downgrade', fontsize=12)
    ax2.set_ylabel('Industry', fontsize=12)
    ax2.grid(True, alpha=0.3)
    
    # Add overall title
    plt.suptitle(title, fontsize=16, fontweight='bold')
    
    # Add overall statistics in a text box
    weighted_avg_detection = (
        industry_performance['correctly_flagged'].sum() / 
        industry_performance['total_downgrades'].sum()
    ) * 100
    
    weighted_avg_lead_time = (
        (industry_performance['avg_days_before'] * industry_performance['total_downgrades']).sum() / 
        industry_performance['total_downgrades'].sum()
    )
    
    stats_text = (
        f"Overall Statistics:\n"
        f"Total Industries: {len(industry_performance)}\n"
        f"Total Downgrades: {industry_performance['total_downgrades'].sum():.0f}\n"
        f"Weighted Avg. Detection Rate: {weighted_avg_detection:.1f}%\n"
        f"Weighted Avg. Lead Time: {weighted_avg_lead_time:.1f} days"
    )
    
    # Add stats text box to the first plot
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax1.text(0.98, 0.05, stats_text, transform=ax1.transAxes, fontsize=10,
             verticalalignment='bottom', horizontalalignment='right', bbox=props)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    
    return fig

def plot_confidence_impact(timing_df, title="Impact of Persona Confidence on Downgrade Prediction"):
    """
    Analyze and visualize how persona confidence levels impact prediction accuracy.
    
    Parameters:
    -----------
    timing_df : pandas.DataFrame
        DataFrame containing timing analysis with confidence levels
        
    Returns:
    --------
    matplotlib.figure.Figure
        The figure object containing the visualization
    """
    if timing_df.empty or 'before_confidence' not in timing_df.columns:
        print("No confidence data available for analysis.")
        return None
    
    # Create confidence bins
    timing_df = timing_df.copy()
    timing_df['confidence_bin'] = pd.cut(
        timing_df['before_confidence'],
        bins=[0, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        labels=['0.0-0.5', '0.5-0.6', '0.6-0.7', '0.7-0.8', '0.8-0.9', '0.9-1.0']
    )
    
    # Calculate metrics by confidence bin
    confidence_impact = timing_df.groupby('confidence_bin').agg({
        'correctly_flagged': ['count', 'sum', 'mean'],
        'days_before': 'mean',
        'downgrade_severity': 'mean'
    }).reset_index()
    
    # Rename columns for clarity
    confidence_impact.columns = [
        'confidence_bin', 'total_downgrades', 'correctly_flagged',
        'detection_rate', 'avg_days_before', 'avg_severity'
    ]
    
    # Create figure with multiple subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # 1. Plot detection rate by confidence bin
    bars1 = ax1.bar(confidence_impact['confidence_bin'], confidence_impact['detection_rate']*100, 
                  color='skyblue', alpha=0.8, edgecolor='black')
    
    # Add data labels
    for bar in bars1:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 1, f'{height:.1f}%',
                ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    # Add sample sizes above bars
    for i, row in confidence_impact.iterrows():
        ax1.text(i, 5, f"n={row['total_downgrades']}", ha='center', fontsize=8)
    
    ax1.set_title('Detection Rate by Confidence Level', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Confidence Level', fontsize=12)
    ax1.set_ylabel('Detection Rate (%)', fontsize=12)
    ax1.set_ylim(0, 105)  # Add space for labels
    ax1.grid(True, alpha=0.3)
    
    # 2. Plot lead time by confidence bin
    line = ax2.plot(confidence_impact['confidence_bin'], confidence_impact['avg_days_before'], 
                   marker='o', markersize=8, linewidth=2, color='salmon')
    
    # Add data labels
    for i, row in confidence_impact.iterrows():
        ax2.text(i, row['avg_days_before'] + 2, f'{row["avg_days_before"]:.1f} days',
                ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    # Add secondary axis for severity
    ax2_twin = ax2.twinx()
    line2 = ax2_twin.plot(confidence_impact['confidence_bin'], confidence_impact['avg_severity'],
                         marker='s', markersize=8, linewidth=2, color='green', linestyle='--')
    
    # Add data labels for severity
    for i, row in confidence_impact.iterrows():
        ax2_twin.text(i, row['avg_severity'] + 0.2, f'{row["avg_severity"]:.1f}',
                     ha='center', va='bottom', fontsize=9, fontweight='bold', color='green')
    
    # Add legend
    ax2.legend([line[0]], ['Lead Time'], loc='upper left')
    ax2_twin.legend([line2[0]], ['Avg. Severity'], loc='upper right')
    
    ax2.set_title('Lead Time and Severity by Confidence Level', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Confidence Level', fontsize=12)
    ax2.set_ylabel('Average Days Before Downgrade', fontsize=12)
    ax2_twin.set_ylabel('Average Downgrade Severity', fontsize=12, color='green')
    ax2.grid(True, alpha=0.3)
    
    # Add overall title
    plt.suptitle(title, fontsize=16, fontweight='bold')
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    
    return fig

def generate_synthetic_credit_score_data(company_ids, start_date=None, end_date=None, random_seed=42):
    """
    Generate synthetic credit score data for backtesting.
    
    Parameters:
    -----------
    company_ids : list
        List of company IDs to generate data for
    start_date : datetime
        Start date for the data range (default: 2 years before today)
    end_date : datetime
        End date for the data range (default: today)
    random_seed : int
        Random seed for reproducibility
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame containing synthetic credit score data
    """
    # Set random seed
    np.random.seed(random_seed)
    
    # Set date range
    if end_date is None:
        end_date = datetime.now()
    if start_date is None:
        start_date = end_date - timedelta(days=365*2)
    
    # Define credit scores
    credit_scores = ['1+', '1', '2+', '2', '2-', '3+', '3', '3-', '4+', '4', 
                     '4-', '5+', '5', '5-', '6+', '6', '6-', '7', '8', '9', '10', 'NC', 'NR']
    
    # Define industries
    industries = ['Manufacturing', 'Retail', 'Technology', 'Healthcare', 'Financial Services',
                 'Energy', 'Telecommunications', 'Construction', 'Transportation', 'Agriculture',
                 'Real Estate', 'Education', 'Hospitality', 'Media', 'Consulting']
    
    # Generate credit score data
    credit_score_data = []
    
    for company_id in company_ids:
        # Assign industry
        industry = np.random.choice(industries)
        
        # Determine if this company will have a downgrade
        has_downgrade = np.random.random() < 0.6  # 60% of companies have downgrades
        
        # Choose initial credit score (weighted toward better scores)
        initial_score_idx = min(
            int(np.random.exponential(scale=7)),
            len(credit_scores) - 3
        )
        initial_score = credit_scores[initial_score_idx]
        
        # Add initial score
        initial_date = start_date + timedelta(days=np.random.randint(0, 30))
        credit_score_data.append({
            'company_id': company_id,
            'date': initial_date,
            'credit_score': initial_score,
            'industry': industry
        })
        
        if has_downgrade:
            # Determine number of downgrades (1-3)
            num_downgrades = np.random.randint(1, 4)
            
            current_score_idx = initial_score_idx
            current_date = initial_date
            
            for _ in range(num_downgrades):
                # Determine time to next downgrade (30-300 days)
                days_to_downgrade = np.random.randint(30, 300)
                next_date = current_date + timedelta(days=days_to_downgrade)
                
                # Skip if beyond end date
                if next_date > end_date:
                    break
                
                # Determine downgrade severity (1-3 steps)
                severity = np.random.randint(1, 4)
                next_score_idx = min(current_score_idx + severity, len(credit_scores) - 1)
                next_score = credit_scores[next_score_idx]
                
                # Add downgrade
                credit_score_data.append({
                    'company_id': company_id,
                    'date': next_date,
                    'credit_score': next_score,
                    'industry': industry
                })
                
                current_score_idx = next_score_idx
                current_date = next_date
        
        # Add some random intermediate scores for companies without downgrades
        else:
            num_updates = np.random.randint(1, 5)
            
            for _ in range(num_updates):
                update_date = start_date + timedelta(days=np.random.randint(30, 730))
                
                # Skip if beyond end date
                if update_date > end_date:
                    continue
                
                # Small fluctuations or same score
                fluctuation = np.random.randint(-1, 2)
                update_score_idx = max(0, min(initial_score_idx + fluctuation, len(credit_scores) - 1))
                update_score = credit_scores[update_score_idx]
                
                credit_score_data.append({
                    'company_id': company_id,
                    'date': update_date,
                    'credit_score': update_score,
                    'industry': industry
                })
    
    # Create credit score dataframe
    return pd.DataFrame(credit_score_data)

def integrated_risk_backtest_workflow(bank_data_path=None, credit_score_path=None, generate_synthetic=True, num_companies=50):
    """
    Execute the end-to-end integrated workflow that combines bank client risk analysis
    with credit score downgrade backtest validation.
    
    Parameters:
    -----------
    bank_data_path : str
        Path to bank client data file (default: None, generates synthetic data)
    credit_score_path : str
        Path to credit score data file (default: None, generates synthetic data)
    generate_synthetic : bool
        Whether to generate synthetic data (default: True)
    num_companies : int
        Number of companies to generate synthetic data for (default: 50)
        
    Returns:
    --------
    dict
        Dictionary containing all analysis results
    """
    results = {}
    
    print("======= BANK CLIENT RISK ANALYSIS SYSTEM =======")
    print("Starting integrated risk analysis with credit score backtest...")
    
    # 1. Load or generate bank client data
    if bank_data_path and not generate_synthetic:
        print("\nLoading bank client data from file...")
        try:
            bank_df = pd.read_csv(bank_data_path)
            # Convert date column
            bank_df['date'] = pd.to_datetime(bank_df['date'])
            print(f"Loaded {len(bank_df)} records for {bank_df['company_id'].nunique()} companies")
        except Exception as e:
            print(f"Error loading bank data: {e}")
            print("Falling back to synthetic data generation")
            generate_synthetic = True
    
    if generate_synthetic or not bank_data_path:
        print("\nGenerating synthetic bank client data...")
        bank_df = generate_bank_client_data(num_companies=num_companies, days=730)  # 2 years of data
        print(f"Generated {len(bank_df)} records for {bank_df['company_id'].nunique()} companies")
    
    # 2. Run bank client risk analysis
    print("\n=== PHASE 1: BANK CLIENT RISK ANALYSIS ===")
    bank_analysis_results = bank_client_risk_analysis(bank_df)
    
    # Store results
    results.update(bank_analysis_results)
    company_ids = bank_analysis_results['persona_df']['company_id'].unique()
    
    # 3. Load or generate credit score data
    if credit_score_path and not generate_synthetic:
        print("\nLoading credit score data from file...")
        try:
            credit_score_df = pd.read_csv(credit_score_path)
            # Convert date column
            credit_score_df['date'] = pd.to_datetime(credit_score_df['date'])
            print(f"Loaded {len(credit_score_df)} credit score records")
        except Exception as e:
            print(f"Error loading credit score data: {e}")
            print("Falling back to synthetic data generation")
            generate_synthetic = True
    
    if generate_synthetic or not credit_score_path:
        print("\nGenerating synthetic credit score data...")
        # Use the same company IDs as in the bank risk analysis
        credit_score_df = generate_synthetic_credit_score_data(
            company_ids=bank_analysis_results['persona_df']['company_id'].unique(),
            start_date=bank_df['date'].min(),
            end_date=bank_df['date'].max(),
        )
        print(f"Generated {len(credit_score_df)} credit score records for {credit_score_df['company_id'].nunique()} companies")
    
    # 4. Credit score backtest analysis
    print("\n=== PHASE 2: CREDIT SCORE BACKTEST ===")
    
    # 4.1 Detect credit downgrades
    print("\nDetecting credit downgrades over the past year...")
    downgrade_df = detect_credit_downgrades(credit_score_df, lookback_years=1)
    print(f"Detected {len(downgrade_df)} downgrade events across {downgrade_df['company_id'].nunique()} companies")
    
    # 4.2 Analyze distribution by industry
    print("\nAnalyzing downgrade distribution by industry over the past 2 years...")
    industry_stats = analyze_downgrade_distribution_by_industry(downgrade_df, lookback_years=2)
    print(f"Analyzed downgrades across {len(industry_stats)} industries")
    
    # Save distribution plot
    industry_plot = plot_downgrade_distribution(industry_stats)
    if industry_plot:
        industry_plot.savefig('downgrade_distribution_by_industry.png')
        print("Saved industry distribution plot to 'downgrade_distribution_by_industry.png'")
    
    # 4.3 Backtest personas vs downgrades
    print("\nBacktesting personas against credit downgrades for the past 6 months...")
    performance_df, timing_df = backtest_personas_vs_downgrades(
        bank_analysis_results['persona_df'], 
        bank_analysis_results['risk_df'], 
        downgrade_df, 
        lookback_months=6
    )
    
    # Store backtest results
    results['downgrade_df'] = downgrade_df
    results['industry_stats'] = industry_stats
    results['backtest_performance'] = performance_df
    results['backtest_timing'] = timing_df
    
    if not performance_df.empty:
        print(f"Analyzed performance for {len(performance_df)-1} personas")  # -1 for OVERALL row
        
        # Save performance plot
        performance_plot = plot_persona_backtest_performance(performance_df)
        if performance_plot:
            performance_plot.savefig('persona_backtest_performance.png')
            print("Saved persona performance plot to 'persona_backtest_performance.png'")
        
        # Save timing distribution plot
        timing_plot = plot_timing_distribution(timing_df)
        if timing_plot:
            timing_plot.savefig('persona_timing_distribution.png')
            print("Saved timing distribution plot to 'persona_timing_distribution.png'")
        
        # 4.4 Analyze lead time by severity
        print("\nAnalyzing lead time by downgrade severity...")
        severity_stats = analyze_lead_time_by_severity(timing_df)
        results['severity_stats'] = severity_stats
        
        if not severity_stats.empty:
            severity_plot = plot_lead_time_by_severity(severity_stats)
            if severity_plot:
                severity_plot.savefig('lead_time_by_severity.png')
                print("Saved lead time by severity plot to 'lead_time_by_severity.png'")
        
        # 4.5 Analyze industry performance
        print("\nAnalyzing model performance by industry...")
        industry_performance = analyze_industry_performance(timing_df)
        results['industry_performance'] = industry_performance
        
        if not industry_performance.empty:
            industry_perf_plot = plot_industry_performance(industry_performance)
            if industry_perf_plot:
                industry_perf_plot.savefig('industry_performance.png')
                print("Saved industry performance plot to 'industry_performance.png'")
        
        # 4.6 Create heatmap of persona-risk relationship
        print("\nCreating heatmap of persona-risk relationship to downgrade timing...")
        heatmap_plot = plot_heatmap_persona_risk_timing(timing_df)
        if heatmap_plot:
            heatmap_plot.savefig('persona_risk_heatmap.png')
            print("Saved persona-risk heatmap to 'persona_risk_heatmap.png'")
        
        # 4.7 Analyze impact of confidence
        print("\nAnalyzing impact of persona confidence on prediction accuracy...")
        confidence_plot = plot_confidence_impact(timing_df)
        if confidence_plot:
            confidence_plot.savefig('confidence_impact.png')
            print("Saved confidence impact plot to 'confidence_impact.png'")
        
        # Print overall results
        if 'OVERALL' in performance_df['persona'].values:
            overall = performance_df[performance_df['persona'] == 'OVERALL'].iloc[0]
            
            print("\n=== OVERALL BACKTEST RESULTS ===")
            print(f"Total downgrades analyzed: {overall['total_downgrades']:.0f}")
            print(f"Correctly flagged: {overall['correctly_flagged']:.0f} ({overall['precision']*100:.1f}%)")
            print(f"Average lead time: {overall['avg_days_before_downgrade']:.1f} days")
            print(f"High-confidence precision: {overall['high_conf_precision']*100:.1f}%")
            
            # Top performing personas
            print("\nTop performing personas:")
            top_personas = performance_df[performance_df['persona'] != 'OVERALL'].sort_values('precision', ascending=False).head(3)
            for _, row in top_personas.iterrows():
                print(f"- {row['persona']}: {row['precision']*100:.1f}% precision, {row['avg_days_before_downgrade']:.1f} days lead time")
    else:
        print("No matching data found for backtest analysis.")
    
    print("\nIntegrated risk analysis and backtest complete!")
    
    return results

if __name__ == "__main__":
    # Execute the integrated workflow
    integrated_risk_backtest_workflow(generate_synthetic=True, num_companies=100)
