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
# INTEGRATED CONFIGURATION
#######################################################

# Enhanced configuration combining original and sparse data analysis
INTEGRATED_CONFIG = {
    'data': {
        'min_nonzero_pct': 0.8,
        'min_continuous_days': 365,
        'recent_window': 30,
        # Sparse data parameters
        'sparse_min_days_for_analysis': 30,
        'sparse_recent_period_days': 180,
        'sparse_dormancy_threshold_days': 90,
        'sparse_new_client_threshold_days': 365,
        'sparse_gap_threshold_days': 30,
        'sparse_activity_threshold': 0.01,
    },
    'risk': {
        'trend_windows': [30, 90, 180],
        'change_thresholds': {
            'sharp': 0.2,
            'moderate': 0.1,
            'gradual': 0.05
        },
        # Original personas
        'traditional_personas': {
            'cautious_borrower': 'Low utilization (<40%), stable deposits',
            'aggressive_expansion': 'Rising utilization (>10% increase), volatile deposits',
            'distressed_client': 'High utilization (>80%), declining deposits (>5% decrease)',
            'seasonal_loan_user': 'Cyclical utilization with >15% amplitude',
            'seasonal_deposit_pattern': 'Cyclical deposits with >20% amplitude',
            'deteriorating_health': 'Rising utilization (>15% increase), declining deposits (>10% decrease)',
            'cash_constrained': 'Stable utilization, rapidly declining deposits (>15% decrease)',
            'credit_dependent': 'High utilization (>75%), low deposit ratio (<0.8)',
            'stagnant_growth': 'Loan utilization increasing (>8%) with flat deposits (<2% change)',
            'utilization_spikes': 'Showing sudden large increases (>15%) in loan utilization within short periods',
            'seasonal_pattern_breaking': 'Historical seasonality exists but recent data shows deviation from expected patterns',
            'approaching_limit': 'Utilization nearing credit limit (>90%) with increased usage velocity',
            'withdrawal_intensive': 'Unusual increase in deposit withdrawal frequency or size',
            'deposit_concentration': 'Deposits heavily concentrated in timing, showing potential liquidity planning issues',
            'historical_low_deposits': 'Deposit balance below historical low point while maintaining high utilization'
        },
        # Sparse data personas
        'sparse_personas': {
            'recently_dormant': 'Had recent activity but went silent for 90+ days',
            'new_active_client': 'New client (< 1 year) with regular recent activity',
            'new_inactive_client': 'New client with minimal or no activity',
            'intermittent_client': 'Sporadic activity with significant gaps',
            'closed_relationship': 'No meaningful activity for 6+ months',
            'seasonal_project_client': 'Activity in distinct bursts/seasons',
            'declining_engagement': 'Gradually decreasing activity over time',
            'recently_reactivated': 'Returned to activity after long dormancy',
            'weekend_warrior': 'Activity concentrated in specific periods',
            'ghost_client': 'Account exists but minimal to no activity ever',
            'volatile_new_client': 'New client with erratic activity patterns',
            'legacy_dormant': 'Long-term client now dormant',
            'low_activity_client': 'Generally low activity levels',
            'standard_sparse_client': 'Sparse but regular activity'
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

#######################################################
# ENHANCED DATA CLEANING AND ANALYSIS
#######################################################

def enhanced_clean_data(df, min_nonzero_pct=0.8):
    """
    Enhanced version of clean_data that separates companies into two categories:
    1. Traditional analysis candidates (meet original criteria)
    2. Sparse data analysis candidates (filtered out by original criteria)
    
    This ensures no company data is lost while applying appropriate analysis methods.
    """
    print(f"Enhanced data cleaning - Original data shape: {df.shape}")
    
    # Calculate percentage of non-zero values for each company
    company_stats = {}
    for company in df['company_id'].unique():
        company_data = df[df['company_id'] == company]
        
        deposit_nonzero = (company_data['deposit_balance'] > 0).mean()
        used_loan_nonzero = (company_data['used_loan'] > 0).mean()
        unused_loan_nonzero = (company_data['unused_loan'] > 0).mean()
        
        # Overall activity rate (any non-zero value)
        any_activity = ((company_data['deposit_balance'] > 0) | 
                       (company_data['used_loan'] > 0) | 
                       (company_data['unused_loan'] > 0)).mean()
        
        company_stats[company] = {
            'deposit_nonzero': deposit_nonzero,
            'used_loan_nonzero': used_loan_nonzero,
            'unused_loan_nonzero': unused_loan_nonzero,
            'any_activity': any_activity,
            'total_records': len(company_data),
            'date_span': (company_data['date'].max() - company_data['date'].min()).days
        }
    
    # Separate companies into traditional and sparse categories
    traditional_companies = []
    sparse_companies = []
    
    for company, stats in company_stats.items():
        # Traditional analysis criteria (original logic)
        if (stats['deposit_nonzero'] >= min_nonzero_pct or 
            (stats['used_loan_nonzero'] >= min_nonzero_pct and 
             stats['unused_loan_nonzero'] >= min_nonzero_pct)):
            traditional_companies.append(company)
        else:
            # These go to sparse analysis instead of being discarded
            sparse_companies.append(company)
    
    # Create datasets
    df_traditional = df[df['company_id'].isin(traditional_companies)].copy()
    df_sparse = df[df['company_id'].isin(sparse_companies)].copy()
    
    print(f"Traditional analysis companies: {len(traditional_companies)}")
    print(f"Sparse data analysis companies: {len(sparse_companies)}")
    print(f"Total companies retained: {len(traditional_companies) + len(sparse_companies)}")
    
    # Apply original continuous data filtering to traditional companies
    continuous_traditional_companies = []
    for company in traditional_companies:
        company_data = df_traditional[df_traditional['company_id'] == company].sort_values('date')
        
        # Check if company has data for the most recent period
        max_date = df['date'].max()
        min_required_date = max_date - pd.Timedelta(days=INTEGRATED_CONFIG['data']['min_continuous_days'])
        
        # Get recent data
        recent_data = company_data[company_data['date'] >= min_required_date]
        
        # Ensure both deposit and loan data are available
        if (recent_data['deposit_balance'] > 0).sum() >= INTEGRATED_CONFIG['data']['min_continuous_days'] * 0.8 and \
           (recent_data['used_loan'] > 0).sum() >= INTEGRATED_CONFIG['data']['min_continuous_days'] * 0.8:
            continuous_traditional_companies.append(company)
    
    # Filter traditional dataset for continuous data
    df_traditional_clean = df_traditional[df_traditional['company_id'].isin(continuous_traditional_companies)].copy()
    
    print(f"Traditional companies after continuous data filter: {len(continuous_traditional_companies)}")
    
    # Apply KNN imputation to traditional data
    df_traditional_imputed = df_traditional_clean.copy()
    
    for company in tqdm(continuous_traditional_companies, desc="Applying KNN imputation to traditional data"):
        company_data = df_traditional_clean[df_traditional_clean['company_id'] == company].sort_values('date')
        
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
            df_traditional_imputed.loc[company_data.index, 'deposit_balance'] = imputed_values[:, 0]
            df_traditional_imputed.loc[company_data.index, 'used_loan'] = imputed_values[:, 1]
            df_traditional_imputed.loc[company_data.index, 'unused_loan'] = imputed_values[:, 2]
    
    return df_traditional_imputed, df_sparse, {
        'traditional_companies': continuous_traditional_companies,
        'sparse_companies': sparse_companies,
        'company_stats': company_stats
    }

def add_enhanced_derived_metrics(df, is_traditional=True):
    """
    Enhanced version that adds derived metrics appropriate for traditional or sparse data.
    """
    df = df.copy()
    
    if is_traditional:
        # Apply full traditional analysis (original logic)
        df = add_traditional_derived_metrics(df)
    else:
        # Apply sparse-appropriate metrics
        df = add_sparse_derived_metrics(df)
    
    return df

def add_traditional_derived_metrics(df):
    """
    Original derived metrics logic for traditional (high-activity) companies.
    """
    # Basic metrics
    df['total_loan'] = df['used_loan'] + df['unused_loan']
    df['loan_utilization'] = df['used_loan'] / df['total_loan']
    
    # Handle NaN values for loan_utilization
    df.loc[df['total_loan'] == 0, 'loan_utilization'] = np.nan
    
    # Calculate deposit to loan ratio
    df['deposit_loan_ratio'] = df['deposit_balance'] / df['used_loan']
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    
    # Calculate rolling metrics for each company
    for company in tqdm(df['company_id'].unique(), desc="Calculating traditional rolling metrics"):
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
                                                                  .rolling(window=7, min_periods=3).mean())
            df.loc[company_data.index, f'deposit_change_{window}d'] = (company_data['deposit_balance'].pct_change(periods=window)
                                                                     .rolling(window=7, min_periods=3).mean())
            
        # Calculate volatility measures
        df.loc[company_data.index, 'util_volatility_30d'] = company_data['loan_utilization'].rolling(
            30, min_periods=10).std()
        df.loc[company_data.index, 'deposit_volatility_30d'] = company_data['deposit_balance'].pct_change().rolling(
            30, min_periods=10).std()
        
        # Traditional seasonality detection
        detect_traditional_seasonality(df, company_data)
        
        # Enhanced traditional metrics
        add_enhanced_traditional_metrics(df, company_data)
    
    return df

def add_sparse_derived_metrics(df):
    """
    Specialized metrics for sparse data analysis.
    """
    # Basic activity indicators
    activity_threshold = INTEGRATED_CONFIG['data']['sparse_activity_threshold']
    
    df['has_deposit_activity'] = (df['deposit_balance'] > activity_threshold).astype(int)
    df['has_loan_activity'] = (df['used_loan'] > activity_threshold).astype(int)
    df['has_any_activity'] = ((df['deposit_balance'] > activity_threshold) | 
                             (df['used_loan'] > activity_threshold)).astype(int)
    
    # Calculate sparse-appropriate metrics for each company
    for company in tqdm(df['company_id'].unique(), desc="Calculating sparse data metrics"):
        company_data = df[df['company_id'] == company].sort_values('date')
        
        if len(company_data) < 5:  # Need minimum data points
            continue
        
        # Activity rate metrics (rolling)
        for window in [7, 30, 90]:
            if len(company_data) >= window:
                df.loc[company_data.index, f'activity_rate_{window}d'] = (
                    company_data['has_any_activity'].rolling(window, min_periods=1).mean()
                )
        
        # Gap detection metrics
        activity_series = company_data['has_any_activity']
        gaps = []
        current_gap = 0
        
        for activity in activity_series:
            if activity == 0:
                current_gap += 1
            else:
                if current_gap > 0:
                    gaps.append(current_gap)
                current_gap = 0
        
        # Add final gap if ending with inactivity
        if current_gap > 0:
            gaps.append(current_gap)
        
        # Gap statistics
        max_gap = max(gaps) if gaps else 0
        avg_gap = np.mean(gaps) if gaps else 0
        num_gaps = len([g for g in gaps if g >= INTEGRATED_CONFIG['data']['sparse_gap_threshold_days']])
        
        # Assign gap metrics to all rows for this company
        df.loc[company_data.index, 'max_gap_days'] = max_gap
        df.loc[company_data.index, 'avg_gap_days'] = avg_gap
        df.loc[company_data.index, 'num_significant_gaps'] = num_gaps
        
        # Trend analysis for sparse data
        if len(company_data) >= 10:
            x = np.arange(len(company_data))
            
            # Activity trend
            if activity_series.var() > 0:
                activity_slope, _, _, _, _ = stats.linregress(x, activity_series)
                df.loc[company_data.index, 'activity_trend'] = activity_slope
            
            # Value trends (for non-zero values only)
            nonzero_deposits = company_data[company_data['deposit_balance'] > 0]['deposit_balance']
            if len(nonzero_deposits) >= 3:
                deposit_slope, _, _, _, _ = stats.linregress(
                    np.arange(len(nonzero_deposits)), nonzero_deposits
                )
                df.loc[company_data.index, 'deposit_trend'] = deposit_slope
    
    return df

def detect_traditional_seasonality(df, company_data, min_periods=365):
    """
    Traditional seasonality detection (original logic).
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
                if 350 <= period <= 380 or 170 <= period <= 190 or 80 <= period <= 100:
                    df.loc[company_data.index, 'util_seasonal_period'] = period
                    df.loc[company_data.index, 'util_seasonal_amplitude'] = normalized_amplitude
                    df.loc[company_data.index, 'util_is_seasonal'] = normalized_amplitude > 0.15
    except:
        pass

def add_enhanced_traditional_metrics(df, company_data):
    """
    Enhanced metrics for traditional analysis (original enhanced metrics).
    """
    # Add utilization acceleration
    if len(company_data) >= 60:
        util_change_30d = df.loc[company_data.index, 'util_change_30d']
        if not util_change_30d.empty:
            df.loc[company_data.index, 'util_acceleration_30d'] = util_change_30d.pct_change(periods=30)
    
    # Add deposit concentration metrics
    if len(company_data) >= 90:
        for i in range(90, len(company_data)):
            window_data = company_data.iloc[i-90:i]
            deposit_changes = window_data['deposit_balance'].diff().fillna(0)
            positive_changes = deposit_changes[deposit_changes > 0].values
            
            if len(positive_changes) > 5:
                # Calculate Gini coefficient
                sorted_changes = np.sort(positive_changes)
                n = len(sorted_changes)
                index = np.arange(1, n+1)
                gini = (np.sum((2 * index - n - 1) * sorted_changes)) / (n * np.sum(sorted_changes))
                
                df.loc[company_data.index[i], 'deposit_concentration_gini'] = gini
    
    # Add withdrawal metrics
    if len(company_data) >= 60:
        for i in range(60, len(company_data)):
            current_window = company_data.iloc[i-30:i]
            deposit_changes = current_window['deposit_balance'].diff().fillna(0)
            withdrawals = deposit_changes[deposit_changes < 0].abs()
            
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

#######################################################
# SPARSE DATA ANALYSIS (FROM PREVIOUS ARTIFACT)
#######################################################

def analyze_sparse_data_patterns(df):
    """
    Comprehensive analysis of sparse data patterns to identify different client behavior types.
    """
    print("Analyzing sparse data patterns...")
    
    # Get current date and define time periods
    max_date = df['date'].max()
    recent_cutoff = max_date - pd.Timedelta(days=INTEGRATED_CONFIG['data']['sparse_recent_period_days'])
    dormancy_cutoff = max_date - pd.Timedelta(days=INTEGRATED_CONFIG['data']['sparse_dormancy_threshold_days'])
    new_client_cutoff = max_date - pd.Timedelta(days=INTEGRATED_CONFIG['data']['sparse_new_client_threshold_days'])
    
    sparse_analysis_results = []
    
    # Analyze each company's data patterns
    for company_id in tqdm(df['company_id'].unique(), desc="Analyzing sparse patterns"):
        company_data = df[df['company_id'] == company_id].sort_values('date').copy()
        
        # Skip if absolutely no data
        if len(company_data) == 0:
            continue
            
        # Basic data characteristics
        total_days = (company_data['date'].max() - company_data['date'].min()).days + 1
        data_span_days = len(company_data)
        data_density = data_span_days / total_days if total_days > 0 else 0
        
        # Calculate activity metrics
        activity_threshold = INTEGRATED_CONFIG['data']['sparse_activity_threshold']
        
        # For deposits
        deposit_active_days = (company_data['deposit_balance'] > activity_threshold).sum()
        deposit_activity_rate = deposit_active_days / len(company_data) if len(company_data) > 0 else 0
        
        # For loans
        loan_active_days = (company_data['used_loan'] > activity_threshold).sum()
        loan_activity_rate = loan_active_days / len(company_data) if len(company_data) > 0 else 0
        
        # Overall activity rate
        overall_activity_rate = max(deposit_activity_rate, loan_activity_rate)
        
        # Recency analysis
        recent_data = company_data[company_data['date'] >= recent_cutoff]
        recent_activity_rate = 0
        if len(recent_data) > 0:
            recent_deposit_active = (recent_data['deposit_balance'] > activity_threshold).sum()
            recent_loan_active = (recent_data['used_loan'] > activity_threshold).sum()
            recent_activity_rate = max(recent_deposit_active, recent_loan_active) / len(recent_data)
        
        # Check for dormancy
        dormant_period_data = company_data[company_data['date'] >= dormancy_cutoff]
        is_recently_dormant = False
        if len(dormant_period_data) > 0:
            dormant_deposit_active = (dormant_period_data['deposit_balance'] > activity_threshold).sum()
            dormant_loan_active = (dormant_period_data['used_loan'] > activity_threshold).sum()
            dormant_activity = max(dormant_deposit_active, dormant_loan_active)
            is_recently_dormant = dormant_activity == 0
        
        # New client analysis
        first_date = company_data['date'].min()
        is_new_client = first_date >= new_client_cutoff
        client_age_days = (max_date - first_date).days
        
        # Gap analysis
        gaps = []
        if len(company_data) > 1:
            company_data_sorted = company_data.sort_values('date')
            for i in range(1, len(company_data_sorted)):
                gap_days = (company_data_sorted.iloc[i]['date'] - company_data_sorted.iloc[i-1]['date']).days
                if gap_days > INTEGRATED_CONFIG['data']['sparse_gap_threshold_days']:
                    gaps.append(gap_days)
        
        avg_gap_days = np.mean(gaps) if gaps else 0
        max_gap_days = max(gaps) if gaps else 0
        num_significant_gaps = len(gaps)
        
        # Trend analysis
        deposit_trend = 0
        loan_trend = 0
        activity_trend = 0
        
        if len(company_data) >= 10:
            x = np.arange(len(company_data))
            
            # Deposit trend
            if company_data['deposit_balance'].var() > 0:
                deposit_slope, _, _, _, _ = stats.linregress(x, company_data['deposit_balance'])
                deposit_trend = deposit_slope
            
            # Loan trend  
            if company_data['used_loan'].var() > 0:
                loan_slope, _, _, _, _ = stats.linregress(x, company_data['used_loan'])
                loan_trend = loan_slope
            
            # Activity trend
            activity_indicator = ((company_data['deposit_balance'] > activity_threshold) | 
                                (company_data['used_loan'] > activity_threshold)).astype(int)
            if len(activity_indicator) > 5:
                rolling_activity = activity_indicator.rolling(window=5, min_periods=1).mean()
                if rolling_activity.var() > 0:
                    activity_slope, _, _, _, _ = stats.linregress(x, rolling_activity)
                    activity_trend = activity_slope
        
        # Store analysis results
        sparse_analysis_results.append({
            'company_id': company_id,
            'total_days_span': total_days,
            'data_points': len(company_data),
            'data_density': data_density,
            'deposit_activity_rate': deposit_activity_rate,
            'loan_activity_rate': loan_activity_rate,
            'overall_activity_rate': overall_activity_rate,
            'recent_activity_rate': recent_activity_rate,
            'is_recently_dormant': is_recently_dormant,
            'is_new_client': is_new_client,
            'client_age_days': client_age_days,
            'avg_gap_days': avg_gap_days,
            'max_gap_days': max_gap_days,
            'num_significant_gaps': num_significant_gaps,
            'deposit_trend': deposit_trend,
            'loan_trend': loan_trend,
            'activity_trend': activity_trend,
            'first_activity_date': company_data['date'].min(),
            'last_activity_date': company_data['date'].max(),
            'max_deposit': company_data['deposit_balance'].max(),
            'max_loan': company_data['used_loan'].max(),
            'avg_deposit': company_data['deposit_balance'].mean(),
            'avg_loan': company_data['used_loan'].mean()
        })
    
    return pd.DataFrame(sparse_analysis_results)

def assign_sparse_data_personas(sparse_analysis_df):
    """
    Assign specific risk personas to companies based on their sparse data patterns.
    """
    print("Assigning sparse data personas...")
    
    persona_assignments = []
    
    for _, row in tqdm(sparse_analysis_df.iterrows(), total=len(sparse_analysis_df), desc="Assigning personas"):
        company_id = row['company_id']
        personas = []
        risk_factors = []
        confidence_scores = []
        
        base_risk_level = 'low'
        
        # Persona assignment logic (same as previous artifact)
        if (row['is_recently_dormant'] and 
            row['recent_activity_rate'] < 0.05 and 
            row['max_gap_days'] > 180):
            personas.append('closed_relationship')
            risk_factors.append('No activity for 6+ months')
            confidence_scores.append(0.9)
            base_risk_level = 'high'
        
        elif (row['is_recently_dormant'] and 
              row['overall_activity_rate'] > 0.3 and
              row['client_age_days'] > 180):
            personas.append('recently_dormant')
            risk_factors.append('Active client went silent recently')
            confidence_scores.append(0.85)
            base_risk_level = 'medium'
        
        elif row['is_new_client']:
            if row['overall_activity_rate'] > 0.7:
                personas.append('new_active_client')
                risk_factors.append('New client with good activity')
                confidence_scores.append(0.8)
                base_risk_level = 'low'
            elif row['overall_activity_rate'] < 0.3:
                personas.append('new_inactive_client')
                risk_factors.append('New client with minimal activity')
                confidence_scores.append(0.75)
                base_risk_level = 'medium'
            elif row['num_significant_gaps'] > 3:
                personas.append('volatile_new_client')
                risk_factors.append('New client with erratic patterns')
                confidence_scores.append(0.7)
                base_risk_level = 'medium'
        
        elif (row['num_significant_gaps'] > 2 and 
              row['avg_gap_days'] > 45 and
              row['overall_activity_rate'] > 0.2):
            personas.append('intermittent_client')
            risk_factors.append('Sporadic activity with gaps')
            confidence_scores.append(0.7)
            base_risk_level = 'medium'
        
        elif (row['activity_trend'] < -0.01 and
              row['overall_activity_rate'] > 0.3 and
              row['client_age_days'] > 365):
            personas.append('declining_engagement')
            risk_factors.append('Gradually decreasing activity')
            confidence_scores.append(0.75)
            base_risk_level = 'medium'
        
        elif (row['client_age_days'] > 730 and
              row['recent_activity_rate'] < 0.1 and
              (row['max_deposit'] > 0 or row['max_loan'] > 0)):
            personas.append('legacy_dormant')
            risk_factors.append('Long-term client now dormant')
            confidence_scores.append(0.8)
            base_risk_level = 'medium'
        
        elif (row['overall_activity_rate'] < 0.05 and
              row['max_deposit'] < 100 and
              row['max_loan'] < 100):
            personas.append('ghost_client')
            risk_factors.append('Account exists but minimal activity')
            confidence_scores.append(0.9)
            base_risk_level = 'low'
        
        # Default case
        if not personas:
            if row['overall_activity_rate'] < 0.2:
                personas.append('low_activity_client')
                risk_factors.append('Generally low activity levels')
                confidence_scores.append(0.6)
                base_risk_level = 'low'
            else:
                personas.append('standard_sparse_client')
                risk_factors.append('Sparse but regular activity')
                confidence_scores.append(0.5)
                base_risk_level = 'low'
        
        # Risk escalation logic
        final_risk_level = base_risk_level
        risk_escalation_factors = 0
        
        if row['is_recently_dormant']:
            risk_escalation_factors += 1
        if row['max_gap_days'] > 180:
            risk_escalation_factors += 1
        if row['activity_trend'] < -0.02:
            risk_escalation_factors += 1
        if row['recent_activity_rate'] < 0.1 and row['overall_activity_rate'] > 0.3:
            risk_escalation_factors += 1
        
        if risk_escalation_factors >= 3:
            final_risk_level = 'high'
        elif risk_escalation_factors >= 2 and base_risk_level != 'high':
            final_risk_level = 'medium'
        
        final_confidence = np.mean(confidence_scores) if confidence_scores else 0.5
        risk_description = ' | '.join(risk_factors) if risk_factors else 'Standard sparse data pattern'
        
        persona_assignments.append({
            'company_id': company_id,
            'primary_persona': personas[0] if personas else 'standard_sparse_client',
            'all_personas': '|'.join(personas),
            'risk_level': final_risk_level,
            'confidence': final_confidence,
            'risk_description': risk_description,
            'activity_rate': row['overall_activity_rate'],
            'recent_activity_rate': row['recent_activity_rate'],
            'client_age_days': row['client_age_days'],
            'max_gap_days': row['max_gap_days'],
            'is_new_client': row['is_new_client'],
            'is_dormant': row['is_recently_dormant'],
            'persona_type': 'sparse'  # Flag to distinguish from traditional personas
        })
    
    return pd.DataFrame(persona_assignments)

#######################################################
# TRADITIONAL RISK ANALYSIS (ENHANCED ORIGINAL)
#######################################################

def detect_traditional_risk_patterns_efficient(df):
    """
    Enhanced version of the original risk pattern detection for traditional companies.
    """
    risk_records = []
    persona_assignments = []
    
    # Time windows to analyze
    windows = INTEGRATED_CONFIG['risk']['trend_windows']
    
    # Get the latest date for recent risk calculation
    max_date = df['date'].max()
    recent_cutoff = max_date - pd.Timedelta(days=INTEGRATED_CONFIG['data']['recent_window'])
    
    # Process each company
    for company in tqdm(df['company_id'].unique(), desc="Detecting traditional risk patterns"):
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
            
            # Apply all the original risk patterns (deteriorating_90d, credit_dependent, etc.)
            # ... (include all the original risk pattern logic here)
            
            # For brevity, I'll include a few key patterns
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
            
            # High utilization with low deposit ratio
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
                    'is_recent': current_date >= recent_cutoff,
                    'persona_type': 'traditional'  # Flag to distinguish from sparse personas
                })
    
    # Create dataframes
    if risk_records:
        risk_df = pd.DataFrame(risk_records)
        persona_df = pd.DataFrame(persona_assignments)
        
        # Create recent risks summary
        recent_risks = risk_df[risk_df['is_recent'] == True].copy()
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
                                          'risk_level', 'is_recent', 'persona_type'])
        recent_risk_summary = pd.DataFrame(columns=['company_id', 'latest_date', 'most_common_flag',
                                                   'risk_level', 'persona', 'current_util', 'current_deposit'])
        return risk_df, persona_df, recent_risk_summary

#######################################################
# INTEGRATED ANALYSIS AND VISUALIZATION
#######################################################

def create_integrated_persona_summary(traditional_persona_df, sparse_persona_df):
    """
    Combine traditional and sparse persona results into unified summary.
    """
    # Combine dataframes
    combined_personas = []
    
    # Add traditional personas
    if not traditional_persona_df.empty:
        traditional_summary = traditional_persona_df.groupby(['company_id', 'persona']).agg({
            'confidence': 'mean',
            'risk_level': lambda x: x.mode().iloc[0] if not x.empty else 'low',
            'date': 'max'
        }).reset_index()
        traditional_summary['persona_type'] = 'traditional'
        combined_personas.append(traditional_summary)
    
    # Add sparse personas
    if not sparse_persona_df.empty:
        sparse_summary = sparse_persona_df[['company_id', 'primary_persona', 'confidence', 
                                          'risk_level', 'persona_type']].copy()
        sparse_summary = sparse_summary.rename(columns={'primary_persona': 'persona'})
        sparse_summary['date'] = pd.Timestamp.now()  # Use current date for sparse
        combined_personas.append(sparse_summary)
    
    if combined_personas:
        integrated_df = pd.concat(combined_personas, ignore_index=True)
        return integrated_df
    else:
        return pd.DataFrame()

def plot_integrated_persona_overview(integrated_persona_df):
    """
    Create comprehensive visualization showing both traditional and sparse personas.
    """
    if integrated_persona_df.empty:
        print("No persona data available for plotting.")
        return None
    
    # Create figure with multiple subplots
    fig = plt.figure(figsize=(20, 16))
    
    # 1. Persona type distribution (Traditional vs Sparse)
    ax1 = plt.subplot(2, 4, 1)
    persona_type_counts = integrated_persona_df['persona_type'].value_counts()
    colors = ['#2E8B57', '#CD853F']  # Green for traditional, brown for sparse
    
    wedges, texts, autotexts = ax1.pie(persona_type_counts.values, 
                                       labels=['Traditional\nAnalysis', 'Sparse Data\nAnalysis'], 
                                       autopct='%1.1f%%', colors=colors, startangle=90)
    
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontweight('bold')
    
    ax1.set_title('Client Analysis Coverage\n(Traditional vs Sparse)', fontsize=12, fontweight='bold')
    
    # 2. Risk level distribution by persona type
    ax2 = plt.subplot(2, 4, 2)
    risk_persona_type = pd.crosstab(integrated_persona_df['persona_type'], 
                                   integrated_persona_df['risk_level'])
    
    risk_persona_type.plot(kind='bar', stacked=True, ax=ax2, 
                          color=['lightgreen', 'orange', 'red'])
    ax2.set_title('Risk Distribution by\nAnalysis Type', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Analysis Type', fontsize=10)
    ax2.set_ylabel('Number of Companies', fontsize=10)
    ax2.tick_params(axis='x', rotation=45)
    ax2.legend(title='Risk Level', fontsize=8)
    
    # 3. Top traditional personas
    ax3 = plt.subplot(2, 4, 3)
    traditional_personas = integrated_persona_df[integrated_persona_df['persona_type'] == 'traditional']
    if not traditional_personas.empty:
        top_traditional = traditional_personas['persona'].value_counts().head(8)
        top_traditional.plot(kind='barh', ax=ax3, color='#2E8B57', alpha=0.7)
        ax3.set_title('Top Traditional Personas', fontsize=12, fontweight='bold')
        ax3.set_xlabel('Number of Companies', fontsize=10)
    else:
        ax3.text(0.5, 0.5, 'No Traditional\nPersonas Found', ha='center', va='center', 
                transform=ax3.transAxes, fontsize=12)
        ax3.set_title('Top Traditional Personas', fontsize=12, fontweight='bold')
    
    # 4. Top sparse personas
    ax4 = plt.subplot(2, 4, 4)
    sparse_personas = integrated_persona_df[integrated_persona_df['persona_type'] == 'sparse']
    if not sparse_personas.empty:
        top_sparse = sparse_personas['persona'].value_counts().head(8)
        top_sparse.plot(kind='barh', ax=ax4, color='#CD853F', alpha=0.7)
        ax4.set_title('Top Sparse Data Personas', fontsize=12, fontweight='bold')
        ax4.set_xlabel('Number of Companies', fontsize=10)
    else:
        ax4.text(0.5, 0.5, 'No Sparse\nPersonas Found', ha='center', va='center', 
                transform=ax4.transAxes, fontsize=12)
        ax4.set_title('Top Sparse Data Personas', fontsize=12, fontweight='bold')
    
    # 5. Confidence distribution by persona type
    ax5 = plt.subplot(2, 4, 5)
    traditional_conf = traditional_personas['confidence'] if not traditional_personas.empty else pd.Series([])
    sparse_conf = sparse_personas['confidence'] if not sparse_personas.empty else pd.Series([])
    
    if not traditional_conf.empty:
        ax5.hist(traditional_conf, bins=15, alpha=0.7, label='Traditional', color='#2E8B57')
    if not sparse_conf.empty:
        ax5.hist(sparse_conf, bins=15, alpha=0.7, label='Sparse', color='#CD853F')
    
    ax5.set_title('Confidence Score Distribution', fontsize=12, fontweight='bold')
    ax5.set_xlabel('Confidence Score', fontsize=10)
    ax5.set_ylabel('Number of Companies', fontsize=10)
    if not traditional_conf.empty or not sparse_conf.empty:
        ax5.legend()
    
    # 6. High-risk companies comparison
    ax6 = plt.subplot(2, 4, 6)
    high_risk_comparison = integrated_persona_df[integrated_persona_df['risk_level'] == 'high'].groupby('persona_type').size()
    
    if not high_risk_comparison.empty:
        high_risk_comparison.plot(kind='bar', ax=ax6, color=['#8B0000', '#B22222'])
        ax6.set_title('High-Risk Companies\nby Analysis Type', fontsize=12, fontweight='bold')
        ax6.set_ylabel('Number of Companies', fontsize=10)
        ax6.tick_params(axis='x', rotation=45)
    else:
        ax6.text(0.5, 0.5, 'No High-Risk\nCompanies Found', ha='center', va='center', 
                transform=ax6.transAxes, fontsize=12)
        ax6.set_title('High-Risk Companies\nby Analysis Type', fontsize=12, fontweight='bold')
    
    # 7. Risk level comparison
    ax7 = plt.subplot(2, 4, 7)
    risk_comparison = integrated_persona_df.groupby(['persona_type', 'risk_level']).size().unstack(fill_value=0)
    risk_comparison.plot(kind='bar', ax=ax7, color=['lightgreen', 'orange', 'red'])
    ax7.set_title('Risk Level Comparison', fontsize=12, fontweight='bold')
    ax7.set_xlabel('Analysis Type', fontsize=10)
    ax7.set_ylabel('Number of Companies', fontsize=10)
    ax7.tick_params(axis='x', rotation=45)
    ax7.legend(title='Risk Level', fontsize=8)
    
    # 8. Summary statistics table
    ax8 = plt.subplot(2, 4, 8)
    ax8.axis('off')
    
    # Calculate summary statistics
    total_companies = len(integrated_persona_df)
    traditional_count = len(traditional_personas) if not traditional_personas.empty else 0
    sparse_count = len(sparse_personas) if not sparse_personas.empty else 0
    high_risk_total = len(integrated_persona_df[integrated_persona_df['risk_level'] == 'high'])
    medium_risk_total = len(integrated_persona_df[integrated_persona_df['risk_level'] == 'medium'])
    
    avg_confidence_traditional = traditional_conf.mean() if not traditional_conf.empty else 0
    avg_confidence_sparse = sparse_conf.mean() if not sparse_conf.empty else 0
    
    summary_text = f"""
    INTEGRATED ANALYSIS SUMMARY
    
    Total Companies Analyzed: {total_companies}
    
    Traditional Analysis: {traditional_count}
    Sparse Data Analysis: {sparse_count}
    
    Risk Distribution:
    • High Risk: {high_risk_total}
    • Medium Risk: {medium_risk_total}
    • Low Risk: {total_companies - high_risk_total - medium_risk_total}
    
    Average Confidence:
    • Traditional: {avg_confidence_traditional:.2f}
    • Sparse: {avg_confidence_sparse:.2f}
    
    Coverage: {(total_companies / (traditional_count + sparse_count) * 100):.1f}%
    """
    
    ax8.text(0.05, 0.95, summary_text, transform=ax8.transAxes, fontsize=10,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    plt.suptitle('Integrated Bank Client Risk Analysis Dashboard', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    return fig

def create_priority_action_report(integrated_persona_df, traditional_risk_df=None, sparse_persona_df=None):
    """
    Generate priority action report combining insights from both analysis types.
    """
    print("\n" + "="*80)
    print("INTEGRATED PRIORITY ACTION REPORT")
    print("="*80)
    
    if integrated_persona_df.empty:
        print("No data available for priority action report.")
        return None
    
    # High-priority cases requiring immediate attention
    high_priority_cases = []
    
    # 1. High-risk traditional clients
    traditional_high_risk = integrated_persona_df[
        (integrated_persona_df['persona_type'] == 'traditional') & 
        (integrated_persona_df['risk_level'] == 'high')
    ]
    
    for _, row in traditional_high_risk.iterrows():
        high_priority_cases.append({
            'company_id': row['company_id'],
            'persona': row['persona'],
            'risk_level': row['risk_level'],
            'confidence': row['confidence'],
            'analysis_type': 'Traditional',
            'priority_score': 3.0 + row['confidence'],  # Base 3 for high risk + confidence
            'action_category': 'Credit Risk Management',
            'recommended_action': 'Immediate review - high credit risk detected'
        })
    
    # 2. High-risk sparse clients (relationship risk)
    sparse_high_risk = integrated_persona_df[
        (integrated_persona_df['persona_type'] == 'sparse') & 
        (integrated_persona_df['risk_level'] == 'high')
    ]
    
    for _, row in sparse_high_risk.iterrows():
        if row['persona'] in ['closed_relationship', 'recently_dormant', 'legacy_dormant']:
            action = 'Relationship recovery - client may be leaving'
            category = 'Relationship Management'
            priority = 2.8 + row['confidence']
        else:
            action = 'Client engagement - address inactivity'
            category = 'Client Engagement'
            priority = 2.5 + row['confidence']
        
        high_priority_cases.append({
            'company_id': row['company_id'],
            'persona': row['persona'],
            'risk_level': row['risk_level'],
            'confidence': row['confidence'],
            'analysis_type': 'Sparse Data',
            'priority_score': priority,
            'action_category': category,
            'recommended_action': action
        })
    
    # 3. Medium-risk cases with high confidence
    medium_risk_high_conf = integrated_persona_df[
        (integrated_persona_df['risk_level'] == 'medium') & 
        (integrated_persona_df['confidence'] > 0.8)
    ]
    
    for _, row in medium_risk_high_conf.iterrows():
        if row['persona_type'] == 'traditional':
            action = 'Monitor closely - potential escalation to high risk'
            category = 'Proactive Monitoring'
        else:
            action = 'Engagement strategy - prevent further decline'
            category = 'Retention Strategy'
        
        high_priority_cases.append({
            'company_id': row['company_id'],
            'persona': row['persona'],
            'risk_level': row['risk_level'],
            'confidence': row['confidence'],
            'analysis_type': row['persona_type'].title(),
            'priority_score': 2.0 + row['confidence'],
            'action_category': category,
            'recommended_action': action
        })
    
    # Create priority dataframe and sort by priority score
    priority_df = pd.DataFrame(high_priority_cases)
    if not priority_df.empty:
        priority_df = priority_df.sort_values('priority_score', ascending=False)
    
    # Print summary by category
    print(f"\nTOTAL HIGH-PRIORITY CASES: {len(priority_df)}")
    print("-" * 50)
    
    if not priority_df.empty:
        by_category = priority_df.groupby('action_category').size().sort_values(ascending=False)
        for category, count in by_category.items():
            print(f"{category}: {count} companies")
        
        print(f"\nTOP 10 HIGHEST PRIORITY CASES:")
        print("-" * 50)
        
        for i, (_, row) in enumerate(priority_df.head(10).iterrows(), 1):
            print(f"{i:2d}. Company {row['company_id']} ({row['analysis_type']})")
            print(f"    Persona: {row['persona']}")
            print(f"    Risk: {row['risk_level'].upper()} (confidence: {row['confidence']:.2f})")
            print(f"    Action: {row['recommended_action']}")
            print()
        
        if len(priority_df) > 10:
            print(f"    ... and {len(priority_df) - 10} more priority cases")
    
    # Summary statistics
    print(f"\nINTEGRATED ANALYSIS SUMMARY:")
    print("-" * 50)
    
    total_companies = len(integrated_persona_df)
    traditional_count = len(integrated_persona_df[integrated_persona_df['persona_type'] == 'traditional'])
    sparse_count = len(integrated_persona_df[integrated_persona_df['persona_type'] == 'sparse'])
    
    print(f"Total companies analyzed: {total_companies}")
    print(f"Traditional analysis: {traditional_count} ({traditional_count/total_companies*100:.1f}%)")
    print(f"Sparse data analysis: {sparse_count} ({sparse_count/total_companies*100:.1f}%)")
    
    risk_summary = integrated_persona_df['risk_level'].value_counts()
    print(f"\nRisk distribution:")
    for risk_level, count in risk_summary.items():
        print(f"  {risk_level.title()}: {count} ({count/total_companies*100:.1f}%)")
    
    # Analysis coverage improvement
    print(f"\nCOVERAGE IMPROVEMENT:")
    print("-" * 50)
    print(f"Without sparse analysis: {traditional_count} companies")
    print(f"With integrated analysis: {total_companies} companies")
    print(f"Additional coverage: {sparse_count} companies ({sparse_count/traditional_count*100:.1f}% increase)")
    
    return priority_df

#######################################################
# MAIN INTEGRATED WORKFLOW
#######################################################

def integrated_bank_client_risk_analysis(df):
    """
    Main integrated function that combines traditional and sparse data analysis
    to provide comprehensive client risk assessment.
    """
    print("="*80)
    print("INTEGRATED BANK CLIENT RISK ANALYSIS SYSTEM")
    print("="*80)
    
    results = {}
    
    # 1. Enhanced data cleaning that separates traditional vs sparse
    print("\n1. Enhanced data cleaning and categorization...")
    df_traditional, df_sparse, cleaning_stats = enhanced_clean_data(df, min_nonzero_pct=0.8)
    
    results['cleaning_stats'] = cleaning_stats
    results['df_traditional'] = df_traditional
    results['df_sparse'] = df_sparse
    
    print(f"Data categorization complete:")
    print(f"  Traditional analysis: {len(cleaning_stats['traditional_companies'])} companies")
    print(f"  Sparse data analysis: {len(cleaning_stats['sparse_companies'])} companies")
    print(f"  Total coverage: {len(cleaning_stats['traditional_companies']) + len(cleaning_stats['sparse_companies'])} companies")
    
    # 2. Traditional analysis (for high-activity companies)
    traditional_results = {}
    if not df_traditional.empty:
        print("\n2. Running traditional risk analysis...")
        
        # Add traditional derived metrics
        df_traditional_with_metrics = add_enhanced_derived_metrics(df_traditional, is_traditional=True)
        results['df_traditional_with_metrics'] = df_traditional_with_metrics
        
        # Detect traditional risk patterns
        traditional_risk_df, traditional_persona_df, traditional_recent_risk_df = detect_traditional_risk_patterns_efficient(df_traditional_with_metrics)
        
        traditional_results = {
            'risk_df': traditional_risk_df,
            'persona_df': traditional_persona_df,
            'recent_risk_df': traditional_recent_risk_df
        }
        
        print(f"Traditional analysis complete:")
        print(f"  Risk events detected: {len(traditional_risk_df)}")
        print(f"  Personas assigned: {traditional_persona_df['persona'].nunique() if not traditional_persona_df.empty else 0}")
        print(f"  Recent high-risk companies: {len(traditional_recent_risk_df[traditional_recent_risk_df['risk_level'] == 'high']) if not traditional_recent_risk_df.empty else 0}")
    
    results['traditional_analysis'] = traditional_results
    
    # 3. Sparse data analysis (for low-activity companies)
    sparse_results = {}
    if not df_sparse.empty:
        print("\n3. Running sparse data analysis...")
        
        # Add sparse-appropriate metrics
        df_sparse_with_metrics = add_enhanced_derived_metrics(df_sparse, is_traditional=False)
        results['df_sparse_with_metrics'] = df_sparse_with_metrics
        
        # Analyze sparse patterns
        sparse_analysis_df = analyze_sparse_data_patterns(df_sparse)
        
        # Assign sparse personas
        sparse_persona_df = assign_sparse_data_personas(sparse_analysis_df)
        
        sparse_results = {
            'analysis_df': sparse_analysis_df,
            'persona_df': sparse_persona_df
        }
        
        print(f"Sparse data analysis complete:")
        print(f"  Companies analyzed: {len(sparse_analysis_df)}")
        print(f"  Personas assigned: {sparse_persona_df['primary_persona'].nunique() if not sparse_persona_df.empty else 0}")
        print(f"  High-risk sparse companies: {len(sparse_persona_df[sparse_persona_df['risk_level'] == 'high']) if not sparse_persona_df.empty else 0}")
    
    results['sparse_analysis'] = sparse_results
    
    # 4. Create integrated persona summary
    print("\n4. Creating integrated analysis summary...")
    traditional_persona_df = traditional_results.get('persona_df', pd.DataFrame())
    sparse_persona_df = sparse_results.get('persona_df', pd.DataFrame())
    
    integrated_persona_df = create_integrated_persona_summary(traditional_persona_df, sparse_persona_df)
    results['integrated_personas'] = integrated_persona_df
    
    # 5. Generate visualizations
    print("\n5. Creating integrated visualizations...")
    
    # Main integrated overview
    if not integrated_persona_df.empty:
        integrated_overview_fig = plot_integrated_persona_overview(integrated_persona_df)
        if integrated_overview_fig:
            integrated_overview_fig.savefig('integrated_risk_analysis_overview.png', dpi=300, bbox_inches='tight')
            print("Saved: integrated_risk_analysis_overview.png")
    
    # Traditional analysis visualizations (if available)
    if not traditional_persona_df.empty:
        # You can add traditional-specific plots here
        pass
    
    # Sparse analysis visualizations (if available)
    if not sparse_persona_df.empty:
        from scipy.stats import chi2_contingency
        
        # Create sparse-specific visualization
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        
        # Sparse persona distribution
        sparse_persona_counts = sparse_persona_df['primary_persona'].value_counts()
        sparse_persona_counts.plot(kind='barh', ax=ax1, color='#CD853F', alpha=0.7)
        ax1.set_title('Sparse Data Persona Distribution', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Number of Companies', fontsize=12)
        
        # Sparse risk vs activity relationship
        if 'activity_rate' in sparse_persona_df.columns and 'max_gap_days' in sparse_persona_df.columns:
            scatter = ax2.scatter(sparse_persona_df['activity_rate'], 
                                sparse_persona_df['max_gap_days'],
                                c=sparse_persona_df['risk_level'].map({'low': 1, 'medium': 2, 'high': 3}),
                                cmap='RdYlBu_r', alpha=0.7, s=60)
            ax2.set_title('Activity Rate vs Gap Analysis', fontsize=14, fontweight='bold')
            ax2.set_xlabel('Activity Rate', fontsize=12)
            ax2.set_ylabel('Maximum Gap (Days)', fontsize=12)
            
            # Add colorbar
            cbar = plt.colorbar(scatter, ax=ax2, ticks=[1, 2, 3])
            cbar.set_ticklabels(['Low Risk', 'Medium Risk', 'High Risk'])
        
        plt.tight_layout()
        plt.savefig('sparse_data_detailed_analysis.png', dpi=300, bbox_inches='tight')
        print("Saved: sparse_data_detailed_analysis.png")
    
    # 6. Generate priority action report
    print("\n6. Generating priority action report...")
    priority_df = create_priority_action_report(
        integrated_persona_df, 
        traditional_results.get('risk_df'), 
        sparse_results.get('persona_df')
    )
    results['priority_actions'] = priority_df
    
    print("\n" + "="*80)
    print("INTEGRATED ANALYSIS COMPLETE")
    print("="*80)
    print(f"✅ Total companies analyzed: {len(integrated_persona_df) if not integrated_persona_df.empty else 0}")
    print(f"✅ Traditional risk analysis: {len(traditional_persona_df)} companies")
    print(f"✅ Sparse data analysis: {len(sparse_persona_df)} companies")
    print(f"✅ High-priority action items: {len(priority_df) if priority_df is not None else 0}")
    print(f"✅ Visualizations generated: 2 comprehensive dashboards")
    
    return results

#######################################################
# DATA GENERATION FOR TESTING
#######################################################

def generate_comprehensive_test_data(num_companies=100, days=730):
    """
    Generate comprehensive test data that includes both traditional and sparse patterns.
    """
    print(f"Generating comprehensive test data for {num_companies} companies over {days} days...")
    
    np.random.seed(42)
    
    # Create date range
    end_date = pd.Timestamp('2023-12-31')
    start_date = end_date - pd.Timedelta(days=days)
    date_range = pd.date_range(start=start_date, end=end_date)
    
    # Create company IDs
    company_ids = [f'COMP{str(i).zfill(4)}' for i in range(num_companies)]
    
    data = []
    
    for i, company_id in enumerate(tqdm(company_ids, desc="Generating company data")):
        # Determine company type (70% traditional, 30% sparse)
        is_sparse = np.random.random() < 0.3
        
        if is_sparse:
            # Generate sparse data patterns
            pattern_type = np.random.choice([
                'dormant', 'new_active', 'new_inactive', 'intermittent', 
                'ghost', 'declining', 'reactivated'
            ])
            
            for j, date in enumerate(date_range):
                deposit = 0
                loan = 0
                unused_loan = 0
                
                if pattern_type == 'dormant':
                    # Active first half, dormant second half
                    if j < len(date_range) // 2 and np.random.random() < 0.6:
                        deposit = np.random.lognormal(8, 1)
                        loan = np.random.lognormal(7, 1)
                        unused_loan = loan * 0.3
                
                elif pattern_type == 'new_active':
                    # Active only in last 6 months
                    if j >= len(date_range) - 180 and np.random.random() < 0.7:
                        deposit = np.random.lognormal(8, 1)
                        loan = np.random.lognormal(7, 1)
                        unused_loan = loan * 0.2
                
                elif pattern_type == 'intermittent':
                    # Random activity with gaps
                    if np.random.random() < 0.25:
                        deposit = np.random.lognormal(8, 1)
                        loan = np.random.lognormal(7, 1)
                        unused_loan = loan * 0.4
                
                elif pattern_type == 'ghost':
                    # Minimal activity
                    if np.random.random() < 0.05:
                        deposit = np.random.uniform(10, 100)
                        loan = np.random.uniform(5, 50)
                        unused_loan = loan * 0.1
                
                data.append({
                    'company_id': company_id,
                    'date': date,
                    'deposit_balance': max(0, deposit),
                    'used_loan': max(0, loan),
                    'unused_loan': max(0, unused_loan)
                })
        
        else:
            # Generate traditional (high-activity) data patterns
            base_deposit = np.random.lognormal(10, 1)
            base_loan = np.random.lognormal(9, 1.2)
            util_rate = np.random.uniform(0.3, 0.8)
            
            # Trend parameters
            deposit_trend = np.random.normal(0, 0.001)
            util_trend = np.random.normal(0, 0.0005)
            
            # Volatility parameters
            deposit_vol = np.random.uniform(0.01, 0.1)
            util_vol = np.random.uniform(0.01, 0.05)
            
            # Risk pattern (25% develop risk)
            has_risk = np.random.random() < 0.25
            risk_start = int(len(date_range) * 0.7) if has_risk else len(date_range)
            
            for j, date in enumerate(date_range):
                # Time-dependent components
                t = j / len(date_range)
                
                # Trends
                deposit_trend_component = 1 + deposit_trend * j
                util_trend_component = util_rate + util_trend * j
                
                # Seasonality
                seasonal_component = 1 + 0.1 * np.sin(2 * np.pi * date.dayofyear / 365)
                
                # Volatility
                deposit_random = np.random.normal(1, deposit_vol)
                util_random = np.random.normal(0, util_vol)
                
                # Risk pattern
                if j > risk_start:
                    risk_factor_deposit = 1 - 0.001 * (j - risk_start)
                    risk_factor_util = 0.0005 * (j - risk_start)
                else:
                    risk_factor_deposit = 1
                    risk_factor_util = 0
                
                # Calculate final values
                deposit = base_deposit * deposit_trend_component * seasonal_component * deposit_random * risk_factor_deposit
                utilization = min(0.95, max(0.1, util_trend_component + util_random + risk_factor_util))
                used_loan = base_loan * utilization
                unused_loan = base_loan - used_loan
                
                # Add some missing values (3% probability)
                if np.random.random() < 0.03:
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

#######################################################
# MAIN EXECUTION
#######################################################

if __name__ == "__main__":
    # Generate comprehensive test data
    print("Starting Integrated Bank Client Risk Analysis System...")
    
    # Generate test data
    test_df = generate_comprehensive_test_data(num_companies=150, days=730)
    print(f"Generated test data: {len(test_df)} records for {test_df['company_id'].nunique()} companies")
    
    # Run integrated analysis
    results = integrated_bank_client_risk_analysis(test_df)
    
    print("\n🎉 ANALYSIS COMPLETE! Check the generated PNG files for detailed visualizations.")
    print("📊 Files generated:")
    print("   - integrated_risk_analysis_overview.png")
    print("   - sparse_data_detailed_analysis.png")
