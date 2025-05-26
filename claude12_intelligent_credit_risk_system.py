import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from scipy import stats
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.ensemble import IsolationForest
from sklearn.metrics import silhouette_score, adjusted_rand_score
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.model_selection import GridSearchCV
from tqdm import tqdm
import warnings
import json
from collections import defaultdict, Counter
from typing import Dict, List, Tuple, Optional, Union
import itertools

warnings.filterwarnings('ignore')

# Set visualization style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("viridis")

#######################################################
# SECTION 1: RULE GENERATION AND FINE-TUNING SYSTEM
#######################################################

def generate_enhanced_synthetic_data(num_companies: int = 200, days: int = 730, 
                                   include_drift: bool = True, random_seed: int = 42) -> pd.DataFrame:
    """
    Generate sophisticated synthetic banking data with realistic patterns for rule discovery.
    
    This function creates diverse company profiles with different risk patterns, seasonal behaviors,
    and realistic financial trajectories. The data includes both healthy companies and those that
    develop various types of financial stress over time.
    
    Parameters:
    -----------
    num_companies : int
        Number of companies to generate (default: 200)
    days : int 
        Number of days of historical data (default: 730, about 2 years)
    include_drift : bool
        Whether to include data drift patterns (default: True)
    random_seed : int
        Random seed for reproducible results (default: 42)
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with columns: company_id, date, deposit_balance, used_loan, unused_loan
    """
    
    print(f"Generating sophisticated synthetic data for {num_companies} companies over {days} days...")
    np.random.seed(random_seed)
    
    # Create date range
    end_date = pd.Timestamp('2024-12-31')
    start_date = end_date - pd.Timedelta(days=days)
    date_range = pd.date_range(start=start_date, end=end_date, freq='D')
    
    # Define company archetypes with realistic parameters
    company_archetypes = {
        'stable_growth': {
            'probability': 0.35,  # 35% of companies
            'deposit_growth_rate': 0.0002,  # Steady growth
            'util_volatility': 0.02,
            'seasonal_strength': 0.05,
            'risk_development': False
        },
        'seasonal_business': {
            'probability': 0.20,  # 20% of companies  
            'deposit_growth_rate': 0.0001,
            'util_volatility': 0.05,
            'seasonal_strength': 0.25,  # Strong seasonality
            'risk_development': False
        },
        'high_growth_volatile': {
            'probability': 0.15,  # 15% of companies
            'deposit_growth_rate': 0.0005,  # Fast growth
            'util_volatility': 0.08,  # High volatility
            'seasonal_strength': 0.03,
            'risk_development': False
        },
        'deteriorating_slowly': {
            'probability': 0.15,  # 15% of companies
            'deposit_growth_rate': 0.0001,
            'util_volatility': 0.04,
            'seasonal_strength': 0.08,
            'risk_development': True,
            'risk_start_pct': 0.6  # Risk starts at 60% through period
        },
        'crisis_development': {
            'probability': 0.10,  # 10% of companies
            'deposit_growth_rate': -0.0001,  # Slight decline
            'util_volatility': 0.06,
            'seasonal_strength': 0.05,
            'risk_development': True,
            'risk_start_pct': 0.7,  # Risk starts later
            'crisis_severity': 'high'
        },
        'cyclical_stress': {
            'probability': 0.05,  # 5% of companies
            'deposit_growth_rate': 0.0001,
            'util_volatility': 0.07,
            'seasonal_strength': 0.15,
            'risk_development': True,
            'risk_pattern': 'cyclical'  # Repeating stress patterns
        }
    }
    
    # Generate company assignments based on probabilities
    company_types = []
    cumulative_prob = 0
    for archetype, params in company_archetypes.items():
        count = int(num_companies * params['probability'])
        company_types.extend([archetype] * count)
    
    # Fill remaining companies with stable_growth
    while len(company_types) < num_companies:
        company_types.append('stable_growth')
    
    # Shuffle company assignments
    np.random.shuffle(company_types)
    
    # Generate data for each company
    data = []
    
    for i, company_type in enumerate(tqdm(company_types, desc="Generating company data")):
        company_id = f'COMP_{i:04d}'
        archetype = company_archetypes[company_type]
        
        # Initialize company parameters
        base_deposit = np.random.lognormal(mean=12, sigma=1.2)  # ~$160K average
        base_loan = np.random.lognormal(mean=11.5, sigma=1.0)   # ~$100K average
        initial_utilization = np.random.uniform(0.2, 0.7)
        
        # Get archetype-specific parameters
        growth_rate = archetype['deposit_growth_rate']
        volatility = archetype['util_volatility']
        seasonal_strength = archetype['seasonal_strength']
        
        # Risk development parameters
        develops_risk = archetype.get('risk_development', False)
        risk_start_day = int(len(date_range) * archetype.get('risk_start_pct', 0.8)) if develops_risk else len(date_range)
        crisis_severity = archetype.get('crisis_severity', 'medium')
        
        # Data drift simulation (affects last 25% of data)
        drift_start = int(len(date_range) * 0.75) if include_drift else len(date_range)
        
        # Generate time series for this company
        for day_idx, current_date in enumerate(date_range):
            
            # Calculate time-based factors
            days_from_start = day_idx
            time_progress = day_idx / len(date_range)
            
            # 1. Base trend component
            deposit_trend = 1 + (growth_rate * days_from_start)
            util_trend = initial_utilization
            
            # 2. Seasonal component
            if seasonal_strength > 0:
                day_of_year = current_date.dayofyear
                seasonal_deposit = 1 + seasonal_strength * np.sin(2 * np.pi * day_of_year / 365)
                seasonal_util = seasonal_strength * 0.5 * np.cos(2 * np.pi * day_of_year / 365)
            else:
                seasonal_deposit = 1
                seasonal_util = 0
            
            # 3. Random volatility
            deposit_noise = np.random.normal(1, volatility * 0.5)
            util_noise = np.random.normal(0, volatility)
            
            # 4. Risk development patterns
            risk_factor_deposit = 1
            risk_factor_util = 0
            
            if develops_risk and day_idx >= risk_start_day:
                risk_progress = (day_idx - risk_start_day) / (len(date_range) - risk_start_day)
                
                if company_type == 'deteriorating_slowly':
                    # Gradual deterioration
                    risk_factor_deposit = 1 - (0.3 * risk_progress)  # 30% deposit decline
                    risk_factor_util = 0.25 * risk_progress  # 25% util increase
                    
                elif company_type == 'crisis_development':
                    # Accelerating crisis
                    if crisis_severity == 'high':
                        risk_factor_deposit = 1 - (0.5 * risk_progress ** 0.7)  # Accelerating decline
                        risk_factor_util = 0.4 * risk_progress ** 0.5  # Quick util increase
                    else:
                        risk_factor_deposit = 1 - (0.35 * risk_progress)
                        risk_factor_util = 0.3 * risk_progress
                        
                elif company_type == 'cyclical_stress':
                    # Cyclical pattern with increasing severity
                    cycle_factor = np.sin(2 * np.pi * risk_progress * 3)  # 3 cycles during risk period
                    risk_factor_deposit = 1 - (0.2 * abs(cycle_factor) * (1 + risk_progress))
                    risk_factor_util = 0.15 * abs(cycle_factor) * (1 + risk_progress)
            
            # 5. Data drift effects (simulate changing market conditions)
            drift_factor_deposit = 1
            drift_factor_util = 0
            
            if include_drift and day_idx >= drift_start:
                drift_progress = (day_idx - drift_start) / (len(date_range) - drift_start)
                # Simulate market stress affecting all companies
                drift_factor_deposit = 1 - (0.1 * drift_progress)  # 10% market-wide deposit pressure
                drift_factor_util = 0.05 * drift_progress  # 5% increased borrowing
            
            # 6. Special events (random shocks)
            event_factor_deposit = 1
            event_factor_util = 0
            
            if np.random.random() < 0.005:  # 0.5% chance per day of special event
                event_type = np.random.choice(['positive', 'negative'], p=[0.3, 0.7])
                if event_type == 'positive':
                    event_factor_deposit = np.random.uniform(1.1, 1.3)  # 10-30% boost
                else:
                    event_factor_deposit = np.random.uniform(0.8, 0.95)  # 5-20% drop
                    event_factor_util = np.random.uniform(0.02, 0.08)  # 2-8% util increase
            
            # Calculate final values
            final_deposit = (base_deposit * deposit_trend * seasonal_deposit * 
                           deposit_noise * risk_factor_deposit * drift_factor_deposit * event_factor_deposit)
            
            final_utilization = min(0.98, max(0.05, 
                util_trend + seasonal_util + util_noise + risk_factor_util + drift_factor_util + event_factor_util))
            
            used_loan = base_loan * final_utilization
            unused_loan = base_loan * (1 - final_utilization)
            
            # Add some realistic missing data (2% probability)
            if np.random.random() < 0.02:
                if np.random.random() < 0.5:
                    final_deposit = 0  # Missing deposit data
                else:
                    used_loan = 0      # Missing loan data
                    unused_loan = 0
            
            # Store the record
            data.append({
                'company_id': company_id,
                'date': current_date,
                'deposit_balance': max(0, final_deposit),
                'used_loan': max(0, used_loan),
                'unused_loan': max(0, unused_loan),
                'company_type': company_type  # Keep for validation purposes
            })
    
    df = pd.DataFrame(data)
    print(f"Generated {len(df)} records with {len(df['company_id'].unique())} companies")
    print(f"Company type distribution:")
    for company_type, count in Counter([row['company_type'] for row in data if 'company_type' in row]).items():
        print(f"  {company_type}: {count//len(date_range)} companies")
    
    return df

def calculate_comprehensive_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate comprehensive financial features for rule discovery.
    
    This function creates a rich set of features that capture different aspects of financial behavior,
    including trends, volatility, seasonality, and behavioral patterns. These features form the basis
    for discovering meaningful risk patterns.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input data with columns: company_id, date, deposit_balance, used_loan, unused_loan
        
    Returns:
    --------
    pd.DataFrame
        Enhanced dataframe with comprehensive financial features
    """
    
    print("Calculating comprehensive financial features for rule discovery...")
    
    # Create a copy and basic derived metrics
    df_features = df.copy()
    df_features['total_loan'] = df_features['used_loan'] + df_features['unused_loan']
    df_features['loan_utilization'] = np.where(
        df_features['total_loan'] > 0,
        df_features['used_loan'] / df_features['total_loan'],
        0
    )
    df_features['deposit_loan_ratio'] = np.where(
        df_features['used_loan'] > 0,
        df_features['deposit_balance'] / df_features['used_loan'],
        np.inf
    )
    
    # Ensure date is datetime
    df_features['date'] = pd.to_datetime(df_features['date'])
    
    # Calculate features for each company
    feature_records = []
    
    for company_id in tqdm(df_features['company_id'].unique(), desc="Computing features"):
        company_data = df_features[df_features['company_id'] == company_id].sort_values('date').copy()
        
        if len(company_data) < 60:  # Need at least 2 months of data
            continue
        
        # Calculate rolling windows and trends
        windows = [7, 30, 60, 90, 180]
        
        for window in windows:
            if len(company_data) >= window:
                # Rolling means
                company_data[f'util_mean_{window}d'] = company_data['loan_utilization'].rolling(
                    window, min_periods=max(3, window//4)).mean()
                company_data[f'deposit_mean_{window}d'] = company_data['deposit_balance'].rolling(
                    window, min_periods=max(3, window//4)).mean()
                
                # Rolling standard deviations (volatility)
                company_data[f'util_vol_{window}d'] = company_data['loan_utilization'].rolling(
                    window, min_periods=max(3, window//4)).std()
                company_data[f'deposit_vol_{window}d'] = company_data['deposit_balance'].pct_change().rolling(
                    window, min_periods=max(3, window//4)).std()
                
                # Trend calculations (linear regression slope)
                if window >= 30:
                    util_trend = company_data['loan_utilization'].rolling(
                        window, min_periods=window//2).apply(
                        lambda x: calculate_trend_slope(x), raw=False)
                    company_data[f'util_trend_{window}d'] = util_trend
                    
                    deposit_trend = company_data['deposit_balance'].rolling(
                        window, min_periods=window//2).apply(
                        lambda x: calculate_trend_slope(x), raw=False)
                    company_data[f'deposit_trend_{window}d'] = deposit_trend
        
        # Calculate change rates between different periods
        for period in [30, 60, 90]:
            if len(company_data) > period:
                company_data[f'util_change_{period}d'] = company_data['loan_utilization'].pct_change(periods=period)
                company_data[f'deposit_change_{period}d'] = company_data['deposit_balance'].pct_change(periods=period)
        
        # Behavioral pattern features
        company_data = calculate_behavioral_features(company_data)
        
        # Seasonality detection
        company_data = detect_advanced_seasonality(company_data)
        
        # Risk acceleration features (second derivatives)
        company_data['util_acceleration_30d'] = company_data['util_change_30d'].pct_change(periods=30)
        company_data['deposit_acceleration_30d'] = company_data['deposit_change_30d'].pct_change(periods=30)
        
        # Add to feature records
        for _, row in company_data.iterrows():
            feature_records.append(row.to_dict())
    
    df_enhanced = pd.DataFrame(feature_records)
    
    print(f"Calculated features for {df_enhanced['company_id'].nunique()} companies")
    print(f"Total features: {len([col for col in df_enhanced.columns if any(suffix in col for suffix in ['_mean_', '_vol_', '_trend_', '_change_'])])} derived features")
    
    return df_enhanced

def calculate_trend_slope(series: pd.Series) -> float:
    """
    Calculate the slope of a linear trend for a time series.
    
    This helper function fits a linear regression to the series and returns the slope,
    which indicates the direction and strength of the trend.
    """
    if len(series) < 3 or series.isna().sum() > len(series) * 0.5:
        return 0.0
    
    # Remove NaN values
    clean_series = series.dropna()
    if len(clean_series) < 3:
        return 0.0
    
    try:
        x = np.arange(len(clean_series))
        slope, _, _, _, _ = stats.linregress(x, clean_series.values)
        return slope
    except:
        return 0.0

def calculate_behavioral_features(company_data: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate behavioral pattern features that capture spending and deposit patterns.
    
    These features help identify companies with unusual behavioral patterns like
    concentrated deposits, irregular withdrawals, or erratic borrowing behavior.
    """
    
    # Deposit pattern analysis
    company_data['deposit_change'] = company_data['deposit_balance'].diff()
    
    # Large movements (defined as >10% of average balance)
    avg_balance = company_data['deposit_balance'].mean()
    if avg_balance > 0:
        large_threshold = avg_balance * 0.1
        company_data['large_deposit_flag'] = (company_data['deposit_change'] > large_threshold).astype(int)
        company_data['large_withdrawal_flag'] = (company_data['deposit_change'] < -large_threshold).astype(int)
    else:
        company_data['large_deposit_flag'] = 0
        company_data['large_withdrawal_flag'] = 0
    
    # Rolling counts of large movements
    for window in [30, 60]:
        company_data[f'large_deposits_{window}d'] = company_data['large_deposit_flag'].rolling(window).sum()
        company_data[f'large_withdrawals_{window}d'] = company_data['large_withdrawal_flag'].rolling(window).sum()
    
    # Utilization behavior patterns
    company_data['util_change_1d'] = company_data['loan_utilization'].diff()
    
    # Spike detection (sudden increases in utilization)
    util_std = company_data['loan_utilization'].std()
    if util_std > 0:
        spike_threshold = util_std * 2
        company_data['util_spike_flag'] = (company_data['util_change_1d'] > spike_threshold).astype(int)
    else:
        company_data['util_spike_flag'] = 0
    
    # Days near credit limit (>90% utilization)
    company_data['near_limit_flag'] = (company_data['loan_utilization'] > 0.9).astype(int)
    
    # Rolling behavioral aggregates
    for window in [30, 60]:
        company_data[f'util_spikes_{window}d'] = company_data['util_spike_flag'].rolling(window).sum()
        company_data[f'near_limit_days_{window}d'] = company_data['near_limit_flag'].rolling(window).sum()
    
    return company_data

def detect_advanced_seasonality(company_data: pd.DataFrame) -> pd.DataFrame:
    """
    Detect sophisticated seasonality patterns in financial data.
    
    This function uses FFT and statistical methods to identify seasonal patterns
    in both utilization and deposit behavior, including annual, quarterly, and monthly cycles.
    """
    
    min_periods_for_seasonality = 180  # 6 months minimum
    
    if len(company_data) < min_periods_for_seasonality:
        # Add default values for companies with insufficient data
        company_data['util_seasonality_strength'] = 0
        company_data['deposit_seasonality_strength'] = 0
        company_data['util_seasonal_period'] = 0
        company_data['deposit_seasonal_period'] = 0
        return company_data
    
    # Analyze utilization seasonality
    util_series = company_data['loan_utilization'].fillna(method='ffill').fillna(method='bfill')
    util_seasonality = analyze_seasonality_fft(util_series)
    
    # Analyze deposit seasonality  
    deposit_series = company_data['deposit_balance'].fillna(method='ffill').fillna(method='bfill')
    deposit_seasonality = analyze_seasonality_fft(deposit_series)
    
    # Add seasonality features
    company_data['util_seasonality_strength'] = util_seasonality['strength']
    company_data['util_seasonal_period'] = util_seasonality['period']
    company_data['deposit_seasonality_strength'] = deposit_seasonality['strength']
    company_data['deposit_seasonal_period'] = deposit_seasonality['period']
    
    return company_data

def analyze_seasonality_fft(series: pd.Series) -> Dict:
    """
    Use FFT to detect seasonality in a time series.
    
    This function applies Fast Fourier Transform to identify dominant frequencies
    that correspond to seasonal patterns.
    """
    
    if len(series) < 90 or series.std() == 0:
        return {'strength': 0, 'period': 0}
    
    try:
        # Detrend the series
        detrended = series - series.rolling(30, center=True).mean().fillna(series.mean())
        
        # Apply FFT
        fft_values = np.fft.fft(detrended.values)
        fft_freq = np.fft.fftfreq(len(detrended))
        
        # Find dominant frequencies (excluding DC component)
        magnitude = np.abs(fft_values[1:len(fft_values)//2])
        
        if len(magnitude) == 0:
            return {'strength': 0, 'period': 0}
        
        # Identify peaks in frequency domain
        peak_idx = np.argmax(magnitude)
        peak_freq = np.abs(fft_freq[peak_idx + 1])
        
        if peak_freq > 0:
            period = int(1 / peak_freq)
            # Focus on meaningful periods (weekly to yearly)
            if 7 <= period <= 400:
                strength = magnitude[peak_idx] / np.sum(magnitude) if np.sum(magnitude) > 0 else 0
                return {'strength': min(strength * 10, 1.0), 'period': period}
        
        return {'strength': 0, 'period': 0}
        
    except Exception:
        return {'strength': 0, 'period': 0}

def discover_risk_patterns_unsupervised(df_features: pd.DataFrame, 
                                       n_clusters_range: Tuple[int, int] = (3, 12)) -> Dict:
    """
    Discover risk patterns using unsupervised learning techniques.
    
    This function uses clustering algorithms to automatically identify groups of companies
    with similar financial behavior patterns. It then analyzes these clusters to extract
    meaningful risk rules.
    
    Parameters:
    -----------
    df_features : pd.DataFrame
        DataFrame with comprehensive financial features
    n_clusters_range : Tuple[int, int]
        Range of cluster numbers to try (default: 3 to 12)
        
    Returns:
    --------
    Dict
        Dictionary containing discovered patterns, rules, and cluster information
    """
    
    print("Discovering risk patterns using unsupervised learning...")
    
    # Prepare feature matrix for clustering
    feature_columns = [col for col in df_features.columns if any(
        suffix in col for suffix in ['_mean_', '_vol_', '_trend_', '_change_', '_30d', '_60d', '_90d']
    )]
    
    # Get the most recent record for each company (for cross-sectional analysis)
    latest_data = df_features.groupby('company_id').last().reset_index()
    
    if len(latest_data) < 20:
        print("Insufficient data for pattern discovery")
        return {}
    
    # Select meaningful features for clustering
    clustering_features = []
    for col in feature_columns:
        if col in latest_data.columns and latest_data[col].notna().sum() > len(latest_data) * 0.5:
            clustering_features.append(col)
    
    if len(clustering_features) < 5:
        print("Insufficient features for pattern discovery")
        return {}
    
    # Prepare clustering data
    X = latest_data[clustering_features].fillna(0)
    
    # Remove features with zero variance
    feature_variance = X.var()
    X = X.loc[:, feature_variance > 0]
    clustering_features = [col for col in clustering_features if col in X.columns]
    
    print(f"Using {len(clustering_features)} features for clustering: {len(X)} companies")
    
    # Scale features
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Find optimal number of clusters
    best_score = -1
    best_n_clusters = 5
    best_labels = None
    
    for n_clusters in range(n_clusters_range[0], n_clusters_range[1] + 1):
        try:
            # Try K-means clustering
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            labels = kmeans.fit_predict(X_scaled)
            
            # Calculate silhouette score
            if len(np.unique(labels)) > 1:
                score = silhouette_score(X_scaled, labels)
                if score > best_score:
                    best_score = score
                    best_n_clusters = n_clusters
                    best_labels = labels
                    
        except Exception as e:
            print(f"Error with {n_clusters} clusters: {e}")
            continue
    
    if best_labels is None:
        print("Could not find suitable clustering solution")
        return {}
    
    print(f"Optimal clustering: {best_n_clusters} clusters with silhouette score: {best_score:.3f}")
    
    # Analyze discovered clusters
    latest_data['cluster'] = best_labels
    discovered_patterns = analyze_discovered_clusters(latest_data, clustering_features, scaler)
    
    # Generate interpretable rules from patterns
    risk_rules = generate_rules_from_patterns(discovered_patterns, clustering_features)
    
    return {
        'patterns': discovered_patterns,
        'rules': risk_rules,
        'clustering_features': clustering_features,
        'scaler': scaler,
        'n_clusters': best_n_clusters,
        'silhouette_score': best_score
    }

def analyze_discovered_clusters(clustered_data: pd.DataFrame, 
                               feature_columns: List[str],
                               scaler) -> Dict:
    """
    Analyze discovered clusters to extract meaningful financial patterns.
    
    This function examines each cluster to understand what financial behaviors
    characterize each group, helping translate clusters into business-meaningful patterns.
    """
    
    patterns = {}
    
    for cluster_id in sorted(clustered_data['cluster'].unique()):
        cluster_data = clustered_data[clustered_data['cluster'] == cluster_id]
        
        # Calculate cluster statistics
        cluster_stats = {}
        for feature in feature_columns:
            if feature in cluster_data.columns:
                values = cluster_data[feature].dropna()
                if len(values) > 0:
                    cluster_stats[feature] = {
                        'mean': values.mean(),
                        'std': values.std(),
                        'median': values.median(),
                        'q25': values.quantile(0.25),
                        'q75': values.quantile(0.75)
                    }
        
        # Identify distinctive characteristics
        distinctive_features = identify_cluster_characteristics(
            cluster_data, clustered_data, feature_columns
        )
        
        # Assess risk level based on financial indicators
        risk_assessment = assess_cluster_risk_level(cluster_stats)
        
        # Generate human-readable description
        description = generate_cluster_description(distinctive_features, risk_assessment)
        
        patterns[f'pattern_{cluster_id}'] = {
            'cluster_id': cluster_id,
            'size': len(cluster_data),
            'companies': cluster_data['company_id'].tolist(),
            'statistics': cluster_stats,
            'distinctive_features': distinctive_features,
            'risk_level': risk_assessment['risk_level'],
            'risk_score': risk_assessment['risk_score'],
            'description': description,
            'key_indicators': risk_assessment['key_indicators']
        }
    
    return patterns

def identify_cluster_characteristics(cluster_data: pd.DataFrame, 
                                   all_data: pd.DataFrame,
                                   feature_columns: List[str]) -> Dict:
    """
    Identify what makes a cluster distinctive compared to the overall population.
    
    This function finds features where the cluster significantly differs from
    the overall average, helping identify the key characteristics of each pattern.
    """
    
    characteristics = {}
    
    for feature in feature_columns:
        if feature in cluster_data.columns and feature in all_data.columns:
            cluster_values = cluster_data[feature].dropna()
            all_values = all_data[feature].dropna()
            
            if len(cluster_values) >= 3 and len(all_values) >= 10:
                cluster_mean = cluster_values.mean()
                all_mean = all_values.mean()
                all_std = all_values.std()
                
                if all_std > 0:
                    # Calculate z-score of cluster mean compared to population
                    z_score = abs(cluster_mean - all_mean) / all_std
                    
                    # Consider it distinctive if z-score > 1.5
                    if z_score > 1.5:
                        direction = 'higher' if cluster_mean > all_mean else 'lower'
                        characteristics[feature] = {
                            'cluster_mean': cluster_mean,
                            'population_mean': all_mean,
                            'z_score': z_score,
                            'direction': direction,
                            'relative_difference': (cluster_mean - all_mean) / all_mean if all_mean != 0 else 0
                        }
    
    return characteristics

def assess_cluster_risk_level(cluster_stats: Dict) -> Dict:
    """
    Assess the risk level of a cluster based on its financial characteristics.
    
    This function analyzes the cluster's financial metrics to determine overall risk level
    and identify the key risk indicators.
    """
    
    risk_indicators = []
    risk_score = 0
    
    # Check utilization-related risks
    if 'util_mean_90d' in cluster_stats:
        util_mean = cluster_stats['util_mean_90d']['mean']
        if util_mean > 0.8:
            risk_score += 3
            risk_indicators.append(f"Very high utilization ({util_mean:.1%})")
        elif util_mean > 0.6:
            risk_score += 2
            risk_indicators.append(f"High utilization ({util_mean:.1%})")
    
    # Check utilization volatility
    if 'util_vol_90d' in cluster_stats:
        util_vol = cluster_stats['util_vol_90d']['mean']
        if util_vol > 0.15:
            risk_score += 2
            risk_indicators.append(f"High utilization volatility ({util_vol:.1%})")
        elif util_vol > 0.08:
            risk_score += 1
            risk_indicators.append(f"Moderate utilization volatility ({util_vol:.1%})")
    
    # Check deposit trends
    if 'deposit_trend_90d' in cluster_stats:
        deposit_trend = cluster_stats['deposit_trend_90d']['mean']
        if deposit_trend < -100:  # Declining deposits
            risk_score += 2
            risk_indicators.append("Declining deposit trend")
        elif deposit_trend < -50:
            risk_score += 1
            risk_indicators.append("Weak deposit trend")
    
    # Check utilization trends
    if 'util_trend_90d' in cluster_stats:
        util_trend = cluster_stats['util_trend_90d']['mean']
        if util_trend > 0.002:  # Increasing utilization
            risk_score += 2
            risk_indicators.append("Rising utilization trend")
        elif util_trend > 0.001:
            risk_score += 1
            risk_indicators.append("Moderately rising utilization")
    
    # Check behavioral patterns
    if 'near_limit_days_30d' in cluster_stats:
        near_limit_days = cluster_stats['near_limit_days_30d']['mean']
        if near_limit_days > 10:
            risk_score += 2
            risk_indicators.append(f"Frequently near credit limit ({near_limit_days:.0f} days)")
        elif near_limit_days > 5:
            risk_score += 1
            risk_indicators.append(f"Occasionally near credit limit ({near_limit_days:.0f} days)")
    
    # Determine overall risk level
    if risk_score >= 6:
        risk_level = 'high'
    elif risk_score >= 4:
        risk_level = 'medium'
    elif risk_score >= 2:
        risk_level = 'low'
    else:
        risk_level = 'minimal'
    
    return {
        'risk_level': risk_level,
        'risk_score': risk_score,
        'key_indicators': risk_indicators
    }

def generate_cluster_description(distinctive_features: Dict, risk_assessment: Dict) -> str:
    """
    Generate a human-readable description of a cluster's characteristics.
    
    This function creates business-friendly descriptions that credit officers can easily understand.
    """
    
    description_parts = []
    
    # Start with risk level
    risk_level = risk_assessment['risk_level']
    description_parts.append(f"Risk Level: {risk_level.upper()}")
    
    # Add key financial characteristics
    util_features = [k for k in distinctive_features.keys() if 'util' in k and 'mean' in k]
    if util_features:
        feature = util_features[0]
        info = distinctive_features[feature]
        direction = info['direction']
        value = info['cluster_mean']
        description_parts.append(f"{direction} than average loan utilization ({value:.1%})")
    
    deposit_features = [k for k in distinctive_features.keys() if 'deposit' in k and 'trend' in k]
    if deposit_features:
        feature = deposit_features[0]
        info = distinctive_features[feature]
        direction = info['direction']
        if 'trend' in feature:
            trend_desc = "growing" if info['cluster_mean'] > 0 else "declining"
            description_parts.append(f"{trend_desc} deposit pattern")
    
    # Add volatility information
    vol_features = [k for k in distinctive_features.keys() if 'vol' in k]
    if vol_features:
        feature = vol_features[0]
        info = distinctive_features[feature]
        direction = info['direction']
        description_parts.append(f"{direction} than average volatility")
    
    # Combine into coherent description
    if len(description_parts) > 1:
        description = description_parts[0] + ". " + ", ".join(description_parts[1:])
    else:
        description = description_parts[0] if description_parts else "Standard financial profile"
    
    return description

def generate_rules_from_patterns(patterns: Dict, feature_columns: List[str]) -> Dict:
    """
    Generate interpretable rules from discovered patterns.
    
    This function translates cluster characteristics into explicit rules that can be
    applied to classify new companies into risk categories.
    """
    
    rules = {}
    
    for pattern_name, pattern_info in patterns.items():
        pattern_rules = []
        
        # Extract key distinctive features for rule generation
        distinctive_features = pattern_info['distinctive_features']
        
        # Generate rules based on most distinctive features
        sorted_features = sorted(
            distinctive_features.items(),
            key=lambda x: x[1]['z_score'],
            reverse=True
        )
        
        # Take top 3-5 most distinctive features
        top_features = sorted_features[:min(5, len(sorted_features))]
        
        for feature_name, feature_info in top_features:
            # Create rule condition based on feature characteristics
            cluster_mean = feature_info['cluster_mean']
            direction = feature_info['direction']
            z_score = feature_info['z_score']
            
            # Generate threshold based on cluster characteristics
            if direction == 'higher':
                # Use cluster mean minus some buffer as threshold
                threshold = cluster_mean * 0.8
                condition = f"{feature_name} > {threshold:.4f}"
            else:
                # Use cluster mean plus some buffer as threshold  
                threshold = cluster_mean * 1.2
                condition = f"{feature_name} < {threshold:.4f}"
            
            rule = {
                'feature': feature_name,
                'condition': condition,
                'threshold': threshold,
                'operator': '>' if direction == 'higher' else '<',
                'importance': z_score,
                'description': generate_rule_description(feature_name, condition, direction)
            }
            
            pattern_rules.append(rule)
        
        rules[pattern_name] = {
            'pattern_id': pattern_info['cluster_id'],
            'risk_level': pattern_info['risk_level'],
            'description': pattern_info['description'],
            'rules': pattern_rules,
            'min_rules_to_match': max(1, len(pattern_rules) // 2)  # Need to match at least half the rules
        }
    
    return rules

def generate_rule_description(feature_name: str, condition: str, direction: str) -> str:
    """
    Generate human-readable descriptions for individual rules.
    
    This function creates business-friendly explanations for each rule condition.
    """
    
    # Map feature names to business descriptions
    feature_descriptions = {
        'util_mean_90d': 'average loan utilization over 90 days',
        'util_mean_30d': 'average loan utilization over 30 days',
        'deposit_trend_90d': 'deposit balance trend over 90 days',
        'util_trend_90d': 'utilization trend over 90 days',
        'util_vol_90d': 'utilization volatility over 90 days',
        'deposit_vol_90d': 'deposit volatility over 90 days',
        'near_limit_days_30d': 'days near credit limit in past 30 days',
        'util_spikes_30d': 'utilization spikes in past 30 days'
    }
    
    base_desc = feature_descriptions.get(feature_name, feature_name)
    direction_desc = "higher than normal" if direction == 'higher' else "lower than normal"
    
    return f"Company has {direction_desc} {base_desc}"

def optimize_rule_thresholds(df_features: pd.DataFrame, 
                           initial_rules: Dict,
                           validation_data: Optional[pd.DataFrame] = None) -> Dict:
    """
    Optimize rule thresholds using statistical validation and business constraints.
    
    This function fine-tunes the automatically generated rules to improve their
    accuracy and business applicability. It uses statistical methods to find
    optimal threshold values.
    
    Parameters:
    -----------
    df_features : pd.DataFrame
        Training data with comprehensive features
    initial_rules : Dict
        Initially discovered rules from pattern analysis
    validation_data : pd.DataFrame, optional
        Separate validation dataset (if not provided, uses train/test split)
        
    Returns:
    --------
    Dict
        Optimized rules with improved thresholds
    """
    
    print("Optimizing rule thresholds using statistical validation...")
    
    optimized_rules = {}
    
    # Prepare data for optimization
    if validation_data is None:
        # Use most recent 30% of data for validation
        latest_data = df_features.groupby('company_id').last().reset_index()
        n_validation = int(len(latest_data) * 0.3)
        validation_companies = latest_data.sample(n=n_validation, random_state=42)['company_id']
        train_data = df_features[~df_features['company_id'].isin(validation_companies)]
        val_data = df_features[df_features['company_id'].isin(validation_companies)]
    else:
        train_data = df_features
        val_data = validation_data
    
    # Get latest records for each company
    train_latest = train_data.groupby('company_id').last().reset_index()
    val_latest = val_data.groupby('company_id').last().reset_index()
    
    for pattern_name, pattern_rules in initial_rules.items():
        print(f"Optimizing rules for {pattern_name}...")
        
        optimized_pattern = pattern_rules.copy()
        optimized_pattern_rules = []
        
        for rule in pattern_rules['rules']:
            feature_name = rule['feature']
            initial_threshold = rule['threshold']
            operator = rule['operator']
            
            if feature_name not in train_latest.columns:
                # Skip rules for missing features
                continue
            
            # Find optimal threshold using grid search
            feature_values = train_latest[feature_name].dropna()
            
            if len(feature_values) < 10:
                # Keep original threshold if insufficient data
                optimized_pattern_rules.append(rule)
                continue
            
            # Define threshold search range around initial value
            if operator == '>':
                search_range = np.linspace(
                    feature_values.quantile(0.1),
                    feature_values.quantile(0.9),
                    20
                )
            else:
                search_range = np.linspace(
                    feature_values.quantile(0.1),
                    feature_values.quantile(0.9),
                    20
                )
            
            best_threshold = initial_threshold
            best_score = 0
            
            for threshold in search_range:
                # Evaluate threshold performance
                score = evaluate_threshold_performance(
                    train_latest, val_latest, feature_name, threshold, operator, pattern_name
                )
                
                if score > best_score:
                    best_score = score
                    best_threshold = threshold
            
            # Update rule with optimized threshold
            optimized_rule = rule.copy()
            optimized_rule['threshold'] = best_threshold
            optimized_rule['condition'] = f"{feature_name} {operator} {best_threshold:.4f}"
            optimized_rule['optimization_score'] = best_score
            
            optimized_pattern_rules.append(optimized_rule)
        
        optimized_pattern['rules'] = optimized_pattern_rules
        optimized_rules[pattern_name] = optimized_pattern
    
    return optimized_rules

def evaluate_threshold_performance(train_data: pd.DataFrame, 
                                 val_data: pd.DataFrame,
                                 feature_name: str, 
                                 threshold: float,
                                 operator: str,
                                 pattern_name: str) -> float:
    """
    Evaluate the performance of a threshold value for rule optimization.
    
    This function calculates how well a threshold separates companies into
    meaningful groups based on their financial characteristics.
    """
    
    if feature_name not in train_data.columns or feature_name not in val_data.columns:
        return 0
    
    # Apply threshold to create binary classification
    if operator == '>':
        train_matches = train_data[feature_name] > threshold
        val_matches = val_data[feature_name] > threshold
    else:
        train_matches = train_data[feature_name] < threshold
        val_matches = val_data[feature_name] < threshold
    
    # Calculate various performance metrics
    train_match_rate = train_matches.mean()
    val_match_rate = val_matches.mean()
    
    # Penalize thresholds that match too many or too few companies
    if train_match_rate < 0.05 or train_match_rate > 0.8:
        return 0
    
    # Reward consistency between train and validation
    consistency_score = 1 - abs(train_match_rate - val_match_rate)
    
    # Calculate separation quality
    train_feature_values = train_data[feature_name].dropna()
    if len(train_feature_values) > 10:
        if operator == '>':
            above_threshold = train_feature_values[train_feature_values > threshold]
            below_threshold = train_feature_values[train_feature_values <= threshold]
        else:
            above_threshold = train_feature_values[train_feature_values >= threshold]
            below_threshold = train_feature_values[train_feature_values < threshold]
        
        if len(above_threshold) > 0 and len(below_threshold) > 0:
            # Calculate separation between groups
            separation_score = abs(above_threshold.mean() - below_threshold.mean()) / train_feature_values.std()
        else:
            separation_score = 0
    else:
        separation_score = 0
    
    # Combine scores
    overall_score = 0.4 * consistency_score + 0.4 * separation_score + 0.2 * (1 - abs(0.2 - train_match_rate))
    
    return max(0, overall_score)

def validate_rules_comprehensive(df_features: pd.DataFrame, 
                               optimized_rules: Dict,
                               validation_split: float = 0.3) -> Dict:
    """
    Perform comprehensive validation of the optimized rules.
    
    This function tests the rules on held-out data to measure their accuracy,
    stability, and business relevance.
    
    Parameters:
    -----------
    df_features : pd.DataFrame
        Full dataset with features
    optimized_rules : Dict
        Rules to validate
    validation_split : float
        Fraction of data to use for validation (default: 0.3)
        
    Returns:
    --------
    Dict
        Comprehensive validation results
    """
    
    print("Performing comprehensive rule validation...")
    
    # Split data into training and validation sets
    latest_data = df_features.groupby('company_id').last().reset_index()
    n_validation = int(len(latest_data) * validation_split)
    
    validation_companies = latest_data.sample(n=n_validation, random_state=42)['company_id']
    train_data = latest_data[~latest_data['company_id'].isin(validation_companies)]
    val_data = latest_data[latest_data['company_id'].isin(validation_companies)]
    
    validation_results = {}
    
    for pattern_name, pattern_rules in optimized_rules.items():
        print(f"Validating {pattern_name}...")
        
        # Apply rules to validation data
        val_predictions = apply_pattern_rules(val_data, pattern_rules)
        train_predictions = apply_pattern_rules(train_data, pattern_rules)
        
        # Calculate performance metrics
        val_match_rate = val_predictions.mean()
        train_match_rate = train_predictions.mean()
        
        # Stability check
        stability_score = 1 - abs(val_match_rate - train_match_rate)
        
        # Coverage analysis
        coverage_score = 1 - abs(0.15 - val_match_rate)  # Target ~15% of companies
        
        # Rule consistency analysis
        rule_consistency = analyze_rule_consistency(val_data, pattern_rules)
        
        # Business logic validation
        business_validation = validate_business_logic(val_data, val_predictions, pattern_rules)
        
        validation_results[pattern_name] = {
            'validation_match_rate': val_match_rate,
            'training_match_rate': train_match_rate,
            'stability_score': stability_score,
            'coverage_score': coverage_score,
            'rule_consistency': rule_consistency,
            'business_validation': business_validation,
            'overall_score': (stability_score + coverage_score + business_validation['score']) / 3,
            'matched_companies': val_data[val_predictions]['company_id'].tolist()
        }
    
    return validation_results

def apply_pattern_rules(data: pd.DataFrame, pattern_rules: Dict) -> pd.Series:
    """
    Apply a pattern's rules to data and return boolean matches.
    
    This function evaluates all rules for a pattern and determines which
    companies match the pattern based on the minimum matching criteria.
    """
    
    rules = pattern_rules['rules']
    min_rules_to_match = pattern_rules.get('min_rules_to_match', 1)
    
    # Track how many rules each company matches
    rule_matches = pd.DataFrame(index=data.index)
    
    for i, rule in enumerate(rules):
        feature_name = rule['feature']
        threshold = rule['threshold']
        operator = rule['operator']
        
        if feature_name in data.columns:
            if operator == '>':
                matches = data[feature_name] > threshold
            else:
                matches = data[feature_name] < threshold
            
            rule_matches[f'rule_{i}'] = matches.fillna(False)
        else:
            rule_matches[f'rule_{i}'] = False
    
    # Count how many rules each company matches
    total_matches = rule_matches.sum(axis=1)
    
    # Return companies that match at least the minimum number of rules
    return total_matches >= min_rules_to_match

def analyze_rule_consistency(data: pd.DataFrame, pattern_rules: Dict) -> Dict:
    """
    Analyze how consistently rules work together within a pattern.
    
    This function checks whether the rules in a pattern are mutually consistent
    and identify companies in a coherent way.
    """
    
    rules = pattern_rules['rules']
    
    # Calculate pairwise correlation between rule matches
    rule_matches = pd.DataFrame(index=data.index)
    
    for i, rule in enumerate(rules):
        feature_name = rule['feature']
        threshold = rule['threshold']
        operator = rule['operator']
        
        if feature_name in data.columns:
            if operator == '>':
                matches = data[feature_name] > threshold
            else:
                matches = data[feature_name] < threshold
            
            rule_matches[f'rule_{i}'] = matches.fillna(False).astype(int)
    
    if len(rule_matches.columns) > 1:
        # Calculate correlation matrix
        correlation_matrix = rule_matches.corr()
        
        # Calculate average correlation (excluding diagonal)
        mask = np.triu(np.ones_like(correlation_matrix, dtype=bool), k=1)
        avg_correlation = correlation_matrix.where(mask).stack().mean()
        
        consistency_score = max(0, avg_correlation)  # Negative correlations indicate inconsistency
    else:
        consistency_score = 1.0  # Single rule is always consistent with itself
    
    return {
        'average_correlation': avg_correlation if len(rule_matches.columns) > 1 else 1.0,
        'consistency_score': consistency_score,
        'n_rules': len(rules)
    }

def validate_business_logic(data: pd.DataFrame, predictions: pd.Series, pattern_rules: Dict) -> Dict:
    """
    Validate that the rules make business sense by checking financial logic.
    
    This function performs sanity checks to ensure the identified patterns
    align with financial common sense and business understanding.
    """
    
    matched_companies = data[predictions]
    
    if len(matched_companies) == 0:
        return {'score': 0, 'issues': ['No companies matched the pattern']}
    
    business_issues = []
    business_score = 1.0
    
    # Check 1: Risk level consistency
    risk_level = pattern_rules.get('risk_level', 'unknown')
    
    if risk_level in ['high', 'medium']:
        # High/medium risk companies should have concerning financial metrics
        if 'util_mean_90d' in matched_companies.columns:
            avg_utilization = matched_companies['util_mean_90d'].mean()
            if avg_utilization < 0.3:
                business_issues.append("High risk pattern but low average utilization")
                business_score -= 0.3
        
        if 'deposit_trend_90d' in matched_companies.columns:
            avg_deposit_trend = matched_companies['deposit_trend_90d'].mean()
            if avg_deposit_trend > 100:  # Strong positive trend
                business_issues.append("High risk pattern but strong deposit growth")
                business_score -= 0.2
    
    elif risk_level == 'minimal':
        # Low risk companies should have stable metrics
        if 'util_vol_90d' in matched_companies.columns:
            avg_volatility = matched_companies['util_vol_90d'].mean()
            if avg_volatility > 0.15:
                business_issues.append("Low risk pattern but high volatility")
                business_score -= 0.2
    
    # Check 2: Logical feature relationships
    if ('util_mean_90d' in matched_companies.columns and 
        'near_limit_days_30d' in matched_companies.columns):
        
        high_util_companies = matched_companies[matched_companies['util_mean_90d'] > 0.8]
        if len(high_util_companies) > 0:
            avg_near_limit_days = high_util_companies['near_limit_days_30d'].mean()
            if avg_near_limit_days < 5:
                business_issues.append("High utilization but few days near limit")
                business_score -= 0.1
    
    # Check 3: Sample size reasonableness
    match_rate = len(matched_companies) / len(data)
    if match_rate > 0.5:
        business_issues.append(f"Pattern matches too many companies ({match_rate:.1%})")
        business_score -= 0.2
    elif match_rate < 0.02:
        business_issues.append(f"Pattern matches too few companies ({match_rate:.1%})")
        business_score -= 0.1
    
    return {
        'score': max(0, business_score),
        'issues': business_issues,
        'matched_companies_count': len(matched_companies),
        'match_rate': match_rate
    }

#######################################################
# SECTION 2: RULE APPLICATION FOR PERSONA TAGGING
#######################################################

def load_and_prepare_client_data(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Load and prepare client data for persona tagging.
    
    This function takes raw banking data and prepares it for rule application
    by calculating all necessary features and ensuring data quality.
    
    Parameters:
    -----------
    df_raw : pd.DataFrame
        Raw banking data with columns: company_id, date, deposit_balance, used_loan, unused_loan
        
    Returns:
    --------
    pd.DataFrame
        Prepared data ready for persona tagging
    """
    
    print("Loading and preparing client data for persona tagging...")
    
    # Basic data validation
    required_columns = ['company_id', 'date', 'deposit_balance', 'used_loan', 'unused_loan']
    missing_columns = [col for col in required_columns if col not in df_raw.columns]
    
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")
    
    # Clean and standardize data
    df_clean = df_raw.copy()
    df_clean['date'] = pd.to_datetime(df_clean['date'])
    df_clean = df_clean.sort_values(['company_id', 'date'])
    
    # Remove companies with insufficient data
    company_data_counts = df_clean['company_id'].value_counts()
    valid_companies = company_data_counts[company_data_counts >= 30].index  # At least 30 days
    df_clean = df_clean[df_clean['company_id'].isin(valid_companies)]
    
    print(f"Retained {len(valid_companies)} companies with sufficient data")
    
    # Calculate comprehensive features using the same function as training
    df_prepared = calculate_comprehensive_features(df_clean)
    
    print(f"Prepared data for {df_prepared['company_id'].nunique()} companies")
    
    return df_prepared

def apply_risk_rules_to_clients(df_prepared: pd.DataFrame, 
                               validated_rules: Dict,
                               rule_weights: Optional[Dict] = None) -> pd.DataFrame:
    """
    Apply validated risk rules to client data for persona tagging.
    
    This function takes the optimized and validated rules and applies them
    to client data to assign risk personas with confidence scores.
    
    Parameters:
    -----------
    df_prepared : pd.DataFrame
        Prepared client data with all necessary features
    validated_rules : Dict
        Validated rules from the rule generation process
    rule_weights : Dict, optional
        Custom weights for different rules (default: equal weights)
        
    Returns:
    --------
    pd.DataFrame
        Client data with persona assignments and confidence scores
    """
    
    print("Applying risk rules to assign client personas...")
    
    # Get the most recent data for each company (cross-sectional analysis)
    latest_client_data = df_prepared.groupby('company_id').last().reset_index()
    
    # Initialize results
    results = []
    
    for _, client_row in tqdm(latest_client_data.iterrows(), total=len(latest_client_data), 
                             desc="Tagging client personas"):
        
        company_id = client_row['company_id']
        
        # Test each pattern against this client
        pattern_matches = {}
        pattern_scores = {}
        
        for pattern_name, pattern_rules in validated_rules.items():
            # Apply pattern rules
            match_result = evaluate_client_against_pattern(client_row, pattern_rules)
            
            pattern_matches[pattern_name] = match_result['matches']
            pattern_scores[pattern_name] = match_result['confidence_score']
        
        # Determine best matching persona
        persona_assignment = determine_best_persona(pattern_matches, pattern_scores, validated_rules)
        
        # Add detailed risk analysis
        risk_analysis = analyze_client_risk_details(client_row, persona_assignment, validated_rules)
        
        results.append({
            'company_id': company_id,
            'assigned_persona': persona_assignment['persona'],
            'confidence_score': persona_assignment['confidence'],
            'risk_level': persona_assignment['risk_level'],
            'matching_patterns': persona_assignment['matching_patterns'],
            'risk_indicators': risk_analysis['indicators'],
            'risk_score': risk_analysis['score'],
            'explanation': risk_analysis['explanation'],
            'assignment_date': datetime.now(),
            'pattern_scores': pattern_scores
        })
    
    # Convert to DataFrame
    results_df = pd.DataFrame(results)
    
    # Add summary statistics
    print(f"\nPersona Assignment Summary:")
    print(f"Total companies analyzed: {len(results_df)}")
    print(f"Persona distribution:")
    for persona, count in results_df['assigned_persona'].value_counts().items():
        print(f"  {persona}: {count} companies ({count/len(results_df)*100:.1f}%)")
    
    print(f"Risk level distribution:")
    for risk_level, count in results_df['risk_level'].value_counts().items():
        print(f"  {risk_level}: {count} companies ({count/len(results_df)*100:.1f}%)")
    
    return results_df

def evaluate_client_against_pattern(client_data: pd.Series, pattern_rules: Dict) -> Dict:
    """
    Evaluate a single client against a specific pattern's rules.
    
    This function checks how well a client matches a particular risk pattern
    and calculates a confidence score for the match.
    """
    
    rules = pattern_rules['rules']
    min_rules_to_match = pattern_rules.get('min_rules_to_match', 1)
    
    matched_rules = []
    rule_scores = []
    
    for rule in rules:
        feature_name = rule['feature']
        threshold = rule['threshold']
        operator = rule['operator']
        importance = rule.get('importance', 1.0)
        
        if feature_name in client_data.index and pd.notna(client_data[feature_name]):
            client_value = client_data[feature_name]
            
            # Check if rule condition is met
            if operator == '>':
                rule_match = client_value > threshold
                # Calculate how far beyond threshold (for confidence)
                if rule_match:
                    excess = (client_value - threshold) / threshold if threshold != 0 else 1
                    confidence_contribution = min(1.0, 0.5 + 0.5 * excess)
                else:
                    confidence_contribution = max(0.0, client_value / threshold) if threshold > 0 else 0
            else:  # operator == '<'
                rule_match = client_value < threshold
                # Calculate how far below threshold (for confidence)
                if rule_match:
                    deficit = (threshold - client_value) / threshold if threshold != 0 else 1
                    confidence_contribution = min(1.0, 0.5 + 0.5 * deficit)
                else:
                    confidence_contribution = max(0.0, threshold / client_value) if client_value > 0 else 0
            
            if rule_match:
                matched_rules.append(rule)
            
            # Weight by rule importance
            weighted_score = confidence_contribution * importance
            rule_scores.append(weighted_score)
        else:
            # Missing feature gets neutral score
            rule_scores.append(0.5)
    
    # Determine if pattern matches
    pattern_matches = len(matched_rules) >= min_rules_to_match
    
    # Calculate overall confidence score
    if len(rule_scores) > 0:
        # Weighted average of rule scores
        total_importance = sum([rule.get('importance', 1.0) for rule in rules])
        confidence_score = sum(rule_scores) / len(rule_scores)
        
        # Boost confidence if more rules are matched
        match_bonus = len(matched_rules) / len(rules) * 0.2
        confidence_score = min(1.0, confidence_score + match_bonus)
    else:
        confidence_score = 0.0
    
    return {
        'matches': pattern_matches,
        'confidence_score': confidence_score,
        'matched_rules': len(matched_rules),
        'total_rules': len(rules),
        'rule_details': matched_rules
    }

def determine_best_persona(pattern_matches: Dict, pattern_scores: Dict, validated_rules: Dict) -> Dict:
    """
    Determine the best persona assignment based on pattern matching results.
    
    This function analyzes all pattern matches and scores to select the most
    appropriate persona for a client.
    """
    
    # Find patterns that match
    matching_patterns = [pattern for pattern, matches in pattern_matches.items() if matches]
    
    if not matching_patterns:
        # No patterns match - assign default low-risk persona
        return {
            'persona': 'stable_client',
            'confidence': 0.3,
            'risk_level': 'minimal',
            'matching_patterns': [],
            'explanation': 'No significant risk patterns detected'
        }
    
    # If multiple patterns match, choose the one with highest confidence and risk level
    pattern_priorities = []
    
    for pattern in matching_patterns:
        confidence = pattern_scores[pattern]
        risk_level = validated_rules[pattern]['risk_level']
        
        # Convert risk level to numeric priority
        risk_priority = {'high': 4, 'medium': 3, 'low': 2, 'minimal': 1}.get(risk_level, 1)
        
        # Combined priority score
        priority_score = confidence * 0.6 + (risk_priority / 4) * 0.4
        
        pattern_priorities.append({
            'pattern': pattern,
            'confidence': confidence,
            'risk_level': risk_level,
            'priority_score': priority_score
        })
    
    # Sort by priority score
    pattern_priorities.sort(key=lambda x: x['priority_score'], reverse=True)
    
    # Select best pattern
    best_pattern = pattern_priorities[0]
    
    # Generate persona name from pattern
    persona_name = generate_persona_name_from_pattern(best_pattern['pattern'], validated_rules)
    
    return {
        'persona': persona_name,
        'confidence': best_pattern['confidence'],
        'risk_level': best_pattern['risk_level'],
        'matching_patterns': matching_patterns,
        'explanation': f"Best match: {best_pattern['pattern']} (confidence: {best_pattern['confidence']:.2f})"
    }

def generate_persona_name_from_pattern(pattern_name: str, validated_rules: Dict) -> str:
    """
    Generate a business-friendly persona name from a pattern identifier.
    
    This function creates meaningful persona names that credit officers can easily understand.
    """
    
    pattern_info = validated_rules[pattern_name]
    risk_level = pattern_info['risk_level']
    description = pattern_info.get('description', '')
    
    # Extract key characteristics from description
    if 'high' in description.lower() and 'utilization' in description.lower():
        if 'volatility' in description.lower():
            return 'volatile_high_utilizer'
        else:
            return 'consistent_high_utilizer'
    
    elif 'declining' in description.lower() and 'deposit' in description.lower():
        if risk_level == 'high':
            return 'deteriorating_client'
        else:
            return 'deposit_declining_client'
    
    elif 'trend' in description.lower() or 'rising' in description.lower():
        return 'expanding_credit_user'
    
    elif 'volatility' in description.lower() or 'volatile' in description.lower():
        return 'erratic_financial_behavior'
    
    elif 'seasonal' in description.lower():
        return 'seasonal_business_client'
    
    else:
        # Default naming based on risk level
        risk_names = {
            'high': 'high_risk_client',
            'medium': 'moderate_risk_client', 
            'low': 'low_risk_client',
            'minimal': 'stable_client'
        }
        return risk_names.get(risk_level, 'unclassified_client')

def analyze_client_risk_details(client_data: pd.Series, persona_assignment: Dict, validated_rules: Dict) -> Dict:
    """
    Provide detailed risk analysis for a client based on their persona assignment.
    
    This function creates comprehensive risk explanations that help credit officers
    understand why a client was assigned to a particular risk category.
    """
    
    risk_indicators = []
    risk_score = 0
    
    # Extract key financial metrics
    current_util = client_data.get('util_mean_30d', client_data.get('loan_utilization', 0))
    util_trend = client_data.get('util_trend_90d', 0)
    deposit_trend = client_data.get('deposit_trend_90d', 0)
    util_volatility = client_data.get('util_vol_90d', 0)
    
    # Utilization analysis
    if current_util > 0.8:
        risk_score += 3
        risk_indicators.append(f"Very high loan utilization ({current_util:.1%})")
    elif current_util > 0.6:
        risk_score += 2
        risk_indicators.append(f"High loan utilization ({current_util:.1%})")
    elif current_util > 0.4:
        risk_score += 1
        risk_indicators.append(f"Moderate loan utilization ({current_util:.1%})")
    
    # Trend analysis
    if util_trend > 0.002:
        risk_score += 2
        risk_indicators.append("Rising utilization trend")
    elif util_trend > 0.001:
        risk_score += 1
        risk_indicators.append("Slowly rising utilization")
    
    if deposit_trend < -100:
        risk_score += 2
        risk_indicators.append("Declining deposit trend")
    elif deposit_trend < -50:
        risk_score += 1
        risk_indicators.append("Weak deposit growth")
    
    # Volatility analysis
    if util_volatility > 0.15:
        risk_score += 2
        risk_indicators.append("High utilization volatility")
    elif util_volatility > 0.08:
        risk_score += 1
        risk_indicators.append("Moderate utilization volatility")
    
    # Additional behavioral indicators
    near_limit_days = client_data.get('near_limit_days_30d', 0)
    if near_limit_days > 10:
        risk_score += 2
        risk_indicators.append(f"Frequently near credit limit ({near_limit_days:.0f} days)")
    elif near_limit_days > 5:
        risk_score += 1
        risk_indicators.append(f"Occasionally near credit limit ({near_limit_days:.0f} days)")
    
    # Generate explanation
    persona = persona_assignment['persona']
    confidence = persona_assignment['confidence']
    
    explanation_parts = [
        f"Client assigned to '{persona}' persona with {confidence:.1%} confidence."
    ]
    
    if risk_indicators:
        explanation_parts.append(f"Key risk factors: {'; '.join(risk_indicators[:3])}")
    else:
        explanation_parts.append("No significant risk factors identified.")
    
    if persona_assignment['matching_patterns']:
        pattern_names = [p.replace('_', ' ').title() for p in persona_assignment['matching_patterns']]
        explanation_parts.append(f"Matches patterns: {', '.join(pattern_names)}")
    
    explanation = " ".join(explanation_parts)
    
    return {
        'indicators': risk_indicators,
        'score': risk_score,
        'explanation': explanation
    }

def generate_persona_explanation_report(results_df: pd.DataFrame, 
                                      validated_rules: Dict,
                                      company_id: Optional[str] = None) -> str:
    """
    Generate detailed explanation reports for persona assignments.
    
    This function creates comprehensive reports that explain persona assignments
    in business-friendly language for credit officers.
    
    Parameters:
    -----------
    results_df : pd.DataFrame
        Results from persona tagging
    validated_rules : Dict
        The rules used for tagging
    company_id : str, optional
        Specific company to report on (if None, generates summary report)
        
    Returns:
    --------
    str
        Formatted explanation report
    """
    
    if company_id:
        # Generate individual company report
        return generate_individual_company_report(results_df, validated_rules, company_id)
    else:
        # Generate summary report
        return generate_summary_report(results_df, validated_rules)

def generate_individual_company_report(results_df: pd.DataFrame, 
                                     validated_rules: Dict,
                                     company_id: str) -> str:
    """Generate detailed report for a specific company."""
    
    company_result = results_df[results_df['company_id'] == company_id]
    
    if company_result.empty:
        return f"No data found for company {company_id}"
    
    result = company_result.iloc[0]
    
    report = []
    report.append("=" * 60)
    report.append(f"RISK PERSONA ANALYSIS REPORT")
    report.append(f"Company ID: {company_id}")
    report.append(f"Analysis Date: {result['assignment_date'].strftime('%Y-%m-%d %H:%M:%S')}")
    report.append("=" * 60)
    report.append("")
    
    # Executive Summary
    report.append("EXECUTIVE SUMMARY")
    report.append("-" * 20)
    report.append(f"Assigned Persona: {result['assigned_persona'].replace('_', ' ').title()}")
    report.append(f"Risk Level: {result['risk_level'].upper()}")
    report.append(f"Confidence Score: {result['confidence_score']:.1%}")
    report.append(f"Overall Risk Score: {result['risk_score']}/10")
    report.append("")
    
    # Detailed Explanation
    report.append("DETAILED ANALYSIS")
    report.append("-" * 20)
    report.append(result['explanation'])
    report.append("")
    
    # Risk Indicators
    if result['risk_indicators']:
        report.append("KEY RISK INDICATORS")
        report.append("-" * 20)
        for i, indicator in enumerate(result['risk_indicators'][:5], 1):
            report.append(f"{i}. {indicator}")
        report.append("")
    
    # Pattern Matches
    if result['matching_patterns']:
        report.append("MATCHING RISK PATTERNS")
        report.append("-" * 20)
        for pattern in result['matching_patterns']:
            pattern_info = validated_rules.get(pattern, {})
            pattern_desc = pattern_info.get('description', 'No description available')
            pattern_score = result['pattern_scores'].get(pattern, 0)
            report.append(f"• {pattern.replace('_', ' ').title()}")
            report.append(f"  Description: {pattern_desc}")
            report.append(f"  Match Score: {pattern_score:.2f}")
            report.append("")
    
    # Recommendations
    report.append("RECOMMENDATIONS")
    report.append("-" * 20)
    risk_level = result['risk_level']
    
    if risk_level == 'high':
        report.append("• IMMEDIATE ACTION REQUIRED")
        report.append("• Schedule urgent review with relationship manager")
        report.append("• Consider credit limit review or additional monitoring")
        report.append("• Implement enhanced due diligence procedures")
    elif risk_level == 'medium':
        report.append("• Enhanced monitoring recommended")
        report.append("• Schedule review within 30 days")
        report.append("• Consider requesting updated financial statements")
        report.append("• Monitor for any deterioration in key metrics")
    elif risk_level == 'low':
        report.append("• Standard monitoring sufficient")
        report.append("• Continue regular account reviews")
        report.append("• Watch for any significant changes in behavior")
    else:
        report.append("• Minimal risk detected")
        report.append("• Standard account management appropriate")
        report.append("• Consider for preferred customer programs")
    
    report.append("")
    report.append("=" * 60)
    
    return "\n".join(report)

def generate_summary_report(results_df: pd.DataFrame, validated_rules: Dict) -> str:
    """Generate summary report across all companies."""
    
    report = []
    report.append("=" * 70)
    report.append("PORTFOLIO RISK PERSONA ANALYSIS SUMMARY")
    report.append(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append(f"Total Companies Analyzed: {len(results_df)}")
    report.append("=" * 70)
    report.append("")
    
    # Overall Risk Distribution
    report.append("RISK LEVEL DISTRIBUTION")
    report.append("-" * 30)
    risk_counts = results_df['risk_level'].value_counts()
    for risk_level in ['high', 'medium', 'low', 'minimal']:
        count = risk_counts.get(risk_level, 0)
        percentage = (count / len(results_df)) * 100
        report.append(f"{risk_level.upper():<10}: {count:>4} companies ({percentage:>5.1f}%)")
    report.append("")
    
    # Persona Distribution
    report.append("PERSONA DISTRIBUTION")
    report.append("-" * 25)
    persona_counts = results_df['assigned_persona'].value_counts()
    for persona, count in persona_counts.head(10).items():
        percentage = (count / len(results_df)) * 100
        clean_persona = persona.replace('_', ' ').title()
        report.append(f"{clean_persona:<25}: {count:>4} ({percentage:>5.1f}%)")
    report.append("")
    
    # High Risk Companies
    high_risk_companies = results_df[results_df['risk_level'] == 'high']
    if not high_risk_companies.empty:
        report.append("HIGH RISK COMPANIES REQUIRING IMMEDIATE ATTENTION")
        report.append("-" * 50)
        for _, company in high_risk_companies.head(10).iterrows():
            report.append(f"• {company['company_id']}: {company['assigned_persona'].replace('_', ' ').title()}")
            report.append(f"  Risk Score: {company['risk_score']}/10, Confidence: {company['confidence_score']:.1%}")
            if company['risk_indicators']:
                report.append(f"  Key Issues: {'; '.join(company['risk_indicators'][:2])}")
            report.append("")
    
    # Confidence Analysis
    report.append("ASSIGNMENT CONFIDENCE ANALYSIS")
    report.append("-" * 35)
    high_confidence = (results_df['confidence_score'] >= 0.7).sum()
    medium_confidence = ((results_df['confidence_score'] >= 0.5) & 
                        (results_df['confidence_score'] < 0.7)).sum()
    low_confidence = (results_df['confidence_score'] < 0.5).sum()
    
    report.append(f"High Confidence (≥70%): {high_confidence} companies ({high_confidence/len(results_df)*100:.1f}%)")
    report.append(f"Medium Confidence (50-70%): {medium_confidence} companies ({medium_confidence/len(results_df)*100:.1f}%)")
    report.append(f"Low Confidence (<50%): {low_confidence} companies ({low_confidence/len(results_df)*100:.1f}%)")
    report.append("")
    
    # Recommendations
    report.append("PORTFOLIO MANAGEMENT RECOMMENDATIONS")
    report.append("-" * 40)
    
    high_risk_count = len(high_risk_companies)
    medium_risk_count = len(results_df[results_df['risk_level'] == 'medium'])
    
    if high_risk_count > 0:
        report.append(f"• URGENT: Review {high_risk_count} high-risk companies immediately")
    
    if medium_risk_count > 0:
        report.append(f"• Schedule enhanced monitoring for {medium_risk_count} medium-risk companies")
    
    if low_confidence > len(results_df) * 0.2:
        report.append(f"• Consider additional data collection for {low_confidence} low-confidence assignments")
    
    avg_risk_score = results_df['risk_score'].mean()
    if avg_risk_score > 3:
        report.append("• Portfolio shows elevated risk levels - consider tightening credit policies")
    elif avg_risk_score < 1.5:
        report.append("• Portfolio shows low risk levels - potential for credit expansion")
    
    report.append("")
    report.append("=" * 70)
    
    return "\n".join(report)

def create_rule_adjustment_interface(validated_rules: Dict) -> Dict:
    """
    Create an interface for credit officers to adjust rule thresholds.
    
    This function provides a structured way for credit officers to view and
    modify rule parameters while understanding the implications of their changes.
    
    Parameters:
    -----------
    validated_rules : Dict
        Current validated rules
        
    Returns:
    --------
    Dict
        Interface structure for rule adjustments
    """
    
    adjustment_interface = {
        'current_rules': {},
        'adjustment_options': {},
        'impact_estimates': {}
    }
    
    for pattern_name, pattern_info in validated_rules.items():
        current_rules = []
        adjustment_options = []
        
        for rule in pattern_info['rules']:
            feature_name = rule['feature']
            current_threshold = rule['threshold']
            operator = rule['operator']
            description = rule.get('description', '')
            
            # Create adjustment interface for this rule
            rule_interface = {
                'feature': feature_name,
                'current_threshold': current_threshold,
                'operator': operator,
                'description': description,
                'business_meaning': translate_feature_to_business(feature_name),
                'adjustment_range': calculate_reasonable_adjustment_range(rule),
                'sensitivity_estimate': estimate_threshold_sensitivity(rule)
            }
            
            current_rules.append(rule_interface)
            
            # Create adjustment options
            adjustment_options.append({
                'rule_id': f"{pattern_name}_{feature_name}",
                'current_value': current_threshold,
                'suggested_ranges': {
                    'conservative': current_threshold * 0.9,
                    'moderate': current_threshold,
                    'aggressive': current_threshold * 1.1
                },
                'impact_description': generate_adjustment_impact_description(rule, operator)
            })
        
        adjustment_interface['current_rules'][pattern_name] = {
            'pattern_description': pattern_info.get('description', ''),
            'risk_level': pattern_info.get('risk_level', ''),
            'rules': current_rules
        }
        
        adjustment_interface['adjustment_options'][pattern_name] = adjustment_options
    
    return adjustment_interface

def translate_feature_to_business(feature_name: str) -> str:
    """Translate technical feature names to business-friendly descriptions."""
    
    translations = {
        'util_mean_90d': 'Average loan utilization over past 90 days',
        'util_mean_30d': 'Average loan utilization over past 30 days',
        'deposit_trend_90d': 'Deposit balance trend over past 90 days',
        'util_trend_90d': 'Loan utilization trend over past 90 days',
        'util_vol_90d': 'Loan utilization volatility over past 90 days',
        'deposit_vol_90d': 'Deposit balance volatility over past 90 days',
        'near_limit_days_30d': 'Days near credit limit in past 30 days',
        'util_spikes_30d': 'Number of utilization spikes in past 30 days',
        'large_withdrawals_30d': 'Number of large withdrawals in past 30 days'
    }
    
    return translations.get(feature_name, feature_name.replace('_', ' ').title())

def calculate_reasonable_adjustment_range(rule: Dict) -> Dict:
    """Calculate reasonable ranges for threshold adjustments."""
    
    current_threshold = rule['threshold']
    feature_name = rule['feature']
    
    # Define adjustment ranges based on feature type
    if 'util' in feature_name and 'mean' in feature_name:
        # Utilization rates should stay between 0 and 1
        min_threshold = max(0.05, current_threshold * 0.7)
        max_threshold = min(0.95, current_threshold * 1.3)
    elif 'trend' in feature_name:
        # Trends can vary more widely
        min_threshold = current_threshold * 0.5
        max_threshold = current_threshold * 2.0
    elif 'vol' in feature_name:
        # Volatility measures
        min_threshold = max(0.01, current_threshold * 0.6)
        max_threshold = min(0.5, current_threshold * 1.5)
    elif 'days' in feature_name:
        # Day counts
        min_threshold = max(1, current_threshold * 0.5)
        max_threshold = min(30, current_threshold * 2.0)
    else:
        # Default range
        min_threshold = current_threshold * 0.8
        max_threshold = current_threshold * 1.2
    
    return {
        'minimum': min_threshold,
        'maximum': max_threshold,
        'suggested_increment': (max_threshold - min_threshold) / 10
    }

def estimate_threshold_sensitivity(rule: Dict) -> str:
    """Estimate how sensitive the rule is to threshold changes."""
    
    importance = rule.get('importance', 1.0)
    
    if importance > 2.0:
        return "HIGH - Small changes will significantly affect results"
    elif importance > 1.5:
        return "MEDIUM - Moderate impact from threshold changes"
    else:
        return "LOW - Threshold changes will have limited impact"

def generate_adjustment_impact_description(rule: Dict, operator: str) -> str:
    """Generate description of what happens when thresholds are adjusted."""
    
    feature_name = rule['feature']
    business_meaning = translate_feature_to_business(feature_name)
    
    if operator == '>':
        return f"Increasing threshold will reduce sensitivity - fewer companies flagged for high {business_meaning.lower()}"
    else:
        return f"Decreasing threshold will reduce sensitivity - fewer companies flagged for low {business_meaning.lower()}"

def run_complete_risk_analysis_pipeline(df_raw: pd.DataFrame, 
                                      save_results: bool = True,
                                      output_dir: str = "risk_analysis_output") -> Dict:
    """
    Run the complete risk analysis pipeline from data generation to persona tagging.
    
    This is the main function that orchestrates the entire process:
    1. Generate comprehensive features
    2. Discover risk patterns
    3. Optimize and validate rules
    4. Apply rules to tag client personas
    5. Generate reports and explanations
    
    Parameters:
    -----------
    df_raw : pd.DataFrame
        Raw banking data
    save_results : bool
        Whether to save intermediate and final results (default: True)
    output_dir : str
        Directory to save outputs (default: "risk_analysis_output")
        
    Returns:
    --------
    Dict
        Complete analysis results including rules, assignments, and reports
    """
    
    print("="*70)
    print("INTELLIGENT CREDIT RISK ANALYSIS PIPELINE")
    print("="*70)
    
    import os
    if save_results:
        os.makedirs(output_dir, exist_ok=True)
    
    # Stage 1: Feature Engineering
    print("\n🔄 STAGE 1: COMPREHENSIVE FEATURE ENGINEERING")
    df_features = calculate_comprehensive_features(df_raw)
    
    if save_results:
        df_features.to_csv(f"{output_dir}/enhanced_features.csv", index=False)
        print(f"💾 Saved enhanced features to {output_dir}/enhanced_features.csv")
    
    # Stage 2: Pattern Discovery
    print("\n🔍 STAGE 2: RISK PATTERN DISCOVERY")
    discovery_results = discover_risk_patterns_unsupervised(df_features)
    
    if save_results:
        with open(f"{output_dir}/discovered_patterns.json", 'w') as f:
            # Convert non-serializable objects to strings
            serializable_results = convert_to_serializable(discovery_results)
            json.dump(serializable_results, f, indent=2)
        print(f"💾 Saved discovered patterns to {output_dir}/discovered_patterns.json")
    
    # Stage 3: Rule Optimization
    print("\n⚙️ STAGE 3: RULE OPTIMIZATION AND VALIDATION")
    initial_rules = discovery_results.get('rules', {})
    
    if initial_rules:
        optimized_rules = optimize_rule_thresholds(df_features, initial_rules)
        validation_results = validate_rules_comprehensive(df_features, optimized_rules)
        
        if save_results:
            with open(f"{output_dir}/optimized_rules.json", 'w') as f:
                json.dump(convert_to_serializable(optimized_rules), f, indent=2)
            with open(f"{output_dir}/validation_results.json", 'w') as f:
                json.dump(convert_to_serializable(validation_results), f, indent=2)
            print(f"💾 Saved optimized rules and validation to {output_dir}/")
    else:
        print("⚠️ No rules discovered - using default patterns")
        optimized_rules, validation_results = create_default_rules(), {}
    
    # Stage 4: Client Persona Tagging
    print("\n🏷️ STAGE 4: CLIENT PERSONA TAGGING")
    df_prepared = load_and_prepare_client_data(df_raw)
    persona_results = apply_risk_rules_to_clients(df_prepared, optimized_rules)
    
    if save_results:
        persona_results.to_csv(f"{output_dir}/persona_assignments.csv", index=False)
        print(f"💾 Saved persona assignments to {output_dir}/persona_assignments.csv")
    
    # Stage 5: Report Generation
    print("\n📊 STAGE 5: REPORT GENERATION")
    summary_report = generate_persona_explanation_report(persona_results, optimized_rules)
    
    if save_results:
        with open(f"{output_dir}/summary_report.txt", 'w') as f:
            f.write(summary_report)
        print(f"💾 Saved summary report to {output_dir}/summary_report.txt")
    
    # Create rule adjustment interface
    adjustment_interface = create_rule_adjustment_interface(optimized_rules)
    
    if save_results:
        with open(f"{output_dir}/rule_adjustment_interface.json", 'w') as f:
            json.dump(convert_to_serializable(adjustment_interface), f, indent=2)
        print(f"💾 Saved rule adjustment interface to {output_dir}/rule_adjustment_interface.json")
    
    # Compile final results
    final_results = {
        'enhanced_features': df_features,
        'discovered_patterns': discovery_results,
        'optimized_rules': optimized_rules,
        'validation_results': validation_results,
        'persona_assignments': persona_results,
        'summary_report': summary_report,
        'adjustment_interface': adjustment_interface,
        'pipeline_metadata': {
            'execution_date': datetime.now(),
            'total_companies': df_features['company_id'].nunique(),
            'total_records': len(df_features),
            'patterns_discovered': len(discovery_results.get('patterns', {})),
            'rules_generated': len(optimized_rules),
            'high_risk_companies': len(persona_results[persona_results['risk_level'] == 'high']),
            'pipeline_version': '1.0'
        }
    }
    
    print("\n✅ PIPELINE COMPLETE!")
    print(f"📈 Analyzed {final_results['pipeline_metadata']['total_companies']} companies")
    print(f"🎯 Discovered {final_results['pipeline_metadata']['patterns_discovered']} risk patterns")
    print(f"📋 Generated {final_results['pipeline_metadata']['rules_generated']} rules")
    print(f"⚠️ Identified {final_results['pipeline_metadata']['high_risk_companies']} high-risk companies")
    
    if save_results:
        print(f"\n📁 All results saved to: {output_dir}/")
        print("📄 Key files generated:")
        print("  • enhanced_features.csv - Complete feature dataset")
        print("  • optimized_rules.json - Validated risk rules")
        print("  • persona_assignments.csv - Client persona tags")
        print("  • summary_report.txt - Executive summary")
        print("  • rule_adjustment_interface.json - Rule tuning interface")
    
    return final_results

def convert_to_serializable(obj) -> Union[Dict, List, str, int, float, bool, None]:
    """
    Convert complex objects to JSON-serializable format.
    
    This function handles numpy arrays, pandas objects, and other non-serializable
    types commonly found in data science workflows.
    """
    if isinstance(obj, dict):
        return {key: convert_to_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(item) for item in obj]
    elif isinstance(obj, (np.ndarray, pd.Series)):
        return obj.tolist()
    elif isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    elif isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    elif hasattr(obj, '__dict__'):
        return str(obj)  # Convert complex objects to string representation
    else:
        return obj

def create_default_rules() -> Dict:
    """
    Create default risk rules when pattern discovery doesn't work.
    
    This function provides a fallback set of sensible risk rules based on
    common banking risk indicators.
    """
    
    default_rules = {
        'high_utilization_pattern': {
            'pattern_id': 'default_high_util',
            'risk_level': 'high',
            'description': 'Companies with consistently high loan utilization',
            'rules': [
                {
                    'feature': 'util_mean_90d',
                    'condition': 'util_mean_90d > 0.8',
                    'threshold': 0.8,
                    'operator': '>',
                    'importance': 2.5,
                    'description': 'Average utilization over 90 days exceeds 80%'
                }
            ],
            'min_rules_to_match': 1
        },
        'deteriorating_pattern': {
            'pattern_id': 'default_deteriorating',
            'risk_level': 'high',
            'description': 'Companies with rising utilization and declining deposits',
            'rules': [
                {
                    'feature': 'util_trend_90d',
                    'condition': 'util_trend_90d > 0.002',
                    'threshold': 0.002,
                    'operator': '>',
                    'importance': 2.0,
                    'description': 'Rising utilization trend over 90 days'
                },
                {
                    'feature': 'deposit_trend_90d',
                    'condition': 'deposit_trend_90d < -100',
                    'threshold': -100,
                    'operator': '<',
                    'importance': 2.0,
                    'description': 'Declining deposit trend over 90 days'
                }
            ],
            'min_rules_to_match': 1
        },
        'volatile_pattern': {
            'pattern_id': 'default_volatile',
            'risk_level': 'medium',
            'description': 'Companies with high financial volatility',
            'rules': [
                {
                    'feature': 'util_vol_90d',
                    'condition': 'util_vol_90d > 0.15',
                    'threshold': 0.15,
                    'operator': '>',
                    'importance': 1.5,
                    'description': 'High utilization volatility over 90 days'
                }
            ],
            'min_rules_to_match': 1
        },
        'stable_pattern': {
            'pattern_id': 'default_stable',
            'risk_level': 'minimal',
            'description': 'Companies with stable, low-risk financial behavior',
            'rules': [
                {
                    'feature': 'util_mean_90d',
                    'condition': 'util_mean_90d < 0.4',
                    'threshold': 0.4,
                    'operator': '<',
                    'importance': 1.0,
                    'description': 'Low average utilization over 90 days'
                },
                {
                    'feature': 'util_vol_90d',
                    'condition': 'util_vol_90d < 0.08',
                    'threshold': 0.08,
                    'operator': '<',
                    'importance': 1.0,
                    'description': 'Low utilization volatility over 90 days'
                }
            ],
            'min_rules_to_match': 2
        }
    }
    
    return default_rules

def adjust_rule_threshold(rules: Dict, pattern_name: str, feature_name: str, 
                         new_threshold: float) -> Dict:
    """
    Adjust a specific rule threshold and return updated rules.
    
    This function allows credit officers to fine-tune rule thresholds based on
    their business judgment and risk appetite.
    
    Parameters:
    -----------
    rules : Dict
        Current rule set
    pattern_name : str
        Name of the pattern to modify
    feature_name : str
        Name of the feature rule to modify
    new_threshold : float
        New threshold value
        
    Returns:
    --------
    Dict
        Updated rules with the new threshold
    """
    
    updated_rules = rules.copy()
    
    if pattern_name in updated_rules:
        pattern_rules = updated_rules[pattern_name]['rules']
        
        for i, rule in enumerate(pattern_rules):
            if rule['feature'] == feature_name:
                # Update threshold and condition
                updated_rules[pattern_name]['rules'][i]['threshold'] = new_threshold
                operator = rule['operator']
                updated_rules[pattern_name]['rules'][i]['condition'] = f"{feature_name} {operator} {new_threshold:.4f}"
                
                print(f"✅ Updated {pattern_name} - {feature_name} threshold to {new_threshold:.4f}")
                return updated_rules
    
    print(f"❌ Could not find rule {pattern_name} - {feature_name}")
    return rules

def create_monitoring_dashboard_data(persona_results: pd.DataFrame, 
                                   rules: Dict,
                                   time_period_days: int = 30) -> Dict:
    """
    Create data structure for a monitoring dashboard.
    
    This function prepares summary statistics and trend data that can be used
    to create visual dashboards for ongoing risk monitoring.
    
    Parameters:
    -----------
    persona_results : pd.DataFrame
        Results from persona tagging
    rules : Dict
        Current rule set
    time_period_days : int
        Number of days to look back for trends (default: 30)
        
    Returns:
    --------
    Dict
        Dashboard data structure
    """
    
    dashboard_data = {
        'summary_stats': {},
        'risk_distribution': {},
        'persona_distribution': {},
        'confidence_analysis': {},
        'alerts': [],
        'trends': {}
    }
    
    # Summary statistics
    total_companies = len(persona_results)
    high_risk_count = len(persona_results[persona_results['risk_level'] == 'high'])
    medium_risk_count = len(persona_results[persona_results['risk_level'] == 'medium'])
    
    dashboard_data['summary_stats'] = {
        'total_companies': total_companies,
        'high_risk_companies': high_risk_count,
        'medium_risk_companies': medium_risk_count,
        'high_risk_percentage': (high_risk_count / total_companies * 100) if total_companies > 0 else 0,
        'avg_confidence': persona_results['confidence_score'].mean(),
        'avg_risk_score': persona_results['risk_score'].mean()
    }
    
    # Risk level distribution
    risk_dist = persona_results['risk_level'].value_counts()
    dashboard_data['risk_distribution'] = {
        level: {'count': int(count), 'percentage': count/total_companies*100}
        for level, count in risk_dist.items()
    }
    
    # Persona distribution
    persona_dist = persona_results['assigned_persona'].value_counts().head(10)
    dashboard_data['persona_distribution'] = {
        persona: {'count': int(count), 'percentage': count/total_companies*100}
        for persona, count in persona_dist.items()
    }
    
    # Confidence analysis
    high_conf = (persona_results['confidence_score'] >= 0.7).sum()
    medium_conf = ((persona_results['confidence_score'] >= 0.5) & 
                   (persona_results['confidence_score'] < 0.7)).sum()
    low_conf = (persona_results['confidence_score'] < 0.5).sum()
    
    dashboard_data['confidence_analysis'] = {
        'high_confidence': {'count': int(high_conf), 'percentage': high_conf/total_companies*100},
        'medium_confidence': {'count': int(medium_conf), 'percentage': medium_conf/total_companies*100},
        'low_confidence': {'count': int(low_conf), 'percentage': low_conf/total_companies*100}
    }
    
    # Generate alerts
    alerts = []
    
    if high_risk_count > total_companies * 0.1:
        alerts.append({
            'level': 'HIGH',
            'message': f"High risk companies ({high_risk_count}) exceed 10% of portfolio",
            'action': 'Review portfolio risk management policies'
        })
    
    if low_conf > total_companies * 0.3:
        alerts.append({
            'level': 'MEDIUM',
            'message': f"Low confidence assignments ({low_conf}) exceed 30% of portfolio",
            'action': 'Consider collecting additional data for better classification'
        })
    
    avg_risk = persona_results['risk_score'].mean()
    if avg_risk > 4:
        alerts.append({
            'level': 'MEDIUM',
            'message': f"Portfolio average risk score ({avg_risk:.1f}) is elevated",
            'action': 'Monitor for deteriorating conditions'
        })
    
    dashboard_data['alerts'] = alerts
    
    return dashboard_data

def demonstrate_system_usage():
    """
    Demonstrate the complete intelligent credit risk system with realistic examples.
    
    This function shows how to use the system end-to-end with synthetic data
    and provides examples of all key functionality.
    """
    
    print("="*80)
    print("INTELLIGENT CREDIT RISK SYSTEM DEMONSTRATION")
    print("="*80)
    
    # Step 1: Generate realistic synthetic data
    print("\n🎯 STEP 1: GENERATING REALISTIC SYNTHETIC DATA")
    print("-" * 50)
    
    # Generate data with diverse company patterns
    df_synthetic = generate_enhanced_synthetic_data(
        num_companies=150,
        days=730,
        include_drift=True,
        random_seed=42
    )
    
    print(f"✅ Generated data for {df_synthetic['company_id'].nunique()} companies")
    print(f"📅 Date range: {df_synthetic['date'].min().date()} to {df_synthetic['date'].max().date()}")
    
    # Step 2: Run the complete analysis pipeline
    print("\n🔄 STEP 2: RUNNING COMPLETE ANALYSIS PIPELINE")
    print("-" * 50)
    
    results = run_complete_risk_analysis_pipeline(
        df_synthetic,
        save_results=True,
        output_dir="demo_output"
    )
    
    # Step 3: Demonstrate rule adjustments
    print("\n⚙️ STEP 3: DEMONSTRATING RULE ADJUSTMENTS")
    print("-" * 50)
    
    original_rules = results['optimized_rules']
    
    # Show current rule for high utilization pattern
    if 'high_utilization_pattern' in original_rules:
        pattern = original_rules['high_utilization_pattern']
        print(f"Original rule: {pattern['description']}")
        
        for rule in pattern['rules']:
            if 'util_mean' in rule['feature']:
                print(f"  Current threshold: {rule['feature']} {rule['operator']} {rule['threshold']:.3f}")
                
                # Adjust the threshold to be more conservative
                adjusted_rules = adjust_rule_threshold(
                    original_rules, 
                    'high_utilization_pattern', 
                    rule['feature'],
                    rule['threshold'] * 1.1  # 10% more conservative
                )
                
                print(f"  Adjusted threshold: {rule['feature']} {rule['operator']} {rule['threshold'] * 1.1:.3f}")
                break
    
    # Step 4: Generate individual company reports
    print("\n📋 STEP 4: GENERATING INDIVIDUAL COMPANY REPORTS")
    print("-" * 50)
    
    persona_results = results['persona_assignments']
    high_risk_companies = persona_results[persona_results['risk_level'] == 'high']
    
    if not high_risk_companies.empty:
        sample_company = high_risk_companies.iloc[0]['company_id']
        individual_report = generate_persona_explanation_report(
            persona_results, 
            original_rules, 
            company_id=sample_company
        )
        
        print(f"📄 Sample report for company {sample_company}:")
        print("-" * 40)
        # Show first few lines of the report
        report_lines = individual_report.split('\n')
        for line in report_lines[:15]:
            print(line)
        print("... (report continues)")
    
    # Step 5: Create monitoring dashboard data
    print("\n📊 STEP 5: CREATING MONITORING DASHBOARD DATA")
    print("-" * 50)
    
    dashboard_data = create_monitoring_dashboard_data(persona_results, original_rules)
    
    print("Dashboard Summary:")
    print(f"  Total Companies: {dashboard_data['summary_stats']['total_companies']}")
    print(f"  High Risk: {dashboard_data['summary_stats']['high_risk_companies']} ({dashboard_data['summary_stats']['high_risk_percentage']:.1f}%)")
    print(f"  Average Confidence: {dashboard_data['summary_stats']['avg_confidence']:.1%}")
    print(f"  Average Risk Score: {dashboard_data['summary_stats']['avg_risk_score']:.1f}/10")
    
    if dashboard_data['alerts']:
        print(f"\n⚠️ Active Alerts: {len(dashboard_data['alerts'])}")
        for alert in dashboard_data['alerts']:
            print(f"  {alert['level']}: {alert['message']}")
    
    # Step 6: Show rule adjustment interface
    print("\n🎛️ STEP 6: RULE ADJUSTMENT INTERFACE")
    print("-" * 50)
    
    adjustment_interface = results['adjustment_interface']
    
    print("Available rule adjustments:")
    for pattern_name, pattern_info in adjustment_interface['current_rules'].items():
        print(f"\n📋 {pattern_name.replace('_', ' ').title()}:")
        print(f"   Risk Level: {pattern_info['risk_level']}")
        print(f"   Description: {pattern_info['pattern_description']}")
        
        for rule in pattern_info['rules'][:2]:  # Show first 2 rules
            print(f"   • {rule['business_meaning']}")
            print(f"     Current: {rule['current_threshold']:.3f}")
            print(f"     Range: {rule['adjustment_range']['minimum']:.3f} - {rule['adjustment_range']['maximum']:.3f}")
    
    print("\n✅ DEMONSTRATION COMPLETE!")
    print("\n📁 Check the 'demo_output' directory for all generated files:")
    print("  • Enhanced feature dataset")
    print("  • Discovered risk patterns")
    print("  • Optimized and validated rules")  
    print("  • Client persona assignments")
    print("  • Executive summary report")
    print("  • Rule adjustment interface")
    
    return results

def validate_system_accuracy(df_with_ground_truth: pd.DataFrame, 
                           persona_results: pd.DataFrame,
                           ground_truth_column: str = 'actual_risk_level') -> Dict:
    """
    Validate system accuracy against known ground truth data.
    
    This function measures how well the system performs against actual known
    risk outcomes, providing validation metrics for the rule-based approach.
    
    Parameters:
    -----------
    df_with_ground_truth : pd.DataFrame
        Data with actual known risk outcomes
    persona_results : pd.DataFrame
        System-generated persona assignments
    ground_truth_column : str
        Column name containing actual risk levels
        
    Returns:
    --------
    Dict
        Validation metrics and accuracy measures
    """
    
    # Merge results with ground truth
    merged = persona_results.merge(
        df_with_ground_truth[['company_id', ground_truth_column]],
        on='company_id',
        how='inner'
    )
    
    if merged.empty:
        return {'error': 'No matching companies found between results and ground truth'}
    
    # Calculate accuracy metrics
    validation_metrics = {}
    
    # Overall accuracy
    correct_predictions = (merged['risk_level'] == merged[ground_truth_column]).sum()
    total_predictions = len(merged)
    overall_accuracy = correct_predictions / total_predictions
    
    validation_metrics['overall_accuracy'] = overall_accuracy
    validation_metrics['total_companies'] = total_predictions
    validation_metrics['correct_predictions'] = correct_predictions
    
    # Risk level specific metrics
    risk_levels = ['high', 'medium', 'low', 'minimal']
    
    for risk_level in risk_levels:
        # True positives, false positives, etc.
        true_positive = ((merged['risk_level'] == risk_level) & 
                        (merged[ground_truth_column] == risk_level)).sum()
        false_positive = ((merged['risk_level'] == risk_level) & 
                         (merged[ground_truth_column] != risk_level)).sum()
        false_negative = ((merged['risk_level'] != risk_level) & 
                         (merged[ground_truth_column] == risk_level)).sum()
        true_negative = ((merged['risk_level'] != risk_level) & 
                        (merged[ground_truth_column] != risk_level)).sum()
        
        # Calculate precision, recall, F1
        precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) > 0 else 0
        recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        validation_metrics[f'{risk_level}_precision'] = precision
        validation_metrics[f'{risk_level}_recall'] = recall
        validation_metrics[f'{risk_level}_f1'] = f1_score
    
    # Confidence analysis
    high_conf_accuracy = merged[merged['confidence_score'] >= 0.7]['risk_level'].eq(
        merged[merged['confidence_score'] >= 0.7][ground_truth_column]
    ).mean()
    
    low_conf_accuracy = merged[merged['confidence_score'] < 0.5]['risk_level'].eq(
        merged[merged['confidence_score'] < 0.5][ground_truth_column]
    ).mean()
    
    validation_metrics['high_confidence_accuracy'] = high_conf_accuracy
    validation_metrics['low_confidence_accuracy'] = low_conf_accuracy
    
    return validation_metrics

# Main execution function
if __name__ == "__main__":
    print("🚀 Starting Intelligent Credit Risk System Demo...")
    
    # Run the complete demonstration
    demo_results = demonstrate_system_usage()
    
    print("\n" + "="*80)
    print("SYSTEM READY FOR PRODUCTION USE")
    print("="*80)
    print("""
📋 TO USE THIS SYSTEM WITH YOUR DATA:

1. PREPARE YOUR DATA:
   • Ensure columns: company_id, date, deposit_balance, used_loan, unused_loan
   • Data should span at least 6 months per company
   • Format dates as YYYY-MM-DD

2. RUN THE PIPELINE:
   results = run_complete_risk_analysis_pipeline(your_dataframe)

3. ADJUST RULES AS NEEDED:
   adjusted_rules = adjust_rule_threshold(results['optimized_rules'], 
                                        pattern_name, feature_name, new_threshold)

4. GENERATE REPORTS:
   report = generate_persona_explanation_report(results['persona_assignments'], 
                                              results['optimized_rules'])

5. MONITOR ONGOING:
   dashboard_data = create_monitoring_dashboard_data(results['persona_assignments'], 
                                                   results['optimized_rules'])

🔧 CUSTOMIZATION OPTIONS:
   • Adjust risk thresholds using the rule adjustment interface
   • Modify feature engineering in calculate_comprehensive_features()
   • Tune clustering parameters in discover_risk_patterns_unsupervised()
   • Customize persona names in generate_persona_name_from_pattern()

💡 BEST PRACTICES:
   • Re-run pattern discovery quarterly to capture new behaviors
   • Validate rule performance against actual defaults monthly
   • Adjust thresholds based on business risk appetite
   • Monitor confidence scores to identify areas needing more data
    """)
