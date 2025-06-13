import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from datetime import datetime, timedelta
from scipy import stats, signal
import statsmodels.api as sm
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.decomposition import PCA
from sklearn.impute import KNNImputer
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from tqdm import tqdm
import matplotlib.ticker as mtick
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import warnings
from statsmodels.tsa.seasonal import seasonal_decompose
from scipy.signal import find_peaks
from scipy.stats import zscore
warnings.filterwarnings('ignore')

# Set styles for better visualizations
plt.style.use('seaborn-v0_8-whitegrid')
custom_params = {"axes.spines.right": False, "axes.spines.top": False}
sns.set_theme(style="ticks", rc=custom_params)

#######################################################
# BUSINESS-READY PERSONA DEFINITIONS (18 PERSONAS)
#######################################################

PERSONA_DEFINITIONS = {
    # === HIGH RISK PERSONAS (Immediate Attention Required) ===
    
    'financial_distress': {
        'risk_level': 'high',
        'description': 'Company showing severe financial deterioration requiring immediate review',
        'business_rules': [
            'Loan utilization increased by more than 25% over 3 months',
            'Deposit balance decreased by more than 25% over 3 months', 
            'Pattern has persisted for at least 2 weeks',
            'Current utilization is above 70%'
        ],
        'action_required': 'Schedule immediate client meeting and credit review',
        'typical_lead_time': '15-30 days before serious problems'
    },
    
    'approaching_credit_limit': {
        'risk_level': 'high',
        'description': 'Company rapidly approaching their credit limit with increasing usage velocity',
        'business_rules': [
            'Current loan utilization is above 90%',
            'Utilization has been increasing by more than 2% per month',
            'Trend has been consistent for at least 1 month'
        ],
        'action_required': 'Contact client immediately about credit limit and usage patterns',
        'typical_lead_time': '7-14 days before limit breach'
    },
    
    'cash_flow_crisis': {
        'risk_level': 'high', 
        'description': 'Company experiencing severe cash flow problems with minimal deposit coverage',
        'business_rules': [
            'Deposit balance is less than 80% of used loan amount',
            'Deposits have declined by more than 30% in the last 2 months',
            'Current loan utilization is above 75%'
        ],
        'action_required': 'Immediate cash flow analysis and potential facility review',
        'typical_lead_time': '10-20 days before potential default'
    },
    
    'erratic_financial_behavior': {
        'risk_level': 'high',
        'description': 'Company showing highly unpredictable financial patterns suggesting operational issues',
        'business_rules': [
            'Deposit or loan usage volatility is 3x higher than historical average',
            'Large unexplained swings (>20%) in weekly averages',
            'Pattern inconsistent with business seasonality'
        ],
        'action_required': 'Investigate underlying business operations and management',
        'typical_lead_time': '20-45 days before major issues surface'
    },
    
    # === MEDIUM RISK PERSONAS (Monitor Closely) ===
    
    'deteriorating_trends': {
        'risk_level': 'medium',
        'description': 'Company showing concerning but not yet critical deterioration in financial metrics',
        'business_rules': [
            'Loan utilization increased by 15-25% over 3 months',
            'Deposit balance decreased by 15-25% over 3 months',
            'Trends have been consistent for at least 3 weeks'
        ],
        'action_required': 'Schedule review meeting within 2 weeks, request updated financials',
        'typical_lead_time': '30-60 days before escalation to high risk'
    },
    
    'credit_dependency_increasing': {
        'risk_level': 'medium',
        'description': 'Company becoming increasingly dependent on credit facilities',
        'business_rules': [
            'Loan utilization consistently above 70% for 2+ months',
            'Average utilization has increased by 10-20% over 6 months',
            'Deposit-to-loan ratio is declining but still above 0.8'
        ],
        'action_required': 'Review credit terms and discuss business growth strategy',
        'typical_lead_time': '45-90 days before becoming high risk'
    },
    
    'seasonal_pattern_break': {
        'risk_level': 'medium',
        'description': 'Company deviating significantly from expected seasonal business patterns',
        'business_rules': [
            'Company historically shows seasonal patterns in deposits or loans',
            'Current financial behavior is 40%+ different from seasonal expectations',
            'Deviation has persisted for more than 3 weeks'
        ],
        'action_required': 'Investigate business changes or market conditions affecting seasonality',
        'typical_lead_time': '30-75 days depending on business cycle'
    },
    
    'withdrawal_pattern_concern': {
        'risk_level': 'medium',
        'description': 'Company showing unusual deposit withdrawal patterns suggesting cash management issues',
        'business_rules': [
            'Withdrawal frequency increased by 50%+ compared to previous quarter',
            'Average withdrawal size increased by 30%+ from historical norm',
            'Deposit balance trending downward despite withdrawals'
        ],
        'action_required': 'Discuss cash flow management and business needs',
        'typical_lead_time': '21-45 days before cash flow issues'
    },
    
    'volatile_utilization': {
        'risk_level': 'medium',
        'description': 'Company showing increased volatility in credit utilization suggesting business instability',
        'business_rules': [
            'Monthly utilization swings exceed 15% for 2+ consecutive months',
            'Utilization changes are not explained by known business seasonality',
            'Overall trend is toward higher average utilization'
        ],
        'action_required': 'Review business operations and cash flow forecasting',
        'typical_lead_time': '30-60 days before operational issues impact credit'
    },
    
    'deposit_concentration_risk': {
        'risk_level': 'medium',
        'description': 'Company showing concerning concentration in deposit timing suggesting liquidity planning issues',
        'business_rules': [
            'More than 60% of monthly deposits occur in concentrated time periods',
            'Deposit pattern suggests dependence on specific payment cycles',
            'Little buffer maintained between deposit cycles'
        ],
        'action_required': 'Discuss diversification of revenue sources and cash management',
        'typical_lead_time': '30-90 days before liquidity issues'
    },
    
    # === LOW RISK PERSONAS (Standard Monitoring) ===
    
    'gradual_growth_concerns': {
        'risk_level': 'low',
        'description': 'Company showing slow but steady increase in credit dependency',
        'business_rules': [
            'Loan utilization has increased by 8-15% over 6 months',
            'Deposit growth is not keeping pace with loan usage growth',
            'Changes are gradual and consistent, not erratic'
        ],
        'action_required': 'Include in quarterly business review discussions',
        'typical_lead_time': '90-180 days before medium risk classification'
    },
    
    'seasonal_loan_user': {
        'risk_level': 'low',
        'description': 'Company with predictable seasonal credit needs showing normal patterns',
        'business_rules': [
            'Loan utilization varies by more than 25% seasonally',
            'Pattern is consistent with historical business cycles',
            'Peak utilization aligns with known industry or business seasonality'
        ],
        'action_required': 'Monitor seasonality remains within expected ranges',
        'typical_lead_time': 'N/A - Normal business pattern'
    },
    
    'minor_volatility': {
        'risk_level': 'low',
        'description': 'Company showing some fluctuation in financial patterns but within acceptable ranges',
        'business_rules': [
            'Deposit or utilization changes are 5-10% above historical volatility',
            'Overall trends remain stable',
            'No concerning patterns in cash flow timing'
        ],
        'action_required': 'Continue standard monitoring protocols',
        'typical_lead_time': '120+ days before escalation likely'
    },
    
    'utilization_plateau': {
        'risk_level': 'low',
        'description': 'Company with consistently high but stable credit utilization',
        'business_rules': [
            'Loan utilization consistently 60-80% for 3+ months',
            'Utilization level is stable (less than 5% monthly variation)',
            'Deposit coverage remains adequate'
        ],
        'action_required': 'Annual review of credit limits and business growth plans',
        'typical_lead_time': '180+ days before risk level changes'
    },
    
    # === POSITIVE RISK PERSONAS (Good Standing) ===
    
    'cautious_borrower': {
        'risk_level': 'low',
        'description': 'Company maintaining conservative credit usage with strong deposit coverage',
        'business_rules': [
            'Loan utilization consistently below 40%',
            'Deposit balance typically exceeds 150% of used credit',
            'Financial patterns are stable and predictable'
        ],
        'action_required': 'Standard relationship management - potential for growth opportunities',
        'typical_lead_time': 'N/A - Low risk profile'
    },
    
    'improving_financial_health': {
        'risk_level': 'low',
        'description': 'Company showing positive trends in financial management',
        'business_rules': [
            'Loan utilization has decreased by 10%+ over 6 months',
            'Deposit balances are growing or stable',
            'Financial volatility is decreasing'
        ],
        'action_required': 'Relationship building - potential for additional services',
        'typical_lead_time': 'N/A - Positive trend'
    },
    
    'stable_operations': {
        'risk_level': 'low',
        'description': 'Company with consistent, predictable financial patterns suggesting stable operations',
        'business_rules': [
            'All financial metrics within 10% of 6-month averages',
            'Utilization between 30-60% consistently',
            'No unexplained volatility or pattern breaks'
        ],
        'action_required': 'Standard monitoring - good candidate for credit line increases',
        'typical_lead_time': 'N/A - Stable profile'
    },
    
    # === SPECIAL MONITORING PERSONAS ===
    
    'data_quality_concerns': {
        'risk_level': 'medium',
        'description': 'Company with inconsistent or poor quality financial data affecting risk assessment',
        'business_rules': [
            'Frequent zero values or missing data points',
            'Unexplained large daily variations (>50%)',
            'Signal quality metrics below reliability thresholds'
        ],
        'action_required': 'Investigate data reporting issues with client',
        'typical_lead_time': 'Immediate - Data quality affects all risk assessments'
    }
}

#######################################################
# ENHANCED CONFIGURATION
#######################################################

CONFIG = {
    'data': {
        'min_nonzero_pct': 0.8,
        'min_continuous_days': 365,
        'recent_window': 30
    },
    'smoothing': {
        'short_window': 7,
        'medium_window': 14,
        'long_window': 30,
        'ema_alpha': 0.3,
        'outlier_threshold': 3.0,
        'min_data_points': 21
    },
    'business_thresholds': {
        'utilization_sharp_increase': 0.25,
        'utilization_moderate_increase': 0.15,
        'utilization_gradual_increase': 0.08,
        'deposit_sharp_decline': -0.25,
        'deposit_moderate_decline': -0.15,
        'deposit_gradual_decline': -0.08,
        'utilization_critical': 0.90,
        'utilization_high': 0.75,
        'utilization_moderate': 0.60,
        'utilization_low': 0.40,
        'coverage_critical': 0.8,
        'coverage_low': 1.0,
        'coverage_adequate': 1.5,
        'volatility_extreme': 3.0,
        'volatility_high': 2.0,
        'volatility_moderate': 1.5,
        'persistence_short': 7,
        'persistence_medium': 14,
        'persistence_long': 21,
        'signal_quality_good': 3.0,
        'signal_quality_acceptable': 2.0,
        'signal_quality_poor': 1.0
    },
    'clustering': {
        'n_clusters_range': range(3, 12),  # Test 3-11 clusters
        'random_state': 42,
        'features_for_clustering': [
            'util_smooth_14d', 'util_change_90d', 'deposit_change_90d',
            'util_volatility', 'deposit_volatility', 'deposit_loan_ratio',
            'util_increasing_persistence', 'deposit_decreasing_persistence'
        ]
    }
}

#######################################################
# DATA PROCESSING FUNCTIONS (Enhanced with Noise Reduction)
#######################################################

def detect_and_handle_outliers(series, method='iqr', threshold=3.0):
    """
    BUSINESS PURPOSE: Remove data spikes that could cause false alarms
    
    This function identifies and smooths outliers so they don't trigger false risk alerts.
    Daily financial data often has outliers due to system maintenance, large transactions,
    weekend effects, and data entry errors.
    """
    series_clean = series.copy()
    
    if len(series_clean.dropna()) < 5:
        return series_clean
    
    if method == 'iqr':
        Q1 = series_clean.quantile(0.25)
        Q3 = series_clean.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        series_clean = series_clean.clip(lower=lower_bound, upper=upper_bound)
        
    elif method == 'modified_zscore':
        median = series_clean.median()
        mad = np.median(np.abs(series_clean - median))
        modified_z_scores = 0.6745 * (series_clean - median) / mad
        outlier_mask = np.abs(modified_z_scores) > threshold
        series_clean.loc[outlier_mask] = median
    
    return series_clean

def calculate_exponential_moving_average(series, alpha=0.3, adjust_alpha=True):
    """
    BUSINESS PURPOSE: Create smoothed trends that respond to recent changes
    
    Exponential moving averages give more weight to recent data points, making them
    ideal for credit risk monitoring because recent deterioration is more important
    than old problems.
    """
    if adjust_alpha and len(series.dropna()) > 20:
        volatility = series.pct_change().std()
        if not pd.isna(volatility) and volatility > 0.1:
            alpha = max(0.1, alpha * (0.1 / volatility))
    
    return series.ewm(alpha=alpha, adjust=False).mean()

def calculate_signal_quality(raw_series, smoothed_series):
    """
    BUSINESS PURPOSE: Identify companies with unreliable data
    
    This function calculates a "signal quality" score. Low scores mean the data
    is too noisy to generate reliable risk alerts.
    """
    if len(raw_series.dropna()) < 10 or len(smoothed_series.dropna()) < 10:
        return pd.Series(index=raw_series.index, dtype=float)
    
    noise = np.abs(raw_series - smoothed_series)
    signal = np.abs(smoothed_series)
    signal_quality = signal / (noise + 1e-8)
    signal_quality = signal_quality.clip(upper=100)
    signal_quality = signal_quality.rolling(window=7, min_periods=3).mean()
    
    return signal_quality

def calculate_pattern_persistence(df, company_idx, util_series, deposit_series):
    """
    BUSINESS PURPOSE: Track how long concerning patterns have persisted
    
    This function tracks how many days a concerning pattern has been in place,
    allowing credit officers to focus on sustained problems rather than temporary fluctuations.
    """
    util_changes = util_series.pct_change(periods=7).rolling(window=7).mean()
    util_increasing = (util_changes > 0.02).astype(int)
    util_increasing_days = util_increasing.rolling(window=21).sum()
    
    deposit_changes = deposit_series.pct_change(periods=7).rolling(window=7).mean()
    deposit_decreasing = (deposit_changes < -0.02).astype(int)
    deposit_decreasing_days = deposit_decreasing.rolling(window=21).sum()
    
    df.loc[company_idx, 'util_increasing_persistence'] = util_increasing_days
    df.loc[company_idx, 'deposit_decreasing_persistence'] = deposit_decreasing_days

def add_derived_metrics_enhanced(df):
    """
    BUSINESS PURPOSE: Create reliable financial indicators from noisy daily data
    
    This function transforms raw daily deposit and loan data into smooth, reliable
    indicators that credit officers can trust for risk assessment.
    """
    df = df.copy()
    
    # Calculate basic ratios
    df['total_loan'] = df['used_loan'] + df['unused_loan']
    df['loan_utilization'] = df['used_loan'] / df['total_loan']
    df.loc[df['total_loan'] == 0, 'loan_utilization'] = np.nan
    df['deposit_loan_ratio'] = df['deposit_balance'] / df['used_loan']
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    
    # Process each company individually for enhanced metrics
    for company in tqdm(df['company_id'].unique(), desc="Processing company data"):
        company_data = df[df['company_id'] == company].sort_values('date')
        
        if len(company_data) < CONFIG['smoothing']['min_data_points']:
            continue
        
        company_idx = company_data.index
        
        # Clean outliers that could cause false alerts
        clean_utilization = detect_and_handle_outliers(
            company_data['loan_utilization'], method='modified_zscore'
        )
        clean_deposits = detect_and_handle_outliers(
            company_data['deposit_balance'], method='iqr'
        )
        
        # Create multiple smoothing levels for different analysis purposes
        util_ema_7d = calculate_exponential_moving_average(clean_utilization, alpha=0.4)
        deposit_ema_7d = calculate_exponential_moving_average(clean_deposits, alpha=0.4)
        
        util_ma_14d = clean_utilization.rolling(window=14, min_periods=5).mean()
        deposit_ma_14d = clean_deposits.rolling(window=14, min_periods=5).mean()
        
        util_ma_30d = clean_utilization.rolling(window=30, min_periods=10).mean()
        deposit_ma_30d = clean_deposits.rolling(window=30, min_periods=10).mean()
        
        # Store smoothed values
        df.loc[company_idx, 'util_smooth_7d'] = util_ema_7d
        df.loc[company_idx, 'deposit_smooth_7d'] = deposit_ema_7d
        df.loc[company_idx, 'util_smooth_14d'] = util_ma_14d
        df.loc[company_idx, 'deposit_smooth_14d'] = deposit_ma_14d
        df.loc[company_idx, 'util_smooth_30d'] = util_ma_30d
        df.loc[company_idx, 'deposit_smooth_30d'] = deposit_ma_30d
        
        # Calculate reliable change indicators
        util_change_30d = util_ma_14d.pct_change(periods=30).rolling(window=7, min_periods=3).mean()
        deposit_change_30d = deposit_ma_14d.pct_change(periods=30).rolling(window=7, min_periods=3).mean()
        util_change_90d = util_ma_14d.pct_change(periods=90).rolling(window=7, min_periods=3).mean()
        deposit_change_90d = deposit_ma_14d.pct_change(periods=90).rolling(window=7, min_periods=3).mean()
        
        df.loc[company_idx, 'util_change_30d'] = util_change_30d
        df.loc[company_idx, 'deposit_change_30d'] = deposit_change_30d
        df.loc[company_idx, 'util_change_90d'] = util_change_90d
        df.loc[company_idx, 'deposit_change_90d'] = deposit_change_90d
        
        # Calculate volatility measures
        util_volatility = util_ma_14d.pct_change().rolling(window=21, min_periods=10).std()
        deposit_volatility = deposit_ma_14d.pct_change().rolling(window=21, min_periods=10).std()
        
        df.loc[company_idx, 'util_volatility'] = util_volatility
        df.loc[company_idx, 'deposit_volatility'] = deposit_volatility
        
        # Calculate signal quality for data reliability assessment
        signal_quality_util = calculate_signal_quality(clean_utilization, util_ma_14d)
        signal_quality_deposit = calculate_signal_quality(clean_deposits, deposit_ma_14d)
        
        df.loc[company_idx, 'signal_quality'] = (signal_quality_util + signal_quality_deposit) / 2
        
        # Track pattern persistence
        calculate_pattern_persistence(df, company_idx, util_ma_14d, deposit_ma_14d)
    
    return df

#######################################################
# OPTIMAL CLUSTERING ANALYSIS FOR PERSONA DISCOVERY
#######################################################

def find_optimal_clusters(df, features_for_clustering):
    """
    BUSINESS PURPOSE: Discover natural groupings in financial behavior
    
    This function uses advanced clustering techniques to find natural groups
    of companies with similar financial patterns. This helps validate our
    rule-based personas and discover new patterns we might have missed.
    
    We test multiple clustering algorithms and different numbers of clusters
    to find the most meaningful groupings in the data.
    """
    print("🔍 Finding optimal number of clusters using multiple validation metrics...")
    
    # Prepare data for clustering
    clustering_data = prepare_clustering_data(df, features_for_clustering)
    
    if clustering_data.empty:
        print("❌ Insufficient data for clustering analysis")
        return None, None, None
    
    print(f"📊 Clustering analysis using {len(clustering_data)} companies with {len(features_for_clustering)} features")
    
    # Test different numbers of clusters
    silhouette_scores = []
    calinski_scores = []
    davies_bouldin_scores = []
    
    cluster_range = CONFIG['clustering']['n_clusters_range']
    
    for n_clusters in cluster_range:
        # Fit KMeans clustering
        kmeans = KMeans(
            n_clusters=n_clusters, 
            random_state=CONFIG['clustering']['random_state'],
            n_init=10
        )
        cluster_labels = kmeans.fit_predict(clustering_data)
        
        # Calculate validation metrics
        sil_score = silhouette_score(clustering_data, cluster_labels)
        cal_score = calinski_harabasz_score(clustering_data, cluster_labels)
        db_score = davies_bouldin_score(clustering_data, cluster_labels)
        
        silhouette_scores.append(sil_score)
        calinski_scores.append(cal_score)
        davies_bouldin_scores.append(db_score)
        
        print(f"  {n_clusters} clusters: Silhouette={sil_score:.3f}, Calinski-Harabasz={cal_score:.1f}, Davies-Bouldin={db_score:.3f}")
    
    # Find optimal number of clusters using combined criteria
    optimal_clusters = find_best_cluster_number(
        cluster_range, silhouette_scores, calinski_scores, davies_bouldin_scores
    )
    
    print(f"🎯 Optimal number of clusters: {optimal_clusters}")
    
    # Fit final clustering model
    final_kmeans = KMeans(
        n_clusters=optimal_clusters,
        random_state=CONFIG['clustering']['random_state'],
        n_init=10
    )
    
    final_labels = final_kmeans.fit_predict(clustering_data)
    
    # Create clustering results
    clustering_results = create_clustering_results(df, clustering_data, final_labels, final_kmeans)
    
    validation_metrics = {
        'optimal_clusters': optimal_clusters,
        'silhouette_score': silhouette_score(clustering_data, final_labels),
        'calinski_harabasz_score': calinski_harabasz_score(clustering_data, final_labels),
        'davies_bouldin_score': davies_bouldin_score(clustering_data, final_labels),
        'cluster_range_tested': list(cluster_range),
        'all_silhouette_scores': silhouette_scores,
        'all_calinski_scores': calinski_scores,
        'all_davies_bouldin_scores': davies_bouldin_scores
    }
    
    return clustering_results, final_kmeans, validation_metrics

def prepare_clustering_data(df, features_for_clustering):
    """
    BUSINESS PURPOSE: Prepare clean, normalized data for clustering analysis
    
    This function takes the enhanced financial metrics and prepares them for
    clustering by handling missing values, outliers, and normalization.
    """
    
    # Get the most recent data point for each company
    latest_data = df.groupby('company_id').last().reset_index()
    
    # Extract clustering features
    clustering_features = []
    
    for feature in features_for_clustering:
        if feature in latest_data.columns:
            clustering_features.append(feature)
        else:
            print(f"⚠️ Feature {feature} not found in data, skipping")
    
    if not clustering_features:
        return pd.DataFrame()
    
    # Create clustering dataset
    cluster_data = latest_data[['company_id'] + clustering_features].copy()
    
    # Handle missing values
    cluster_data = cluster_data.dropna()
    
    if len(cluster_data) < 10:
        print(f"⚠️ Too few companies ({len(cluster_data)}) with complete data for clustering")
        return pd.DataFrame()
    
    # Detect and handle outliers for each feature
    for feature in clustering_features:
        cluster_data[feature] = detect_and_handle_outliers(cluster_data[feature], method='iqr')
    
    # Normalize features using RobustScaler (less sensitive to outliers)
    scaler = RobustScaler()
    feature_data = cluster_data[clustering_features]
    scaled_features = scaler.fit_transform(feature_data)
    
    # Create scaled dataframe
    scaled_df = pd.DataFrame(
        scaled_features, 
        columns=clustering_features,
        index=cluster_data.index
    )
    
    # Add company IDs back
    scaled_df['company_id'] = cluster_data['company_id'].values
    
    return scaled_df.set_index('company_id')

def find_best_cluster_number(cluster_range, silhouette_scores, calinski_scores, davies_bouldin_scores):
    """
    BUSINESS PURPOSE: Use multiple criteria to find the optimal number of clusters
    
    Different clustering validation metrics sometimes disagree, so we use a
    combined scoring approach that balances multiple criteria:
    - Silhouette Score: Higher is better (measures cluster separation)
    - Calinski-Harabasz: Higher is better (measures cluster compactness vs separation)  
    - Davies-Bouldin: Lower is better (measures average similarity between clusters)
    """
    
    # Normalize scores to 0-1 range for comparison
    sil_normalized = np.array(silhouette_scores)
    sil_normalized = (sil_normalized - sil_normalized.min()) / (sil_normalized.max() - sil_normalized.min())
    
    cal_normalized = np.array(calinski_scores)
    cal_normalized = (cal_normalized - cal_normalized.min()) / (cal_normalized.max() - cal_normalized.min())
    
    db_normalized = np.array(davies_bouldin_scores)
    db_normalized = (db_normalized.max() - db_normalized) / (db_normalized.max() - db_normalized.min())  # Invert because lower is better
    
    # Combined score with equal weighting
    combined_scores = (sil_normalized + cal_normalized + db_normalized) / 3
    
    # Find the number of clusters with the highest combined score
    optimal_idx = np.argmax(combined_scores)
    optimal_clusters = list(cluster_range)[optimal_idx]
    
    return optimal_clusters

def create_clustering_results(df, clustering_data, cluster_labels, kmeans_model):
    """
    BUSINESS PURPOSE: Create interpretable results from clustering analysis
    
    This function takes the raw clustering output and creates business-friendly
    summaries that show the characteristics of each discovered cluster.
    """
    
    # Add cluster labels to the data
    cluster_df = clustering_data.copy()
    cluster_df['cluster'] = cluster_labels
    
    # Calculate cluster characteristics
    cluster_characteristics = {}
    
    for cluster_id in range(len(np.unique(cluster_labels))):
        cluster_companies = cluster_df[cluster_df['cluster'] == cluster_id]
        
        if len(cluster_companies) == 0:
            continue
        
        # Calculate mean characteristics for this cluster
        cluster_means = cluster_companies.drop(['cluster'], axis=1).mean()
        
        # Describe this cluster in business terms
        cluster_description = describe_cluster_business_characteristics(cluster_means)
        
        cluster_characteristics[f'cluster_{cluster_id}'] = {
            'size': len(cluster_companies),
            'percentage': len(cluster_companies) / len(cluster_df) * 100,
            'companies': cluster_companies.index.tolist(),
            'mean_characteristics': cluster_means.to_dict(),
            'business_description': cluster_description
        }
    
    # Create summary
    clustering_summary = {
        'total_companies': len(cluster_df),
        'num_clusters': len(cluster_characteristics),
        'cluster_characteristics': cluster_characteristics,
        'cluster_assignments': cluster_df['cluster'].to_dict()
    }
    
    return clustering_summary

def describe_cluster_business_characteristics(cluster_means):
    """
    BUSINESS PURPOSE: Translate cluster statistics into business language
    
    This function takes the statistical characteristics of a cluster and
    translates them into business terms that credit officers can understand.
    """
    
    description_elements = []
    
    # Analyze utilization level
    util_level = cluster_means.get('util_smooth_14d', 0)
    if util_level > 0.8:
        description_elements.append("Very high credit utilization (>80%)")
    elif util_level > 0.6:
        description_elements.append("High credit utilization (60-80%)")
    elif util_level > 0.4:
        description_elements.append("Moderate credit utilization (40-60%)")
    else:
        description_elements.append("Low credit utilization (<40%)")
    
    # Analyze utilization trend
    util_change = cluster_means.get('util_change_90d', 0)
    if util_change > 0.15:
        description_elements.append("rapidly increasing utilization")
    elif util_change > 0.05:
        description_elements.append("gradually increasing utilization")
    elif util_change < -0.05:
        description_elements.append("decreasing utilization")
    else:
        description_elements.append("stable utilization")
    
    # Analyze deposit trends
    deposit_change = cluster_means.get('deposit_change_90d', 0)
    if deposit_change < -0.15:
        description_elements.append("declining deposits")
    elif deposit_change > 0.15:
        description_elements.append("growing deposits")
    else:
        description_elements.append("stable deposits")
    
    # Analyze volatility
    util_volatility = cluster_means.get('util_volatility', 0)
    if util_volatility > 0.1:
        description_elements.append("high volatility")
    elif util_volatility > 0.05:
        description_elements.append("moderate volatility")
    else:
        description_elements.append("low volatility")
    
    # Analyze deposit coverage
    deposit_ratio = cluster_means.get('deposit_loan_ratio', 1)
    if deposit_ratio < 0.8:
        description_elements.append("low deposit coverage")
    elif deposit_ratio > 1.5:
        description_elements.append("strong deposit coverage")
    else:
        description_elements.append("adequate deposit coverage")
    
    return "; ".join(description_elements)

def compare_clusters_with_personas(clustering_results, persona_assignments):
    """
    BUSINESS PURPOSE: Compare discovered clusters with rule-based personas
    
    This function analyzes how well our rule-based personas align with the
    natural clusters discovered in the data. This helps validate our business
    rules and identify potential gaps or improvements.
    """
    
    if not clustering_results or persona_assignments.empty:
        print("❌ Cannot compare clusters with personas - insufficient data")
        return None
    
    print("🔍 Comparing discovered clusters with rule-based personas...")
    
    # Get the most recent persona assignment for each company
    latest_personas = persona_assignments.groupby('company_id').last()
    
    # Match companies between clustering and persona assignments
    cluster_assignments = clustering_results['cluster_assignments']
    
    comparison_results = []
    
    for company_id, cluster_id in cluster_assignments.items():
        if company_id in latest_personas.index:
            persona = latest_personas.loc[company_id, 'persona']
            risk_level = latest_personas.loc[company_id, 'risk_level']
            confidence = latest_personas.loc[company_id, 'confidence']
            
            comparison_results.append({
                'company_id': company_id,
                'cluster_id': cluster_id,
                'persona': persona,
                'risk_level': risk_level,
                'confidence': confidence
            })
    
    comparison_df = pd.DataFrame(comparison_results)
    
    if comparison_df.empty:
        print("❌ No matching companies between clusters and personas")
        return None
    
    # Analyze cluster-persona alignment
    cluster_persona_analysis = {}
    
    for cluster_id in comparison_df['cluster_id'].unique():
        cluster_data = comparison_df[comparison_df['cluster_id'] == cluster_id]
        
        # Most common personas in this cluster
        persona_distribution = cluster_data['persona'].value_counts()
        risk_level_distribution = cluster_data['risk_level'].value_counts()
        avg_confidence = cluster_data['confidence'].mean()
        
        cluster_persona_analysis[f'cluster_{cluster_id}'] = {
            'size': len(cluster_data),
            'most_common_persona': persona_distribution.index[0] if not persona_distribution.empty else 'Unknown',
            'persona_distribution': persona_distribution.to_dict(),
            'risk_level_distribution': risk_level_distribution.to_dict(),
            'avg_confidence': avg_confidence,
            'companies': cluster_data['company_id'].tolist()
        }
    
    # Calculate alignment metrics
    alignment_metrics = calculate_cluster_persona_alignment(comparison_df)
    
    comparison_summary = {
        'cluster_persona_analysis': cluster_persona_analysis,
        'alignment_metrics': alignment_metrics,
        'comparison_data': comparison_df
    }
    
    print(f"✅ Cluster-Persona comparison complete:")
    print(f"   - Companies analyzed: {len(comparison_df)}")
    print(f"   - Clusters found: {comparison_df['cluster_id'].nunique()}")
    print(f"   - Unique personas: {comparison_df['persona'].nunique()}")
    print(f"   - Average alignment score: {alignment_metrics.get('average_alignment', 0):.2f}")
    
    return comparison_summary

def calculate_cluster_persona_alignment(comparison_df):
    """
    BUSINESS PURPOSE: Calculate how well clusters align with persona assignments
    
    This function measures the "purity" of clusters - how consistently companies
    in the same cluster are assigned to similar personas and risk levels.
    """
    
    alignment_scores = []
    
    for cluster_id in comparison_df['cluster_id'].unique():
        cluster_data = comparison_df[comparison_df['cluster_id'] == cluster_id]
        
        # Calculate persona consistency within cluster
        persona_counts = cluster_data['persona'].value_counts()
        max_persona_count = persona_counts.iloc[0] if not persona_counts.empty else 0
        persona_purity = max_persona_count / len(cluster_data)
        
        # Calculate risk level consistency within cluster
        risk_counts = cluster_data['risk_level'].value_counts()
        max_risk_count = risk_counts.iloc[0] if not risk_counts.empty else 0
        risk_purity = max_risk_count / len(cluster_data)
        
        # Combined alignment score
        alignment_score = (persona_purity + risk_purity) / 2
        alignment_scores.append(alignment_score)
    
    return {
        'average_alignment': np.mean(alignment_scores) if alignment_scores else 0,
        'cluster_alignment_scores': alignment_scores,
        'persona_entropy': calculate_entropy(comparison_df, 'persona'),
        'risk_level_entropy': calculate_entropy(comparison_df, 'risk_level')
    }

def calculate_entropy(df, column):
    """Calculate entropy of a categorical variable (lower entropy = better clustering)"""
    value_counts = df[column].value_counts()
    probabilities = value_counts / len(df)
    entropy = -np.sum(probabilities * np.log2(probabilities + 1e-10))
    return entropy

#######################################################
# RULE-BASED PERSONA ASSIGNMENT (Enhanced)
#######################################################

def assign_persona_with_business_rules(current_row, company_data, current_index):
    """
    BUSINESS PURPOSE: Assign risk personas using clear, explainable business rules
    
    This function evaluates each company against the defined business rules for all 18 personas.
    It returns the best-matching persona along with confidence level and the specific rules triggered.
    """
    
    # Extract current financial metrics (smoothed for reliability)
    current_util = current_row.get('util_smooth_14d', current_row.get('loan_utilization', 0))
    current_deposit = current_row.get('deposit_smooth_14d', current_row.get('deposit_balance', 0))
    deposit_ratio = current_row.get('deposit_loan_ratio', float('inf'))
    signal_quality = current_row.get('signal_quality', 0)
    
    # Extract trend indicators
    util_change_30d = current_row.get('util_change_30d', 0)
    util_change_90d = current_row.get('util_change_90d', 0) 
    deposit_change_30d = current_row.get('deposit_change_30d', 0)
    deposit_change_90d = current_row.get('deposit_change_90d', 0)
    
    # Extract volatility and persistence indicators
    util_volatility = current_row.get('util_volatility', 0)
    deposit_volatility = current_row.get('deposit_volatility', 0)
    util_persistence = current_row.get('util_increasing_persistence', 0)
    deposit_persistence = current_row.get('deposit_decreasing_persistence', 0)
    
    # Skip analysis if data quality is too poor
    if signal_quality < CONFIG['business_thresholds']['signal_quality_poor']:
        return 'data_quality_concerns', 0.5, ['Poor data quality affects reliable risk assessment']
    
    matched_personas = []
    
    # HIGH RISK PERSONA EVALUATION
    
    # === FINANCIAL DISTRESS ===
    if (not pd.isna(util_change_90d) and not pd.isna(deposit_change_90d) and
        util_change_90d > CONFIG['business_thresholds']['utilization_sharp_increase'] and
        deposit_change_90d < CONFIG['business_thresholds']['deposit_sharp_decline'] and
        current_util > 0.70 and
        util_persistence >= CONFIG['business_thresholds']['persistence_medium']):
        
        confidence = 0.9
        triggered_rules = [
            f"Utilization increased {util_change_90d:.1%} over 3 months (threshold: {CONFIG['business_thresholds']['utilization_sharp_increase']:.1%})",
            f"Deposits declined {deposit_change_90d:.1%} over 3 months (threshold: {CONFIG['business_thresholds']['deposit_sharp_decline']:.1%})",
            f"Current utilization {current_util:.1%} above 70% threshold",
            f"Pattern persisted for {util_persistence:.0f} days (minimum: {CONFIG['business_thresholds']['persistence_medium']} days)"
        ]
        matched_personas.append(('financial_distress', confidence, triggered_rules))
    
    # === APPROACHING CREDIT LIMIT ===
    if (current_util > CONFIG['business_thresholds']['utilization_critical'] and
        not pd.isna(util_change_30d) and util_change_30d > 0.02 and
        util_persistence >= CONFIG['business_thresholds']['persistence_short']):
        
        confidence = 0.85
        triggered_rules = [
            f"Current utilization {current_util:.1%} above critical threshold {CONFIG['business_thresholds']['utilization_critical']:.1%}",
            f"Utilization increasing {util_change_30d:.1%} per month",
            f"Increasing trend sustained for {util_persistence:.0f} days"
        ]
        matched_personas.append(('approaching_credit_limit', confidence, triggered_rules))
    
    # === CASH FLOW CRISIS ===
    if (deposit_ratio < CONFIG['business_thresholds']['coverage_critical'] and
        not pd.isna(deposit_change_90d) and deposit_change_90d < -0.30 and
        current_util > CONFIG['business_thresholds']['utilization_high']):
        
        confidence = 0.88
        triggered_rules = [
            f"Deposit coverage {deposit_ratio:.2f} below critical threshold {CONFIG['business_thresholds']['coverage_critical']:.2f}",
            f"Deposits declined {deposit_change_90d:.1%} over 2 months",
            f"High utilization {current_util:.1%} indicates credit dependency"
        ]
        matched_personas.append(('cash_flow_crisis', confidence, triggered_rules))
    
    # MEDIUM RISK PERSONA EVALUATION
    
    # === DETERIORATING TRENDS ===
    if (not pd.isna(util_change_90d) and not pd.isna(deposit_change_90d) and
        CONFIG['business_thresholds']['utilization_moderate_increase'] <= util_change_90d < CONFIG['business_thresholds']['utilization_sharp_increase'] and
        CONFIG['business_thresholds']['deposit_moderate_decline'] <= deposit_change_90d < CONFIG['business_thresholds']['deposit_sharp_decline'] and
        util_persistence >= CONFIG['business_thresholds']['persistence_long']):
        
        confidence = 0.75
        triggered_rules = [
            f"Moderate utilization increase {util_change_90d:.1%} over 3 months",
            f"Moderate deposit decline {deposit_change_90d:.1%} over 3 months", 
            f"Trends sustained for {util_persistence:.0f} days"
        ]
        matched_personas.append(('deteriorating_trends', confidence, triggered_rules))
    
    # === CREDIT DEPENDENCY INCREASING ===
    if (current_util > CONFIG['business_thresholds']['utilization_high'] and
        not pd.isna(util_change_90d) and
        CONFIG['business_thresholds']['utilization_gradual_increase'] <= util_change_90d < CONFIG['business_thresholds']['utilization_moderate_increase'] and
        deposit_ratio > CONFIG['business_thresholds']['coverage_critical']):
        
        confidence = 0.70
        triggered_rules = [
            f"High utilization {current_util:.1%} maintained for extended period",
            f"Gradual utilization increase {util_change_90d:.1%} over 6 months",
            f"Deposit coverage {deposit_ratio:.2f} still adequate but concerning trend"
        ]
        matched_personas.append(('credit_dependency_increasing', confidence, triggered_rules))
    
    # LOW RISK PERSONA EVALUATION
    
    # === CAUTIOUS BORROWER ===
    if (current_util < CONFIG['business_thresholds']['utilization_low'] and
        deposit_ratio > CONFIG['business_thresholds']['coverage_adequate'] and
        (pd.isna(util_change_90d) or abs(util_change_90d) < 0.05)):
        
        confidence = 0.80
        triggered_rules = [
            f"Low utilization {current_util:.1%} well below {CONFIG['business_thresholds']['utilization_low']:.1%} threshold",
            f"Strong deposit coverage {deposit_ratio:.2f}x used credit",
            "Stable financial patterns indicate conservative management"
        ]
        matched_personas.append(('cautious_borrower', confidence, triggered_rules))
    
    # === STABLE OPERATIONS ===
    if (CONFIG['business_thresholds']['utilization_low'] <= current_util <= CONFIG['business_thresholds']['utilization_moderate'] and
        deposit_ratio >= CONFIG['business_thresholds']['coverage_low'] and
        (pd.isna(util_change_90d) or abs(util_change_90d) < 0.10) and
        (pd.isna(util_volatility) or util_volatility < 0.03)):
        
        confidence = 0.75
        triggered_rules = [
            f"Moderate utilization {current_util:.1%} within normal operating range",
            f"Adequate deposit coverage {deposit_ratio:.2f}",
            "Low volatility and stable trends indicate healthy operations"
        ]
        matched_personas.append(('stable_operations', confidence, triggered_rules))
    
    # === IMPROVING FINANCIAL HEALTH ===
    if (not pd.isna(util_change_90d) and util_change_90d < -0.10 and
        not pd.isna(deposit_change_90d) and deposit_change_90d > 0.05):
        
        confidence = 0.70
        triggered_rules = [
            f"Utilization decreased {abs(util_change_90d):.1%} over 3 months",
            f"Deposits increased {deposit_change_90d:.1%} over 3 months",
            "Positive trends indicate improving financial management"
        ]
        matched_personas.append(('improving_financial_health', confidence, triggered_rules))
    
    # DATA QUALITY CONCERNS
    if signal_quality < CONFIG['business_thresholds']['signal_quality_acceptable']:
        confidence = 0.85
        triggered_rules = [
            f"Signal quality {signal_quality:.2f} below acceptable threshold {CONFIG['business_thresholds']['signal_quality_acceptable']:.2f}",
            "Frequent data anomalies or missing values detected",
            "Recommend investigating data reporting processes with client"
        ]
        matched_personas.append(('data_quality_concerns', confidence, triggered_rules))
    
    # PERSONA SELECTION
    if not matched_personas:
        # Default assignment for companies that don't match specific patterns
        if current_util < 0.3:
            return 'cautious_borrower', 0.50, ['Default assignment: Low utilization with no concerning patterns']
        elif current_util > 0.7:
            return 'utilization_plateau', 0.50, ['Default assignment: High but stable utilization']
        else:
            return 'stable_operations', 0.50, ['Default assignment: Moderate utilization with no patterns']
    
    # Select the highest confidence match
    best_match = max(matched_personas, key=lambda x: x[1])
    persona_name, confidence, triggered_rules = best_match
    
    # Adjust confidence based on signal quality
    if signal_quality > CONFIG['business_thresholds']['signal_quality_good']:
        confidence = min(1.0, confidence + 0.05)
    elif signal_quality < CONFIG['business_thresholds']['signal_quality_acceptable']:
        confidence = max(0.3, confidence - 0.15)
    
    return persona_name, confidence, triggered_rules

def detect_risk_patterns_with_business_rules(df):
    """
    BUSINESS PURPOSE: Generate risk alerts using clear, explainable business logic
    
    This function evaluates each company against predefined business rules and generates
    risk alerts that credit officers can easily understand and explain to stakeholders.
    """
    
    risk_records = []
    persona_assignments = []
    
    max_date = df['date'].max()
    recent_cutoff = max_date - pd.Timedelta(days=CONFIG['data']['recent_window'])
    
    print(f"Analyzing {df['company_id'].nunique()} companies for risk patterns...")
    
    for company in tqdm(df['company_id'].unique(), desc="Evaluating companies"):
        company_data = df[df['company_id'] == company].sort_values('date')
        
        if len(company_data) < CONFIG['smoothing']['min_data_points']:
            continue
        
        step_size = 7 if len(company_data) > 180 else 14
        
        for i in range(CONFIG['smoothing']['min_data_points'], len(company_data), step_size):
            current_row = company_data.iloc[i]
            current_date = current_row['date']
            
            if i < len(company_data) - 1 and current_date < recent_cutoff:
                continue
            
            persona_name, confidence, triggered_rules = assign_persona_with_business_rules(
                current_row, company_data, i
            )
            
            if persona_name and confidence >= 0.5:
                persona_info = PERSONA_DEFINITIONS.get(persona_name, {})
                risk_level = persona_info.get('risk_level', 'low')
                action_required = persona_info.get('action_required', 'Standard monitoring')
                
                risk_records.append({
                    'company_id': company,
                    'date': current_date,
                    'persona': persona_name,
                    'risk_level': risk_level,
                    'confidence': confidence,
                    'triggered_rules': ' | '.join(triggered_rules),
                    'action_required': action_required,
                    'signal_quality': current_row.get('signal_quality', 0),
                    'current_utilization': current_row.get('util_smooth_14d', 0),
                    'current_deposit_ratio': current_row.get('deposit_loan_ratio', 0),
                    'is_recent': current_date >= recent_cutoff
                })
                
                persona_assignments.append({
                    'company_id': company,
                    'date': current_date,
                    'persona': persona_name,
                    'confidence': confidence,
                    'risk_level': risk_level,
                    'is_recent': current_date >= recent_cutoff
                })
    
    if risk_records:
        risk_df = pd.DataFrame(risk_records)
        persona_df = pd.DataFrame(persona_assignments)
        
        recent_risks = risk_df[risk_df['is_recent'] == True].copy()
        recent_risk_summary = create_business_risk_summary(recent_risks)
        
        return risk_df, persona_df, recent_risk_summary
    else:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

def create_business_risk_summary(recent_risks):
    """Create executive summary of current risk alerts for daily monitoring"""
    if recent_risks.empty:
        return pd.DataFrame()
    
    summary_records = []
    
    for company in recent_risks['company_id'].unique():
        company_risks = recent_risks[recent_risks['company_id'] == company]
        latest_risk = company_risks.sort_values('date').iloc[-1]
        
        persona_info = PERSONA_DEFINITIONS.get(latest_risk['persona'], {})
        
        summary_records.append({
            'company_id': company,
            'risk_level': latest_risk['risk_level'],
            'persona': latest_risk['persona'],
            'persona_description': persona_info.get('description', ''),
            'confidence': latest_risk['confidence'],
            'action_required': latest_risk['action_required'],
            'current_utilization': latest_risk['current_utilization'],
            'signal_quality': latest_risk['signal_quality'],
            'triggered_rules': latest_risk['triggered_rules'],
            'typical_lead_time': persona_info.get('typical_lead_time', 'Unknown'),
            'last_updated': latest_risk['date']
        })
    
    summary_df = pd.DataFrame(summary_records)
    
    # Sort by risk level and confidence for prioritization
    risk_order = {'high': 3, 'medium': 2, 'low': 1}
    summary_df['risk_score'] = summary_df['risk_level'].map(risk_order)
    summary_df = summary_df.sort_values(['risk_score', 'confidence'], ascending=[False, False])
    
    return summary_df.drop('risk_score', axis=1)

#######################################################
# COMPREHENSIVE VISUALIZATION FUNCTIONS (RESTORED & ENHANCED)
#######################################################

def plot_clustering_validation_metrics(validation_metrics):
    """
    BUSINESS PURPOSE: Visualize cluster validation metrics to show optimal number of clusters
    
    This plot helps credit managers understand why a specific number of clusters was chosen
    by showing how different validation metrics perform across different cluster counts.
    """
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    cluster_range = validation_metrics['cluster_range_tested']
    
    # Silhouette Score (higher is better)
    ax1.plot(cluster_range, validation_metrics['all_silhouette_scores'], 'bo-', linewidth=2, markersize=8)
    ax1.axvline(x=validation_metrics['optimal_clusters'], color='red', linestyle='--', alpha=0.7, label='Optimal')
    ax1.set_title('Silhouette Score\n(Higher = Better Cluster Separation)', fontweight='bold')
    ax1.set_xlabel('Number of Clusters')
    ax1.set_ylabel('Silhouette Score')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Calinski-Harabasz Score (higher is better)
    ax2.plot(cluster_range, validation_metrics['all_calinski_scores'], 'go-', linewidth=2, markersize=8)
    ax2.axvline(x=validation_metrics['optimal_clusters'], color='red', linestyle='--', alpha=0.7, label='Optimal')
    ax2.set_title('Calinski-Harabasz Score\n(Higher = Better Cluster Definition)', fontweight='bold')
    ax2.set_xlabel('Number of Clusters')
    ax2.set_ylabel('Calinski-Harabasz Score')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # Davies-Bouldin Score (lower is better)
    ax3.plot(cluster_range, validation_metrics['all_davies_bouldin_scores'], 'ro-', linewidth=2, markersize=8)
    ax3.axvline(x=validation_metrics['optimal_clusters'], color='red', linestyle='--', alpha=0.7, label='Optimal')
    ax3.set_title('Davies-Bouldin Score\n(Lower = Less Cluster Overlap)', fontweight='bold')
    ax3.set_xlabel('Number of Clusters')
    ax3.set_ylabel('Davies-Bouldin Score')
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    
    # Combined optimization summary
    ax4.axis('off')
    summary_text = f"""
CLUSTERING OPTIMIZATION SUMMARY

Optimal Number of Clusters: {validation_metrics['optimal_clusters']}

Final Metrics:
• Silhouette Score: {validation_metrics['silhouette_score']:.3f}
• Calinski-Harabasz: {validation_metrics['calinski_harabasz_score']:.1f}  
• Davies-Bouldin: {validation_metrics['davies_bouldin_score']:.3f}

Interpretation:
• Silhouette: Measures how well-separated clusters are
• Calinski-Harabasz: Measures cluster compactness vs separation
• Davies-Bouldin: Measures average similarity between clusters

Higher silhouette and Calinski-Harabasz scores are better.
Lower Davies-Bouldin scores are better.
    """
    
    ax4.text(0.1, 0.9, summary_text, transform=ax4.transAxes, fontsize=11,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    plt.suptitle('Cluster Validation Metrics Analysis', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    return fig

def plot_cluster_characteristics(clustering_results):
    """
    BUSINESS PURPOSE: Visualize the characteristics of discovered clusters
    
    This visualization shows credit officers what types of financial behavior
    patterns were discovered by the clustering analysis.
    """
    
    if not clustering_results:
        print("No clustering results to plot")
        return None
    
    cluster_chars = clustering_results['cluster_characteristics']
    
    # Create subplots
    n_clusters = len(cluster_chars)
    cols = min(3, n_clusters)
    rows = (n_clusters + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 4*rows))
    if n_clusters == 1:
        axes = [axes]
    elif rows == 1:
        axes = axes if isinstance(axes, (list, np.ndarray)) else [axes]
    else:
        axes = axes.flatten()
    
    for i, (cluster_name, cluster_info) in enumerate(cluster_chars.items()):
        if i >= len(axes):
            break
        
        ax = axes[i]
        
        # Extract key metrics for visualization
        means = cluster_info['mean_characteristics']
        
        metrics = ['util_smooth_14d', 'util_change_90d', 'deposit_change_90d', 
                  'util_volatility', 'deposit_loan_ratio']
        values = [means.get(metric, 0) for metric in metrics]
        labels = ['Utilization\nLevel', 'Utilization\nChange', 'Deposit\nChange', 
                 'Volatility', 'Deposit\nCoverage']
        
        # Create bar plot
        bars = ax.bar(labels, values, color=plt.cm.viridis(i/n_clusters), alpha=0.7)
        ax.set_title(f'{cluster_name.replace("_", " ").title()}\n'
                    f'{cluster_info["size"]} companies ({cluster_info["percentage"]:.1f}%)',
                    fontweight='bold')
        ax.set_ylabel('Normalized Values')
        ax.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{value:.2f}', ha='center', va='bottom', fontweight='bold')
        
        # Add business description
        description = cluster_info['business_description']
        ax.text(0.5, -0.3, description, transform=ax.transAxes, ha='center',
               fontsize=9, style='italic', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Hide unused subplots
    for j in range(i+1, len(axes)):
        axes[j].axis('off')
    
    plt.suptitle('Discovered Cluster Characteristics', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    return fig

def plot_cluster_vs_persona_comparison(comparison_summary):
    """
    BUSINESS PURPOSE: Compare discovered clusters with rule-based personas
    
    This visualization shows how well our business rules align with natural
    data patterns, helping validate and improve our risk assessment approach.
    """
    
    if not comparison_summary:
        print("No comparison data to plot")
        return None
    
    cluster_persona_analysis = comparison_summary['cluster_persona_analysis']
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # Plot 1: Cluster composition by persona
    cluster_names = []
    all_personas = set()
    
    # First pass: collect all cluster names and personas
    for cluster_name, cluster_info in cluster_persona_analysis.items():
        cluster_names.append(cluster_name.replace('_', ' ').title())
        persona_dist = cluster_info['persona_distribution']
        all_personas.update(persona_dist.keys())
    
    # Second pass: build complete persona_data matrix
    persona_data = {}
    for persona in all_personas:
        persona_data[persona] = []
        
        for cluster_name, cluster_info in cluster_persona_analysis.items():
            persona_dist = cluster_info['persona_distribution']
            count = persona_dist.get(persona, 0)  # Use get() with default 0
            persona_data[persona].append(count)
    
    # Verify all arrays have the same length
    n_clusters = len(cluster_names)
    for persona, counts in persona_data.items():
        if len(counts) != n_clusters:
            print(f"Warning: Persona {persona} has {len(counts)} values but {n_clusters} clusters expected")
            # Pad with zeros if needed
            while len(counts) < n_clusters:
                counts.append(0)
            # Truncate if too long
            persona_data[persona] = counts[:n_clusters]
    
    # Create stacked bar chart only if we have valid data
    if len(cluster_names) > 0 and len(persona_data) > 0:
        bottom = np.zeros(len(cluster_names))
        colors = plt.cm.Set3(np.linspace(0, 1, len(persona_data)))
        
        for i, (persona, counts) in enumerate(persona_data.items()):
            # Convert to numpy array to ensure compatibility
            counts_array = np.array(counts)
            ax1.bar(cluster_names, counts_array, bottom=bottom, 
                   label=persona.replace('_', ' ').title(),
                   color=colors[i], alpha=0.8)
            bottom += counts_array
    
    ax1.set_title('Cluster Composition by Persona', fontweight='bold')
    ax1.set_xlabel('Discovered Clusters')
    ax1.set_ylabel('Number of Companies')
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Risk level distribution by cluster
    risk_level_data = {'high': [], 'medium': [], 'low': []}
    
    for cluster_name, cluster_info in cluster_persona_analysis.items():
        risk_dist = cluster_info['risk_level_distribution']
        
        for risk_level in ['high', 'medium', 'low']:
            risk_level_data[risk_level].append(risk_dist.get(risk_level, 0))
    
    # Create stacked bar chart for risk levels
    bottom = np.zeros(len(cluster_names))
    risk_colors = {'high': 'red', 'medium': 'orange', 'low': 'green'}
    
    for risk_level, counts in risk_level_data.items():
        ax2.bar(cluster_names, counts, bottom=bottom, label=f'{risk_level.title()} Risk',
               color=risk_colors[risk_level], alpha=0.7)
        bottom += counts
    
    ax2.set_title('Risk Level Distribution by Cluster', fontweight='bold')
    ax2.set_xlabel('Discovered Clusters')
    ax2.set_ylabel('Number of Companies')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Add alignment metrics as text
    alignment_metrics = comparison_summary['alignment_metrics']
    metrics_text = f"""
CLUSTER-PERSONA ALIGNMENT METRICS

Average Alignment Score: {alignment_metrics['average_alignment']:.2f}
(Range: 0.0 = Poor alignment, 1.0 = Perfect alignment)

Persona Entropy: {alignment_metrics['persona_entropy']:.2f}
Risk Level Entropy: {alignment_metrics['risk_level_entropy']:.2f}
(Lower entropy = better clustering)

Interpretation:
• High alignment score indicates clusters match business rules well
• Low entropy indicates clusters have consistent risk profiles
    """
    
    fig.text(0.02, 0.02, metrics_text, fontsize=10,
            bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    
    plt.suptitle('Cluster vs Persona Comparison Analysis', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    return fig

def plot_risk_persona_distribution(persona_df, recent_risk_summary):
    """
    BUSINESS PURPOSE: Show distribution of risk personas for management reporting
    
    This visualization provides an executive overview of the current risk landscape
    across the portfolio, showing both historical trends and current status.
    """
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # Plot 1: Current risk level distribution
    if not recent_risk_summary.empty:
        risk_counts = recent_risk_summary['risk_level'].value_counts()
        colors = {'high': 'red', 'medium': 'orange', 'low': 'green'}
        plot_colors = [colors.get(level, 'gray') for level in risk_counts.index]
        
        wedges, texts, autotexts = ax1.pie(risk_counts.values, labels=risk_counts.index, 
                                          autopct='%1.1f%%', colors=plot_colors, startangle=90)
        ax1.set_title('Current Risk Level Distribution', fontweight='bold')
        
        # Add count annotations
        for i, (wedge, count) in enumerate(zip(wedges, risk_counts.values)):
            angle = (wedge.theta2 + wedge.theta1) / 2
            x = 1.3 * np.cos(np.radians(angle))
            y = 1.3 * np.sin(np.radians(angle))
            ax1.annotate(f'({count} companies)', xy=(x, y), ha='center', fontweight='bold')
    else:
        ax1.text(0.5, 0.5, 'No current risk alerts', ha='center', va='center', 
                transform=ax1.transAxes, fontsize=14)
        ax1.set_title('Current Risk Level Distribution', fontweight='bold')
    
    # Plot 2: Top personas by frequency
    if not persona_df.empty:
        recent_personas = persona_df[persona_df['is_recent'] == True]
        persona_counts = recent_personas['persona'].value_counts().head(10)
        
        bars = ax2.barh(range(len(persona_counts)), persona_counts.values, 
                       color=plt.cm.viridis(np.linspace(0, 1, len(persona_counts))))
        ax2.set_yticks(range(len(persona_counts)))
        ax2.set_yticklabels([p.replace('_', ' ').title() for p in persona_counts.index])
        ax2.set_xlabel('Number of Companies')
        ax2.set_title('Top 10 Active Personas', fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        # Add count labels
        for i, (bar, count) in enumerate(zip(bars, persona_counts.values)):
            ax2.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height()/2,
                    str(count), va='center', fontweight='bold')
    
    # Plot 3: Confidence distribution
    if not recent_risk_summary.empty:
        confidence_bins = pd.cut(recent_risk_summary['confidence'], 
                               bins=[0, 0.5, 0.7, 0.85, 1.0], 
                               labels=['Low (50-70%)', 'Medium (70-85%)', 'High (85%+)', 'Very High (95%+)'])
        confidence_counts = confidence_bins.value_counts()
        
        ax3.bar(range(len(confidence_counts)), confidence_counts.values,
               color=['lightcoral', 'gold', 'lightgreen', 'darkgreen'])
        ax3.set_xticks(range(len(confidence_counts)))
        ax3.set_xticklabels(confidence_counts.index, rotation=45)
        ax3.set_ylabel('Number of Alerts')
        ax3.set_title('Alert Confidence Distribution', fontweight='bold')
        ax3.grid(True, alpha=0.3)
        
        # Add percentage labels
        total = confidence_counts.sum()
        for i, count in enumerate(confidence_counts.values):
            percentage = count / total * 100
            ax3.text(i, count + 0.5, f'{percentage:.1f}%', ha='center', fontweight='bold')
    
    # Plot 4: Risk level vs confidence scatter
    if not recent_risk_summary.empty:
        risk_level_map = {'high': 3, 'medium': 2, 'low': 1}
        x = recent_risk_summary['confidence']
        y = recent_risk_summary['risk_level'].map(risk_level_map)
        colors = recent_risk_summary['risk_level'].map({'high': 'red', 'medium': 'orange', 'low': 'green'})
        
        scatter = ax4.scatter(x, y, c=colors, alpha=0.6, s=60)
        ax4.set_xlabel('Confidence Level')
        ax4.set_ylabel('Risk Level')
        ax4.set_yticks([1, 2, 3])
        ax4.set_yticklabels(['Low', 'Medium', 'High'])
        ax4.set_title('Risk Level vs Confidence', fontweight='bold')
        ax4.grid(True, alpha=0.3)
        
        # Add trend line
        if len(x) > 1:
            z = np.polyfit(x, y, 1)
            p = np.poly1d(z)
            ax4.plot(x, p(x), "r--", alpha=0.8, linewidth=2)
    
    plt.suptitle('Risk Portfolio Analysis Dashboard', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    return fig

def plot_company_risk_timeline(df_with_metrics, company_id, risk_df):
    """
    BUSINESS PURPOSE: Show detailed risk evolution for a specific company
    
    This visualization helps credit officers understand how a company's risk
    profile has evolved over time and what specific factors drove risk alerts.
    """
    
    company_data = df_with_metrics[df_with_metrics['company_id'] == company_id].sort_values('date')
    company_risks = risk_df[risk_df['company_id'] == company_id].sort_values('date')
    
    if company_data.empty:
        print(f"No data found for company {company_id}")
        return None
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    dates = company_data['date']
    
    # Plot 1: Utilization trend with smoothing
    ax1.plot(dates, company_data['loan_utilization'], alpha=0.3, color='lightblue', label='Raw Data')
    ax1.plot(dates, company_data['util_smooth_14d'], color='blue', linewidth=2, label='14-day Smoothed')
    ax1.plot(dates, company_data['util_smooth_30d'], color='darkblue', linewidth=2, label='30-day Smoothed')
    
    # Add risk level thresholds
    ax1.axhline(y=0.9, color='red', linestyle='--', alpha=0.7, label='Critical (90%)')
    ax1.axhline(y=0.75, color='orange', linestyle='--', alpha=0.7, label='High (75%)')
    ax1.axhline(y=0.6, color='yellow', linestyle='--', alpha=0.7, label='Moderate (60%)')
    
    ax1.set_title(f'Loan Utilization Trend - {company_id}', fontweight='bold')
    ax1.set_ylabel('Utilization Rate')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.tick_params(axis='x', rotation=45)
    
    # Plot 2: Deposit balance trend
    ax2.plot(dates, company_data['deposit_balance'], alpha=0.3, color='lightgreen', label='Raw Data')
    ax2.plot(dates, company_data['deposit_smooth_14d'], color='green', linewidth=2, label='14-day Smoothed')
    ax2.plot(dates, company_data['deposit_smooth_30d'], color='darkgreen', linewidth=2, label='30-day Smoothed')
    
    ax2.set_title(f'Deposit Balance Trend - {company_id}', fontweight='bold')
    ax2.set_ylabel('Deposit Balance')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.tick_params(axis='x', rotation=45)
    
    # Plot 3: Risk alerts timeline
    if not company_risks.empty:
        risk_dates = company_risks['date']
        risk_levels_numeric = company_risks['risk_level'].map({'high': 3, 'medium': 2, 'low': 1})
        risk_colors = company_risks['risk_level'].map({'high': 'red', 'medium': 'orange', 'low': 'green'})
        
        scatter = ax3.scatter(risk_dates, risk_levels_numeric, c=risk_colors, s=100, alpha=0.7)
        
        # Add personas as text annotations
        for i, (date, risk_level, persona) in enumerate(zip(risk_dates, risk_levels_numeric, company_risks['persona'])):
            if i % 3 == 0:  # Annotate every 3rd point to avoid crowding
                ax3.annotate(persona.replace('_', ' ').title()[:15], 
                           (date, risk_level), xytext=(5, 5), textcoords='offset points',
                           fontsize=8, ha='left')
        
        ax3.set_yticks([1, 2, 3])
        ax3.set_yticklabels(['Low', 'Medium', 'High'])
        ax3.set_title('Risk Alert Timeline', fontweight='bold')
        ax3.set_ylabel('Risk Level')
        ax3.grid(True, alpha=0.3)
        ax3.tick_params(axis='x', rotation=45)
    else:
        ax3.text(0.5, 0.5, 'No risk alerts for this company', ha='center', va='center',
                transform=ax3.transAxes, fontsize=12)
        ax3.set_title('Risk Alert Timeline', fontweight='bold')
    
    # Plot 4: Signal quality and volatility
    if 'signal_quality' in company_data.columns and 'util_volatility' in company_data.columns:
        ax4_twin = ax4.twinx()
        
        line1 = ax4.plot(dates, company_data['signal_quality'], color='blue', linewidth=2, label='Signal Quality')
        line2 = ax4_twin.plot(dates, company_data['util_volatility'], color='red', linewidth=2, label='Utilization Volatility')
        
        ax4.set_ylabel('Signal Quality', color='blue')
        ax4_twin.set_ylabel('Volatility', color='red')
        ax4.set_title('Data Quality Metrics', fontweight='bold')
        
        # Add threshold lines
        ax4.axhline(y=CONFIG['business_thresholds']['signal_quality_acceptable'], 
                   color='blue', linestyle='--', alpha=0.5, label='Quality Threshold')
        
        # Combine legends
        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        ax4.legend(lines, labels, loc='upper left')
        
        ax4.grid(True, alpha=0.3)
        ax4.tick_params(axis='x', rotation=45)
    
    plt.suptitle(f'Detailed Risk Analysis - {company_id}', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    return fig

#######################################################
# CREDIT SCORE BACKTEST FRAMEWORK (RESTORED & ENHANCED)
#######################################################

def convert_credit_score_to_numeric(score):
    """Convert credit score to numeric value for comparison (higher = better credit)"""
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
    BUSINESS PURPOSE: Identify credit score downgrades for backtest validation
    
    This function identifies companies that experienced credit downgrades, which
    we'll use to validate whether our risk personas were predictive of actual
    credit deterioration.
    """
    
    end_date = credit_score_df['date'].max()
    start_date = end_date - pd.Timedelta(days=365*lookback_years)
    
    recent_data = credit_score_df[credit_score_df['date'] >= start_date].copy()
    
    downgrades = []
    
    for company_id, company_data in tqdm(recent_data.groupby('company_id'), desc="Detecting downgrades"):
        company_data = company_data.sort_values('date')
        company_data['numeric_score'] = company_data['credit_score'].apply(convert_credit_score_to_numeric)
        
        if len(company_data) < 2:
            continue
        
        for i in range(1, len(company_data)):
            prev_score = company_data.iloc[i-1]['credit_score']
            curr_score = company_data.iloc[i]['credit_score']
            prev_numeric = company_data.iloc[i-1]['numeric_score']
            curr_numeric = company_data.iloc[i]['numeric_score']
            
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

def backtest_personas_vs_downgrades(persona_df, risk_df, downgrade_df, lookback_months=6):
    """
    BUSINESS PURPOSE: Validate persona predictions against actual credit downgrades
    
    This function measures how well our risk personas predicted actual credit
    downgrades, providing validation of our risk assessment methodology.
    """
    
    end_date = max(
        persona_df['date'].max() if not persona_df.empty else datetime.now(),
        downgrade_df['downgrade_date'].max() if not downgrade_df.empty else datetime.now()
    )
    start_date = end_date - pd.Timedelta(days=30*lookback_months)
    
    recent_persona_df = persona_df[persona_df['date'] >= start_date].copy()
    recent_risk_df = risk_df[risk_df['date'] >= start_date].copy()
    recent_downgrade_df = downgrade_df[downgrade_df['downgrade_date'] >= start_date].copy()
    
    common_companies = set(recent_persona_df['company_id'].unique()).intersection(
        set(recent_downgrade_df['company_id'].unique())
    )
    
    print(f"Found {len(common_companies)} companies with both persona data and downgrades")
    
    if not common_companies:
        return pd.DataFrame(), pd.DataFrame()
    
    timing_data = []
    
    # Define risky personas for prediction validation
    risky_personas = [
        'financial_distress', 'approaching_credit_limit', 'cash_flow_crisis', 'erratic_financial_behavior',
        'deteriorating_trends', 'credit_dependency_increasing', 'volatile_utilization'
    ]
    
    for company_id in tqdm(common_companies, desc="Back-testing personas"):
        company_downgrades = recent_downgrade_df[recent_downgrade_df['company_id'] == company_id].sort_values('downgrade_date')
        company_personas = recent_persona_df[recent_persona_df['company_id'] == company_id].sort_values('date')
        company_risks = recent_risk_df[recent_risk_df['company_id'] == company_id].sort_values('date')
        
        for _, downgrade in company_downgrades.iterrows():
            downgrade_date = downgrade['downgrade_date']
            
            # Find most recent persona before downgrade
            before_personas = company_personas[company_personas['date'] < downgrade_date]
            
            if not before_personas.empty:
                latest_before = before_personas.iloc[-1]
                before_persona = latest_before['persona']
                before_date = latest_before['date']
                before_confidence = latest_before['confidence']
                
                days_before = (downgrade_date - before_date).days
                correctly_flagged = before_persona in risky_personas
                
                # Find risk level before downgrade
                risk_before = None
                if not company_risks.empty:
                    risk_before_data = company_risks[company_risks['date'] < downgrade_date]
                    if not risk_before_data.empty:
                        risk_before = risk_before_data.iloc[-1]['risk_level']
                
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
                    'correctly_flagged': correctly_flagged,
                    'risk_level_before': risk_before,
                    'industry': downgrade['industry']
                })
    
    timing_df = pd.DataFrame(timing_data)
    
    # Calculate performance metrics
    if not timing_df.empty:
        performance_data = []
        
        for persona in recent_persona_df['persona'].unique():
            persona_downgrades = timing_df[timing_df['before_persona'] == persona]
            
            if len(persona_downgrades) == 0:
                continue
            
            true_positives = persona_downgrades['correctly_flagged'].sum()
            total_instances = len(persona_downgrades)
            is_risky_persona = persona in risky_personas
            avg_days_before = persona_downgrades['days_before'].mean()
            avg_severity = persona_downgrades['downgrade_severity'].mean()
            
            high_conf_downgrades = persona_downgrades[persona_downgrades['before_confidence'] >= 0.7]
            high_conf_precision = (high_conf_downgrades['correctly_flagged'].sum() / len(high_conf_downgrades) 
                                 if len(high_conf_downgrades) > 0 else 0)
            
            performance_data.append({
                'persona': persona,
                'total_downgrades': total_instances,
                'correctly_flagged': true_positives,
                'precision': true_positives / total_instances,
                'is_risky_persona': is_risky_persona,
                'avg_days_before_downgrade': avg_days_before,
                'avg_severity': avg_severity,
                'high_conf_precision': high_conf_precision
            })
        
        performance_df = pd.DataFrame(performance_data)
        
        # Add overall statistics
        all_downgrades = len(timing_df)
        correctly_flagged = timing_df['correctly_flagged'].sum()
        
        overall_stats = {
            'persona': 'OVERALL',
            'total_downgrades': all_downgrades,
            'correctly_flagged': correctly_flagged,
            'precision': correctly_flagged / all_downgrades if all_downgrades > 0 else 0,
            'is_risky_persona': None,
            'avg_days_before_downgrade': timing_df['days_before'].mean(),
            'avg_severity': timing_df['downgrade_severity'].mean(),
            'high_conf_precision': timing_df[timing_df['before_confidence'] >= 0.7]['correctly_flagged'].mean() 
                                  if not timing_df[timing_df['before_confidence'] >= 0.7].empty else 0
        }
        
        performance_df = pd.concat([performance_df, pd.DataFrame([overall_stats])], ignore_index=True)
        performance_df = performance_df.sort_values('total_downgrades', ascending=False)
    else:
        performance_df = pd.DataFrame()
    
    return performance_df, timing_df

def plot_backtest_performance(performance_df, timing_df):
    """
    BUSINESS PURPOSE: Visualize backtest results to validate persona effectiveness
    
    This visualization shows credit managers how well our risk personas predicted
    actual credit downgrades, providing confidence in the risk assessment system.
    """
    
    if performance_df.empty:
        print("No backtest performance data to plot")
        return None
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # Filter out overall row for detailed plots
    performance_detail = performance_df[performance_df['persona'] != 'OVERALL'].copy()
    
    # Plot 1: Precision by persona
    if not performance_detail.empty:
        precision_sorted = performance_detail.sort_values('precision', ascending=True).tail(10)
        colors = ['red' if risky else 'blue' for risky in precision_sorted['is_risky_persona']]
        
        bars = ax1.barh(range(len(precision_sorted)), precision_sorted['precision']*100, color=colors, alpha=0.7)
        ax1.set_yticks(range(len(precision_sorted)))
        ax1.set_yticklabels([p.replace('_', ' ').title() for p in precision_sorted['persona']])
        ax1.set_xlabel('Precision (%)')
        ax1.set_title('Persona Prediction Precision\n(Red = Risk Personas, Blue = Others)', fontweight='bold')
        ax1.grid(True, alpha=0.3)
        
        # Add precision values
        for i, (bar, precision) in enumerate(zip(bars, precision_sorted['precision'])):
            ax1.text(precision*100 + 1, bar.get_y() + bar.get_height()/2,
                    f'{precision:.1%}', va='center', fontweight='bold')
    
    # Plot 2: Lead time distribution
    if not timing_df.empty:
        ax2.hist(timing_df['days_before'], bins=20, alpha=0.7, color='skyblue', edgecolor='black')
        ax2.axvline(timing_df['days_before'].mean(), color='red', linestyle='--', 
                   label=f'Mean: {timing_df["days_before"].mean():.1f} days')
        ax2.axvline(timing_df['days_before'].median(), color='green', linestyle='--',
                   label=f'Median: {timing_df["days_before"].median():.1f} days')
        ax2.set_xlabel('Days Before Downgrade')
        ax2.set_ylabel('Frequency')
        ax2.set_title('Lead Time Distribution', fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
    
    # Plot 3: Severity vs Lead Time
    if not timing_df.empty:
        colors = ['red' if flagged else 'blue' for flagged in timing_df['correctly_flagged']]
        scatter = ax3.scatter(timing_df['days_before'], timing_df['downgrade_severity'], 
                            c=colors, alpha=0.6, s=50)
        ax3.set_xlabel('Days Before Downgrade')
        ax3.set_ylabel('Downgrade Severity')
        ax3.set_title('Severity vs Lead Time\n(Red = Correctly Flagged, Blue = Missed)', fontweight='bold')
        ax3.grid(True, alpha=0.3)
        
        # Add trend line
        if len(timing_df) > 1:
            z = np.polyfit(timing_df['days_before'], timing_df['downgrade_severity'], 1)
            p = np.poly1d(z)
            ax3.plot(timing_df['days_before'], p(timing_df['days_before']), "r--", alpha=0.8)
    
    # Plot 4: Performance summary
    ax4.axis('off')
    
    if 'OVERALL' in performance_df['persona'].values:
        overall = performance_df[performance_df['persona'] == 'OVERALL'].iloc[0]
        
        summary_text = f"""
BACKTEST PERFORMANCE SUMMARY

Total Downgrades Analyzed: {overall['total_downgrades']:.0f}
Correctly Predicted: {overall['correctly_flagged']:.0f}
Overall Precision: {overall['precision']:.1%}

Average Lead Time: {overall['avg_days_before_downgrade']:.1f} days
Average Severity: {overall['avg_severity']:.1f} points
High-Confidence Precision: {overall['high_conf_precision']:.1%}

Model Effectiveness:
• Early Warning Capability: {overall['avg_days_before_downgrade']:.0f} days average notice
• Risk Identification: {overall['precision']:.1%} of downgrades predicted
• High-Quality Alerts: {overall['high_conf_precision']:.1%} precision for confident predictions

Recommendations:
• Focus on high-confidence alerts for immediate action
• Use medium-confidence alerts for enhanced monitoring
• Continue refining personas based on backtest insights
        """
        
        ax4.text(0.1, 0.9, summary_text, transform=ax4.transAxes, fontsize=11,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
    
    plt.suptitle('Persona Backtest Performance Analysis', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    return fig

def generate_synthetic_credit_score_data(company_ids, start_date=None, end_date=None, random_seed=42):
    """
    BUSINESS PURPOSE: Generate realistic credit score data for backtest validation
    
    This function creates synthetic credit scores that follow realistic patterns,
    including gradual deterioration for some companies and occasional downgrades
    that align with the banking data patterns.
    """
    
    np.random.seed(random_seed)
    
    if end_date is None:
        end_date = datetime.now()
    if start_date is None:
        start_date = end_date - timedelta(days=365*2)
    
    # Define credit scores and industries
    credit_scores = ['1+', '1', '2+', '2', '2-', '3+', '3', '3-', '4+', '4', 
                     '4-', '5+', '5', '5-', '6+', '6', '6-', '7', '8', '9', '10', 'NC', 'NR']
    
    industries = ['Manufacturing', 'Retail', 'Technology', 'Healthcare', 'Financial Services',
                 'Energy', 'Telecommunications', 'Construction', 'Transportation', 'Agriculture',
                 'Real Estate', 'Education', 'Hospitality', 'Media', 'Consulting']
    
    credit_score_data = []
    
    for company_id in company_ids:
        # Assign industry
        industry = np.random.choice(industries)
        
        # Determine if this company will have a downgrade
        has_downgrade = np.random.random() < 0.6  # 60% chance of downgrades
        
        # Choose initial credit score (weighted toward better scores)
        initial_score_idx = min(int(np.random.exponential(scale=7)), len(credit_scores) - 3)
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
        
        else:
            # Add some random intermediate scores for companies without downgrades
            num_updates = np.random.randint(1, 5)
            
            for _ in range(num_updates):
                update_date = start_date + timedelta(days=np.random.randint(30, 730))
                
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
    
    return pd.DataFrame(credit_score_data)

#######################################################
# DATA GENERATION AND MAIN WORKFLOW
#######################################################

def clean_data(df, min_nonzero_pct=0.8):
    """Clean and prepare data for analysis with enhanced validation."""
    print(f"Original data shape: {df.shape}")
    
    # Basic data validation and cleaning
    company_stats = {}
    for company in df['company_id'].unique():
        company_data = df[df['company_id'] == company]
        
        deposit_nonzero = (company_data['deposit_balance'] > 0).mean()
        used_loan_nonzero = (company_data['used_loan'] > 0).mean()
        
        company_stats[company] = {
            'deposit_nonzero': deposit_nonzero,
            'used_loan_nonzero': used_loan_nonzero
        }
    
    # Filter companies with sufficient data
    valid_companies = [
        company for company, stats in company_stats.items()
        if stats['deposit_nonzero'] >= min_nonzero_pct or stats['used_loan_nonzero'] >= min_nonzero_pct
    ]
    
    df_clean = df[df['company_id'].isin(valid_companies)].copy()
    print(f"Retained {len(valid_companies)}/{len(df['company_id'].unique())} companies")
    
    return df_clean, df_clean.copy()

def generate_realistic_test_data(num_companies=30, days=730):
    """Generate realistic test data with various risk patterns for demonstration."""
    print("Generating realistic test data with embedded risk patterns...")
    
    np.random.seed(42)
    
    end_date = pd.Timestamp('2024-12-31')
    start_date = end_date - pd.Timedelta(days=days)
    date_range = pd.date_range(start=start_date, end=end_date)
    
    company_ids = [f'COMP{str(i).zfill(4)}' for i in range(num_companies)]
    
    data = []
    
    # Define company archetypes for testing
    risk_archetypes = {
        'stable': 0.4,           # 40% stable companies
        'deteriorating': 0.2,    # 20% deteriorating
        'volatile': 0.15,        # 15% volatile  
        'seasonal': 0.15,        # 15% seasonal
        'improving': 0.1         # 10% improving
    }
    
    archetype_assignments = []
    for archetype, proportion in risk_archetypes.items():
        count = int(num_companies * proportion)
        archetype_assignments.extend([archetype] * count)
    
    # Fill remainder with stable
    while len(archetype_assignments) < num_companies:
        archetype_assignments.append('stable')
    
    np.random.shuffle(archetype_assignments)
    
    for i, company_id in enumerate(company_ids):
        archetype = archetype_assignments[i]
        
        # Base parameters
        base_deposit = np.random.lognormal(10, 0.8)  # $100K-$1M range typically
        base_loan = np.random.lognormal(9, 0.8)     # $50K-$500K range typically
        initial_util = np.random.uniform(0.3, 0.7)
        
        # Generate data based on archetype
        for j, date in enumerate(date_range):
            t = j / len(date_range)  # Normalized time
            
            if archetype == 'stable':
                # Stable companies - low volatility, consistent patterns
                util_noise = np.random.normal(0, 0.02)
                deposit_noise = np.random.normal(1, 0.05)
                utilization = np.clip(initial_util + util_noise, 0.1, 0.9)
                deposit_multiplier = deposit_noise
                
            elif archetype == 'deteriorating':
                # Companies developing problems over time
                deterioration_start = 0.6  # Problems start 60% through timeline
                if t > deterioration_start:
                    # Accelerating problems
                    problem_severity = (t - deterioration_start) / (1 - deterioration_start)
                    util_increase = problem_severity * 0.4  # Up to 40% increase
                    deposit_decrease = problem_severity * 0.5  # Up to 50% decrease
                else:
                    util_increase = 0
                    deposit_decrease = 0
                
                utilization = np.clip(initial_util + util_increase + np.random.normal(0, 0.03), 0.1, 0.95)
                deposit_multiplier = (1 - deposit_decrease) * np.random.normal(1, 0.08)
                
            elif archetype == 'volatile':
                # High volatility companies
                util_noise = np.random.normal(0, 0.08)  # High volatility
                deposit_noise = np.random.normal(1, 0.15)
                # Add occasional large swings
                if np.random.random() < 0.05:  # 5% chance of large swing
                    util_noise += np.random.choice([-1, 1]) * np.random.uniform(0.1, 0.2)
                    deposit_noise *= np.random.uniform(0.7, 1.4)
                
                utilization = np.clip(initial_util + util_noise, 0.1, 0.95)
                deposit_multiplier = deposit_noise
                
            elif archetype == 'seasonal':
                # Seasonal patterns
                seasonal_util = 0.15 * np.sin(2 * np.pi * j / 365)  # Annual cycle
                seasonal_deposit = 0.25 * np.sin(2 * np.pi * j / 365 + np.pi)  # Opposite phase
                
                utilization = np.clip(initial_util + seasonal_util + np.random.normal(0, 0.02), 0.1, 0.9)
                deposit_multiplier = (1 + seasonal_deposit) * np.random.normal(1, 0.05)
                
            elif archetype == 'improving':
                # Companies getting better over time
                improvement = t * 0.3  # Gradual improvement
                utilization = np.clip(initial_util - improvement + np.random.normal(0, 0.02), 0.1, 0.8)
                deposit_multiplier = (1 + improvement) * np.random.normal(1, 0.04)
            
            # Calculate final values
            deposit = base_deposit * deposit_multiplier
            used_loan = base_loan * utilization
            unused_loan = base_loan - used_loan
            
            # Add realistic data issues
            if np.random.random() < 0.03:  # 3% chance of data issues
                if np.random.random() < 0.5:
                    deposit = 0  # System downtime
                else:
                    used_loan = 0
                    unused_loan = 0
            
            data.append({
                'company_id': company_id,
                'date': date,
                'deposit_balance': max(0, deposit),
                'used_loan': max(0, used_loan),
                'unused_loan': max(0, unused_loan),
                'archetype': archetype  # For validation purposes
            })
    
    df = pd.DataFrame(data)
    print(f"Generated {len(df)} records for {num_companies} companies")
    print("Archetype distribution:")
    print(df.groupby('company_id')['archetype'].first().value_counts())
    
    return df

def print_persona_definitions():
    """Display all persona definitions for credit officer reference"""
    print("=" * 100)
    print("CREDIT RISK PERSONA REFERENCE GUIDE")
    print("=" * 100)
    
    # Group by risk level
    for risk_level in ['high', 'medium', 'low']:
        personas_at_level = {k: v for k, v in PERSONA_DEFINITIONS.items() 
                           if v.get('risk_level') == risk_level}
        
        if personas_at_level:
            print(f"\n{risk_level.upper()} RISK PERSONAS")
            print("-" * 50)
            
            for persona_name, persona_info in personas_at_level.items():
                print(f"\n📋 {persona_name.upper().replace('_', ' ')}")
                print(f"   Description: {persona_info['description']}")
                print(f"   Action Required: {persona_info['action_required']}")
                print(f"   Typical Lead Time: {persona_info['typical_lead_time']}")
                print(f"   Business Rules:")
                for rule in persona_info['business_rules']:
                    print(f"     • {rule}")

def generate_risk_report(risk_summary_df, output_path='risk_report.txt'):
    """Generate human-readable risk report for credit officers"""
    if risk_summary_df.empty:
        print("No risk alerts to report.")
        return
    
    report_lines = []
    report_lines.append("=" * 80)
    report_lines.append("DAILY CREDIT RISK ALERT REPORT")
    report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append("=" * 80)
    
    # Summary statistics
    total_companies = len(risk_summary_df)
    high_risk = len(risk_summary_df[risk_summary_df['risk_level'] == 'high'])
    medium_risk = len(risk_summary_df[risk_summary_df['risk_level'] == 'medium'])
    low_risk = len(risk_summary_df[risk_summary_df['risk_level'] == 'low'])
    
    report_lines.append(f"\nSUMMARY:")
    report_lines.append(f"Total Companies with Alerts: {total_companies}")
    report_lines.append(f"  High Risk (Immediate Attention): {high_risk}")
    report_lines.append(f"  Medium Risk (Monitor Closely): {medium_risk}")
    report_lines.append(f"  Low Risk (Standard Monitoring): {low_risk}")
    
    # High risk alerts (requiring immediate action)
    if high_risk > 0:
        report_lines.append(f"\n{'='*60}")
        report_lines.append("HIGH RISK ALERTS - IMMEDIATE ACTION REQUIRED")
        report_lines.append(f"{'='*60}")
        
        high_risk_companies = risk_summary_df[risk_summary_df['risk_level'] == 'high']
        
        for idx, company in high_risk_companies.iterrows():
            report_lines.append(f"\nCompany: {company['company_id']}")
            report_lines.append(f"Risk Persona: {company['persona']}")
            report_lines.append(f"Description: {company['persona_description']}")
            report_lines.append(f"Confidence Level: {company['confidence']:.1%}")
            report_lines.append(f"Current Utilization: {company['current_utilization']:.1%}")
            report_lines.append(f"Action Required: {company['action_required']}")
            report_lines.append(f"Expected Timeline: {company['typical_lead_time']}")
            report_lines.append(f"Triggered Rules: {company['triggered_rules']}")
            report_lines.append("-" * 60)
    
    # Write to file and print
    report_text = '\n'.join(report_lines)
    
    try:
        with open(output_path, 'w') as f:
            f.write(report_text)
        print(f"Risk report saved to {output_path}")
    except:
        pass
    
    # Print to console
    print(report_text)

#######################################################
# MAIN COMPREHENSIVE WORKFLOW
#######################################################

def comprehensive_risk_analysis_workflow(df, include_clustering=True, include_backtest=True):
    """
    MAIN COMPREHENSIVE WORKFLOW: Complete risk analysis with clustering, rules, visualizations, and backtest
    
    This is the complete end-to-end workflow that combines:
    1. Enhanced data processing with noise reduction
    2. Optimal clustering for persona discovery  
    3. Rule-based persona assignment
    4. Comprehensive visualizations
    5. Credit score backtest validation
    """
    
    print("🏦 COMPREHENSIVE BANK RISK ANALYSIS SYSTEM")
    print("=" * 80)
    print("Features: Clustering + Business Rules + Visualizations + Backtest Validation")
    print("=" * 80)
    
    results = {}
    
    # PHASE 1: Data Processing and Feature Engineering
    print("\n📊 PHASE 1: DATA PROCESSING & FEATURE ENGINEERING")
    print("-" * 60)
    
    print("Step 1: Cleaning and validating data...")
    df_clean, df_calc = clean_data(df)
    
    print("Step 2: Enhanced feature engineering with noise reduction...")
    df_with_metrics = add_derived_metrics_enhanced(df_clean)
    
    results['data_with_metrics'] = df_with_metrics
    
    # PHASE 2: Clustering Analysis (Optional)
    if include_clustering:
        print("\n🎯 PHASE 2: OPTIMAL CLUSTERING ANALYSIS")
        print("-" * 60)
        
        clustering_results, kmeans_model, validation_metrics = find_optimal_clusters(
            df_with_metrics, CONFIG['clustering']['features_for_clustering']
        )
        
        if clustering_results:
            results['clustering_results'] = clustering_results
            results['clustering_validation'] = validation_metrics
            
            # Generate clustering visualizations
            print("Generating clustering validation plots...")
            cluster_validation_fig = plot_clustering_validation_metrics(validation_metrics)
            cluster_validation_fig.savefig('cluster_validation_metrics.png', dpi=300, bbox_inches='tight')
            
            cluster_chars_fig = plot_cluster_characteristics(clustering_results)
            if cluster_chars_fig:
                cluster_chars_fig.savefig('cluster_characteristics.png', dpi=300, bbox_inches='tight')
            
            print("✅ Clustering analysis complete - visualizations saved")
        else:
            print("⚠️ Clustering analysis skipped - insufficient data")
    
    # PHASE 3: Rule-Based Risk Pattern Detection
    print("\n🚨 PHASE 3: RULE-BASED RISK PATTERN DETECTION")
    print("-" * 60)
    
    print("Detecting risk patterns using business rules...")
    risk_df, persona_df, recent_risk_summary = detect_risk_patterns_with_business_rules(df_with_metrics)
    
    results['risk_alerts'] = risk_df
    results['persona_assignments'] = persona_df
    results['daily_risk_summary'] = recent_risk_summary
    
    if not recent_risk_summary.empty:
        print(f"✅ Risk analysis complete:")
        print(f"   🚨 Total alerts: {len(recent_risk_summary)}")
        print(f"   🔴 High risk: {len(recent_risk_summary[recent_risk_summary['risk_level'] == 'high'])}")
        print(f"   🟡 Medium risk: {len(recent_risk_summary[recent_risk_summary['risk_level'] == 'medium'])}")
        print(f"   🟢 Low risk: {len(recent_risk_summary[recent_risk_summary['risk_level'] == 'low'])}")
        
        # Generate business report
        generate_risk_report(recent_risk_summary)
    else:
        print("✅ No risk alerts generated - all companies within normal parameters")
    
    # PHASE 4: Cluster-Persona Comparison (if clustering was performed)
    if include_clustering and clustering_results and not persona_df.empty:
        print("\n🔍 PHASE 4: CLUSTER-PERSONA COMPARISON")
        print("-" * 60)
        
        comparison_summary = compare_clusters_with_personas(clustering_results, persona_df)
        
        if comparison_summary:
            results['cluster_persona_comparison'] = comparison_summary
            
            # Generate comparison visualization
            comparison_fig = plot_cluster_vs_persona_comparison(comparison_summary)
            if comparison_fig:
                comparison_fig.savefig('cluster_persona_comparison.png', dpi=300, bbox_inches='tight')
            
            print("✅ Cluster-persona comparison complete")
        else:
            print("⚠️ Cluster-persona comparison skipped - insufficient matching data")
    
    # PHASE 5: Comprehensive Visualizations
    print("\n📈 PHASE 5: COMPREHENSIVE VISUALIZATIONS")
    print("-" * 60)
    
    # Risk portfolio dashboard
    portfolio_fig = plot_risk_persona_distribution(persona_df, recent_risk_summary)
    portfolio_fig.savefig('risk_portfolio_dashboard.png', dpi=300, bbox_inches='tight')
    
    # Individual company analysis (for top 3 high-risk companies)
    if not recent_risk_summary.empty:
        high_risk_companies = recent_risk_summary[
            recent_risk_summary['risk_level'] == 'high'
        ]['company_id'].head(3)
        
        for company_id in high_risk_companies:
            company_fig = plot_company_risk_timeline(df_with_metrics, company_id, risk_df)
            if company_fig:
                company_fig.savefig(f'company_analysis_{company_id}.png', dpi=300, bbox_inches='tight')
    
    print("✅ Visualizations generated and saved")
    
    # PHASE 6: Credit Score Backtest (Optional)
    if include_backtest:
        print("\n🔬 PHASE 6: CREDIT SCORE BACKTEST VALIDATION")
        print("-" * 60)
        
        # Generate synthetic credit score data for backtest
        print("Generating synthetic credit score data...")
        company_ids = df_with_metrics['company_id'].unique()
        credit_score_df = generate_synthetic_credit_score_data(
            company_ids=company_ids,
            start_date=df_with_metrics['date'].min(),
            end_date=df_with_metrics['date'].max()
        )
        
        # Detect credit downgrades
        print("Detecting credit downgrades...")
        downgrade_df = detect_credit_downgrades(credit_score_df, lookback_years=1)
        
        if not downgrade_df.empty:
            print(f"Found {len(downgrade_df)} downgrades across {downgrade_df['company_id'].nunique()} companies")
            
            # Backtest personas against downgrades
            print("Running backtest validation...")
            performance_df, timing_df = backtest_personas_vs_downgrades(
                persona_df, risk_df, downgrade_df, lookback_months=6
            )
            
            if not performance_df.empty:
                results['backtest_performance'] = performance_df
                results['backtest_timing'] = timing_df
                results['credit_downgrades'] = downgrade_df
                
                # Generate backtest visualization
                backtest_fig = plot_backtest_performance(performance_df, timing_df)
                if backtest_fig:
                    backtest_fig.savefig('backtest_performance.png', dpi=300, bbox_inches='tight')
                
                # Print backtest summary
                if 'OVERALL' in performance_df['persona'].values:
                    overall = performance_df[performance_df['persona'] == 'OVERALL'].iloc[0]
                    print(f"✅ Backtest validation complete:")
                    print(f"   📊 Downgrades analyzed: {overall['total_downgrades']:.0f}")
                    print(f"   🎯 Correctly predicted: {overall['correctly_flagged']:.0f} ({overall['precision']:.1%})")
                    print(f"   ⏰ Average lead time: {overall['avg_days_before_downgrade']:.1f} days")
                
            else:
                print("⚠️ Backtest validation incomplete - insufficient matching data")
        else:
            print("⚠️ No credit downgrades found for backtest validation")
    
    # PHASE 7: Final Summary and Recommendations
    print("\n📋 PHASE 7: ANALYSIS SUMMARY & RECOMMENDATIONS")
    print("-" * 60)
    
    analysis_summary = {
        'total_companies_analyzed': df['company_id'].nunique(),
        'companies_with_alerts': len(recent_risk_summary) if not recent_risk_summary.empty else 0,
        'high_risk_companies': len(recent_risk_summary[recent_risk_summary['risk_level'] == 'high']) if not recent_risk_summary.empty else 0,
        'clustering_performed': include_clustering and clustering_results is not None,
        'optimal_clusters_found': validation_metrics.get('optimal_clusters', 0) if include_clustering and clustering_results else 0,
        'backtest_performed': include_backtest and 'backtest_performance' in results,
        'backtest_precision': results.get('backtest_performance', pd.DataFrame()).loc[
            results.get('backtest_performance', pd.DataFrame())['persona'] == 'OVERALL', 'precision'
        ].iloc[0] if include_backtest and 'backtest_performance' in results and not results['backtest_performance'].empty else 0,
    }
    
    results['analysis_summary'] = analysis_summary
    
    print("📊 FINAL ANALYSIS SUMMARY:")
    print(f"   Companies Analyzed: {analysis_summary['total_companies_analyzed']}")
    print(f"   Risk Alerts Generated: {analysis_summary['companies_with_alerts']}")
    print(f"   High-Risk Companies: {analysis_summary['high_risk_companies']}")
    
    if analysis_summary['clustering_performed']:
        print(f"   Optimal Clusters Found: {analysis_summary['optimal_clusters_found']}")
    
    if analysis_summary['backtest_performed']:
        print(f"   Backtest Precision: {analysis_summary['backtest_precision']:.1%}")
    
    print("\n🎯 RECOMMENDATIONS:")
    if analysis_summary['high_risk_companies'] > 0:
        print("   • Review high-risk companies immediately")
        print("   • Implement enhanced monitoring for medium-risk companies")
    
    if analysis_summary['clustering_performed']:
        print("   • Use cluster insights to refine business rules")
        print("   • Consider cluster-specific risk thresholds")
    
    if analysis_summary['backtest_performed'] and analysis_summary['backtest_precision'] > 0.6:
        print("   • Model shows good predictive performance - deploy with confidence")
    elif analysis_summary['backtest_performed']:
        print("   • Consider refining persona rules based on backtest insights")
    
    print("\n✅ COMPREHENSIVE ANALYSIS COMPLETE!")
    print("📁 Check generated PNG files for detailed visualizations")
    print("📋 Check 'risk_report.txt' for executive summary")
    
    return results

#######################################################
# MAIN EXECUTION
#######################################################

if __name__ == "__main__":
    
    print("🎯 COMPREHENSIVE BANK RISK ANALYSIS SYSTEM")
    print("Features: Clustering + Business Rules + Visualizations + Backtest")
    print("=" * 80)
    
    # Display persona definitions for reference
    print("\n📚 CREDIT RISK PERSONA REFERENCE GUIDE")
    print_persona_definitions()
    
    print("\n" + "=" * 80)
    print("🧪 RUNNING COMPREHENSIVE DEMONSTRATION")
    print("=" * 80)
    
    # Generate realistic test data
    test_df = generate_realistic_test_data(num_companies=30, days=365)
    
    # Run the comprehensive analysis
    results = comprehensive_risk_analysis_workflow(
        test_df, 
        include_clustering=True, 
        include_backtest=True
    )
    
    print("\n" + "=" * 80)
    print("🎊 DEMONSTRATION COMPLETE!")
    print("=" * 80)
    
    summary = results['analysis_summary']
    print(f"✅ Analyzed {summary['total_companies_analyzed']} companies")
    print(f"🚨 Generated {summary['companies_with_alerts']} risk alerts")
    print(f"🔴 Identified {summary['high_risk_companies']} high-risk companies")
    
    if summary['clustering_performed']:
        print(f"🎯 Discovered {summary['optimal_clusters_found']} optimal clusters")
    
    if summary['backtest_performed']:
        print(f"📈 Achieved {summary['backtest_precision']:.1%} prediction accuracy")
    
    print(f"\n📁 Generated visualization files:")
    print(f"   • risk_portfolio_dashboard.png")
    print(f"   • cluster_validation_metrics.png")
    print(f"   • cluster_characteristics.png") 
    print(f"   • cluster_persona_comparison.png")
    print(f"   • backtest_performance.png")
    print(f"   • company_analysis_[ID].png (for high-risk companies)")
    print(f"   • risk_report.txt (executive summary)")
