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
from sklearn.metrics import silhouette_score
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
# SIMPLIFIED CONFIGURATION
#######################################################

# Simplified configuration focusing on clear, business-actionable personas
SIMPLIFIED_CONFIG = {
    'data': {
        'min_nonzero_pct': 0.8,           # Traditional analysis threshold
        'min_continuous_days': 365,        # Minimum days for traditional analysis
        'recent_window': 90,               # Recent risk window in days
        'new_client_threshold_days': 365,  # Define "new client" as < 1 year
        'change_threshold': 0.02,          # 2% change threshold for "no change"
        'min_history_points': 50,          # Minimum data points for sufficient history
        'noise_threshold': 2.0,            # Standard deviations for noise detection
    },
    'risk': {
        'trend_windows': [30, 45, 60, 90, 120, 180],
        'change_thresholds': {
            'sharp': 0.2,
            'moderate': 0.1,
            'gradual': 0.05
        },
        # Traditional personas (from original system)
        'traditional_personas': {
            'cautious_borrower': 'Low utilization (<40%), stable deposits',
            'aggressive_expansion': 'Rising utilization (>10% increase), volatile deposits',
            'distressed_client': 'High utilization (>80%), declining deposits (>5% decrease)',
            'seasonal_loan_user': 'Cyclical utilization with >15% amplitude',
            'seasonal_deposit_pattern': 'Cyclical deposits with >20% amplitude',
            'deteriorating_health': 'Rising utilization (>15% increase), declining deposits (>10% decrease)',
            'cash_constrained': 'Stable utilization, rapidly declining deposits (>15% decrease)',
            'credit_dependent': 'High utilization (>75%), low deposit ratio (<0.8)',
        },
        # Simplified sparse personas - clear and business-actionable
        'simplified_sparse_personas': {
            'new_client': 'Client relationship less than 1 year old',
            'no_utilization_change': 'Loan utilization has remained essentially unchanged',
            'no_deposit_change': 'Deposit levels have remained essentially unchanged', 
            'insufficient_utilization_history': 'Not enough loan utilization data for analysis',
            'insufficient_deposit_history': 'Not enough deposit data for reliable analysis',
            'noisy_history': 'Data patterns are too irregular for reliable analysis'
        },
        'risk_levels': {
            'high': 3,
            'medium': 2,
            'low': 1,
            'none': 0
        }
    },
    'clustering': {
        'max_clusters': 20,      # Maximum clusters to test
        'min_clusters': 4,       # Minimum clusters to test
        'random_state': 42
    }
}

#######################################################
# ENHANCED DATA CLEANING (SIMPLIFIED VERSION)
#######################################################

def simplified_clean_data(df, min_nonzero_pct=0.8):
    """
    Simplified version of data cleaning that separates companies into:
    1. Traditional analysis candidates (rich data)
    2. Sparse analysis candidates (limited data)
    
    This approach is more straightforward and focuses on clear business logic.
    """
    print(f"Simplified data cleaning - Original data shape: {df.shape}")
    
    # Calculate basic statistics for each company
    company_stats = {}
    for company in df['company_id'].unique():
        company_data = df[df['company_id'] == company].sort_values('date')
        
        # Basic metrics
        total_records = len(company_data)
        date_span = (company_data['date'].max() - company_data['date'].min()).days
        
        # Activity rates (percentage of non-zero values)
        deposit_activity = (company_data['deposit_balance'] > 0).mean()
        loan_activity = (company_data['used_loan'] > 0).mean()
        
        # Overall activity (either deposit or loan activity)
        any_activity = ((company_data['deposit_balance'] > 0) | 
                       (company_data['used_loan'] > 0)).mean()
        
        company_stats[company] = {
            'total_records': total_records,
            'date_span_days': date_span,
            'deposit_activity_rate': deposit_activity,
            'loan_activity_rate': loan_activity,
            'overall_activity_rate': any_activity,
            'data_density': total_records / max(1, date_span) if date_span > 0 else 0
        }
    
    # Simple classification logic
    traditional_companies = []
    sparse_companies = []
    
    for company, stats in company_stats.items():
        # Traditional analysis criteria: high activity rates and sufficient data
        if (stats['deposit_activity_rate'] >= min_nonzero_pct and 
            stats['loan_activity_rate'] >= min_nonzero_pct and
            stats['total_records'] >= SIMPLIFIED_CONFIG['data']['min_history_points']):
            traditional_companies.append(company)
        else:
            # Everything else goes to sparse analysis
            sparse_companies.append(company)
    
    # Create datasets
    df_traditional = df[df['company_id'].isin(traditional_companies)].copy()
    df_sparse = df[df['company_id'].isin(sparse_companies)].copy()
    
    print(f"Traditional analysis companies: {len(traditional_companies)}")
    print(f"Sparse analysis companies: {len(sparse_companies)}")
    print(f"Coverage: {len(traditional_companies) + len(sparse_companies)} / {len(df['company_id'].unique())} companies")
    
    return df_traditional, df_sparse, {
        'traditional_companies': traditional_companies,
        'sparse_companies': sparse_companies,
        'company_stats': company_stats
    }

#######################################################
# SIMPLIFIED SPARSE ANALYSIS WITH CLUSTERING
#######################################################

def calculate_sparse_features(df):
    """
    Calculate simple, interpretable features for sparse data analysis.
    These features directly support the 6 business personas.
    """
    print("Calculating features for sparse data analysis...")
    
    sparse_features = []
    
    # Define thresholds
    new_client_threshold = pd.Timestamp.now() - pd.Timedelta(
        days=SIMPLIFIED_CONFIG['data']['new_client_threshold_days']
    )
    change_threshold = SIMPLIFIED_CONFIG['data']['change_threshold']
    min_history = SIMPLIFIED_CONFIG['data']['min_history_points']
    
    for company_id in tqdm(df['company_id'].unique(), desc="Calculating sparse features"):
        company_data = df[df['company_id'] == company_id].sort_values('date')
        
        if len(company_data) == 0:
            continue
        
        # Feature 1: Is new client (< 1 year)
        first_date = company_data['date'].min()
        is_new_client = first_date >= new_client_threshold
        client_age_days = (pd.Timestamp.now() - first_date).days
        
        # Feature 2: Utilization change analysis
        utilization_records = company_data[company_data['used_loan'] > 0]
        if len(utilization_records) >= 2:
            # Calculate loan utilization where possible
            company_data_copy = company_data.copy()
            company_data_copy['total_loan'] = company_data_copy['used_loan'] + company_data_copy['unused_loan']
            company_data_copy['utilization'] = (
                company_data_copy['used_loan'] / company_data_copy['total_loan']
            ).fillna(0)
            
            # Remove infinite values
            company_data_copy['utilization'] = company_data_copy['utilization'].replace([np.inf, -np.inf], 0)
            
            utilization_values = company_data_copy[company_data_copy['utilization'] > 0]['utilization']
            
            if len(utilization_values) >= 2:
                utilization_change = utilization_values.max() - utilization_values.min()
                utilization_std = utilization_values.std()
                has_utilization_change = utilization_change > change_threshold
                sufficient_utilization_history = len(utilization_values) >= min_history
            else:
                utilization_values = pd.Series(dtype=float)  # Initialize as empty Series
                utilization_change = 0
                utilization_std = 0
                has_utilization_change = False
                sufficient_utilization_history = False
        else:
            utilization_values = pd.Series(dtype=float)  # Initialize as empty Series
            utilization_change = 0
            utilization_std = 0
            has_utilization_change = False
            sufficient_utilization_history = False
        
        # Feature 3: Deposit change analysis
        deposit_records = company_data[company_data['deposit_balance'] > 0]
        if len(deposit_records) >= 2:
            deposit_values = deposit_records['deposit_balance']
            deposit_change_abs = deposit_values.max() - deposit_values.min()
            deposit_change_rel = deposit_change_abs / deposit_values.mean() if deposit_values.mean() > 0 else 0
            deposit_std = deposit_values.std()
            has_deposit_change = deposit_change_rel > change_threshold
            sufficient_deposit_history = len(deposit_values) >= min_history
        else:
            deposit_values = pd.Series(dtype=float)  # Initialize as empty Series
            deposit_change_abs = 0
            deposit_change_rel = 0
            deposit_std = 0
            has_deposit_change = False
            sufficient_deposit_history = False
        
        # Feature 4: Data quality metrics
        total_records = len(company_data)
        non_zero_records = len(company_data[
            (company_data['deposit_balance'] > 0) | (company_data['used_loan'] > 0)
        ])
        data_completeness = non_zero_records / total_records if total_records > 0 else 0
        
        # Feature 5: Noise detection (coefficient of variation)
        # High CV indicates noisy/irregular data
        utilization_cv = utilization_std / utilization_values.mean() if len(utilization_values) > 0 and utilization_values.mean() > 0 else 0
        deposit_cv = deposit_std / deposit_values.mean() if len(deposit_values) > 0 and deposit_values.mean() > 0 else 0
        
        # Determine if data is too noisy
        is_noisy = (utilization_cv > SIMPLIFIED_CONFIG['data']['noise_threshold'] or 
                   deposit_cv > SIMPLIFIED_CONFIG['data']['noise_threshold'])
        
        # Store features
        sparse_features.append({
            'company_id': company_id,
            'is_new_client': is_new_client,
            'client_age_days': client_age_days,
            'has_utilization_change': has_utilization_change,
            'utilization_change': utilization_change,
            'utilization_std': utilization_std,
            'utilization_cv': utilization_cv,
            'sufficient_utilization_history': sufficient_utilization_history,
            'has_deposit_change': has_deposit_change,
            'deposit_change_rel': deposit_change_rel,
            'deposit_std': deposit_std,
            'deposit_cv': deposit_cv,
            'sufficient_deposit_history': sufficient_deposit_history,
            'data_completeness': data_completeness,
            'total_records': total_records,
            'is_noisy': is_noisy,
            # Additional metrics for clustering
            'utilization_records_count': len(utilization_records),
            'deposit_records_count': len(deposit_records),
            'avg_utilization': utilization_values.mean() if len(utilization_values) > 0 else 0,
            'avg_deposit': deposit_values.mean() if len(deposit_values) > 0 else 0,
        })
    
    return pd.DataFrame(sparse_features)

def assign_simplified_sparse_personas(sparse_features_df):
    """
    Assign personas using simple, clear business logic based on the 6 defined categories.
    This approach prioritizes interpretability and business actionability.
    """
    print("Assigning simplified sparse personas...")
    
    persona_assignments = []
    
    for _, row in sparse_features_df.iterrows():
        company_id = row['company_id']
        
        # Initialize persona assignment variables
        persona = None
        risk_level = 'low'
        confidence = 0.0
        reasoning = []
        
        # Rule 1: New client (highest priority - business rule)
        if row['is_new_client']:
            persona = 'new_client'
            confidence = 0.95  # High confidence for clear date-based rule
            risk_level = 'medium'  # New clients need monitoring
            reasoning.append(f"Client relationship is {row['client_age_days']} days old (< 365 days)")
        
        # Rule 2: Insufficient utilization history
        elif not row['sufficient_utilization_history'] and row['utilization_records_count'] > 0:
            persona = 'insufficient_utilization_history'
            confidence = 0.90  # High confidence for data-based rule
            risk_level = 'low'  # Not enough data to assess risk
            reasoning.append(f"Only {row['utilization_records_count']} utilization records (< {SIMPLIFIED_CONFIG['data']['min_history_points']} required)")
        
        # Rule 3: Insufficient deposit history
        elif not row['sufficient_deposit_history'] and row['deposit_records_count'] > 0:
            persona = 'insufficient_deposit_history' 
            confidence = 0.90
            risk_level = 'low'
            reasoning.append(f"Only {row['deposit_records_count']} deposit records (< {SIMPLIFIED_CONFIG['data']['min_history_points']} required)")
        
        # Rule 4: Noisy history (takes precedence over change analysis)
        elif row['is_noisy']:
            persona = 'noisy_history'
            confidence = 0.85
            risk_level = 'medium'  # Noisy data makes risk assessment difficult
            reasoning.append(f"High variability detected (utilization CV: {row['utilization_cv']:.2f}, deposit CV: {row['deposit_cv']:.2f})")
        
        # Rule 5: No utilization change (but has sufficient history)
        elif (row['sufficient_utilization_history'] and not row['has_utilization_change']):
            persona = 'no_utilization_change'
            confidence = 0.80
            risk_level = 'low'  # Stable utilization is generally low risk
            reasoning.append(f"Utilization change of {row['utilization_change']:.3f} is below threshold ({SIMPLIFIED_CONFIG['data']['change_threshold']})")
        
        # Rule 6: No deposit change (but has sufficient history)
        elif (row['sufficient_deposit_history'] and not row['has_deposit_change']):
            persona = 'no_deposit_change'
            confidence = 0.80
            risk_level = 'low'  # Stable deposits are generally low risk
            reasoning.append(f"Deposit change of {row['deposit_change_rel']:.3f} is below threshold ({SIMPLIFIED_CONFIG['data']['change_threshold']})")
        
        # Default case: insufficient data overall
        else:
            # Determine which type of insufficient data
            if row['utilization_records_count'] == 0 and row['deposit_records_count'] == 0:
                persona = 'insufficient_deposit_history'  # Default to deposit since it's more fundamental
                reasoning.append("No meaningful utilization or deposit records found")
            elif row['utilization_records_count'] == 0:
                persona = 'insufficient_utilization_history'
                reasoning.append("No utilization records found")
            else:
                persona = 'insufficient_deposit_history'
                reasoning.append("No deposit records found")
            
            confidence = 0.75
            risk_level = 'low'
        
        # Risk level adjustments based on additional factors
        if row['data_completeness'] < 0.1:  # Very sparse data
            if risk_level == 'low':
                risk_level = 'medium'
            reasoning.append(f"Very low data completeness ({row['data_completeness']:.2f})")
        
        persona_assignments.append({
            'company_id': company_id,
            'persona': persona,
            'risk_level': risk_level,
            'confidence': confidence,
            'reasoning': ' | '.join(reasoning),
            'client_age_days': row['client_age_days'],
            'is_new_client': row['is_new_client'],
            'has_utilization_change': row['has_utilization_change'],
            'has_deposit_change': row['has_deposit_change'],
            'sufficient_utilization_history': row['sufficient_utilization_history'],
            'sufficient_deposit_history': row['sufficient_deposit_history'],
            'is_noisy': row['is_noisy'],
            'data_completeness': row['data_completeness'],
            'persona_type': 'sparse'
        })
    
    return pd.DataFrame(persona_assignments)

def perform_clustering_analysis(sparse_features_df):
    """
    Perform clustering analysis to validate the optimal number of sparse personas
    and compare with our business-logic approach.
    """
    print("Performing clustering analysis to validate persona count...")
    
    # Select features for clustering (numerical features only)
    clustering_features = [
        'client_age_days', 'utilization_change', 'utilization_std', 'utilization_cv',
        'deposit_change_rel', 'deposit_std', 'deposit_cv', 'data_completeness',
        'total_records', 'utilization_records_count', 'deposit_records_count',
        'avg_utilization', 'avg_deposit'
    ]
    
    # Prepare data for clustering
    clustering_data = sparse_features_df[clustering_features].copy()
    
    # Handle any remaining infinite or NaN values
    clustering_data = clustering_data.replace([np.inf, -np.inf], np.nan)
    clustering_data = clustering_data.fillna(0)
    
    # Standardize features for clustering
    scaler = StandardScaler()
    clustering_data_scaled = scaler.fit_transform(clustering_data)
    
    # Test different numbers of clusters
    cluster_range = range(
        SIMPLIFIED_CONFIG['clustering']['min_clusters'], 
        SIMPLIFIED_CONFIG['clustering']['max_clusters'] + 1
    )
    
    silhouette_scores = []
    inertias = []
    cluster_models = {}
    
    for n_clusters in cluster_range:
        # Perform K-means clustering
        kmeans = KMeans(
            n_clusters=n_clusters, 
            random_state=SIMPLIFIED_CONFIG['clustering']['random_state'],
            n_init=10
        )
        cluster_labels = kmeans.fit_predict(clustering_data_scaled)
        
        # Calculate silhouette score
        if n_clusters > 1:  # Silhouette score requires at least 2 clusters
            silhouette_avg = silhouette_score(clustering_data_scaled, cluster_labels)
            silhouette_scores.append(silhouette_avg)
        else:
            silhouette_scores.append(0)
        
        inertias.append(kmeans.inertia_)
        cluster_models[n_clusters] = kmeans
    
    # Find optimal number of clusters
    if len(silhouette_scores) > 1:
        optimal_clusters_silhouette = cluster_range[np.argmax(silhouette_scores)]
        max_silhouette_score = max(silhouette_scores)
    else:
        optimal_clusters_silhouette = 2
        max_silhouette_score = 0
    
    # Use elbow method for additional validation
    # Calculate the rate of change in inertia
    if len(inertias) > 2:
        inertia_changes = []
        for i in range(1, len(inertias)):
            change = inertias[i-1] - inertias[i]
            inertia_changes.append(change)
        
        # Find the elbow (point where improvement slows down significantly)
        if len(inertia_changes) > 1:
            change_rates = []
            for i in range(1, len(inertia_changes)):
                rate_change = inertia_changes[i-1] - inertia_changes[i]
                change_rates.append(rate_change)
            
            # The elbow is where the rate of change starts to level off
            optimal_clusters_elbow = cluster_range[np.argmax(change_rates) + 2] if change_rates else optimal_clusters_silhouette
        else:
            optimal_clusters_elbow = optimal_clusters_silhouette
    else:
        optimal_clusters_elbow = optimal_clusters_silhouette
    
    # Get the optimal clustering result
    optimal_clusters = optimal_clusters_silhouette  # Prioritize silhouette score
    optimal_model = cluster_models[optimal_clusters]
    optimal_labels = optimal_model.fit_predict(clustering_data_scaled)
    
    # Add cluster labels to the features dataframe
    sparse_features_with_clusters = sparse_features_df.copy()
    sparse_features_with_clusters['cluster_label'] = optimal_labels
    
    # Analyze cluster characteristics
    cluster_analysis = analyze_cluster_characteristics(sparse_features_with_clusters, optimal_clusters)
    
    clustering_results = {
        'optimal_clusters': optimal_clusters,
        'silhouette_scores': list(zip(cluster_range, silhouette_scores)),
        'inertias': list(zip(cluster_range, inertias)),
        'max_silhouette_score': max_silhouette_score,
        'optimal_clusters_elbow': optimal_clusters_elbow,
        'features_with_clusters': sparse_features_with_clusters,
        'cluster_analysis': cluster_analysis,
        'scaler': scaler,
        'optimal_model': optimal_model
    }
    
    print(f"Clustering analysis complete:")
    print(f"  Optimal clusters (silhouette): {optimal_clusters}")
    print(f"  Max silhouette score: {max_silhouette_score:.3f}")
    print(f"  Business logic personas: 6")
    print(f"  Clustering suggests: {optimal_clusters} clusters")
    
    return clustering_results

def analyze_cluster_characteristics(features_with_clusters, n_clusters):
    """
    Analyze the characteristics of each cluster to understand what patterns
    the clustering algorithm discovered.
    """
    cluster_characteristics = {}
    
    for cluster_id in range(n_clusters):
        cluster_data = features_with_clusters[features_with_clusters['cluster_label'] == cluster_id]
        
        if len(cluster_data) == 0:
            continue
        
        characteristics = {
            'size': len(cluster_data),
            'percentage': len(cluster_data) / len(features_with_clusters) * 100,
            'avg_client_age_days': cluster_data['client_age_days'].mean(),
            'new_client_rate': cluster_data['is_new_client'].mean(),
            'utilization_change_rate': cluster_data['has_utilization_change'].mean(),
            'deposit_change_rate': cluster_data['has_deposit_change'].mean(),
            'sufficient_util_history_rate': cluster_data['sufficient_utilization_history'].mean(),
            'sufficient_deposit_history_rate': cluster_data['sufficient_deposit_history'].mean(),
            'noisy_data_rate': cluster_data['is_noisy'].mean(),
            'avg_data_completeness': cluster_data['data_completeness'].mean(),
            'avg_utilization_cv': cluster_data['utilization_cv'].mean(),
            'avg_deposit_cv': cluster_data['deposit_cv'].mean(),
        }
        
        # Determine dominant pattern for this cluster
        patterns = []
        if characteristics['new_client_rate'] > 0.5:
            patterns.append('new_clients')
        if characteristics['utilization_change_rate'] < 0.3:
            patterns.append('stable_utilization')
        if characteristics['deposit_change_rate'] < 0.3:
            patterns.append('stable_deposits')
        if characteristics['sufficient_util_history_rate'] < 0.3:
            patterns.append('insufficient_utilization_history')
        if characteristics['sufficient_deposit_history_rate'] < 0.3:
            patterns.append('insufficient_deposit_history')
        if characteristics['noisy_data_rate'] > 0.3:
            patterns.append('noisy_data')
        
        characteristics['dominant_patterns'] = patterns
        cluster_characteristics[cluster_id] = characteristics
    
    return cluster_characteristics

#######################################################
# TRADITIONAL ANALYSIS (SIMPLIFIED VERSION)
#######################################################

def add_traditional_derived_metrics_simplified(df):
    """
    Simplified version of traditional metrics calculation.
    Focuses on the core risk indicators without overwhelming complexity.
    """
    df = df.copy()
    
    # Basic metrics
    df['total_loan'] = df['used_loan'] + df['unused_loan']
    df['loan_utilization'] = df['used_loan'] / df['total_loan']
    df.loc[df['total_loan'] == 0, 'loan_utilization'] = np.nan
    
    # Calculate deposit to loan ratio
    df['deposit_loan_ratio'] = df['deposit_balance'] / df['used_loan']
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    
    # Calculate core rolling metrics for each company
    for company in tqdm(df['company_id'].unique(), desc="Calculating traditional metrics"):
        company_data = df[df['company_id'] == company].sort_values('date')
        
        if len(company_data) < 30:
            continue
            
        # Core rolling averages
        for window in [30, 90]:
            df.loc[company_data.index, f'util_ma_{window}d'] = company_data['loan_utilization'].rolling(
                window, min_periods=max(3, window//4)).mean()
            df.loc[company_data.index, f'deposit_ma_{window}d'] = company_data['deposit_balance'].rolling(
                window, min_periods=max(3, window//4)).mean()
            
        # Core change rates
        for window in [30, 90]:
            df.loc[company_data.index, f'util_change_{window}d'] = (
                company_data['loan_utilization'].pct_change(periods=window)
            )
            df.loc[company_data.index, f'deposit_change_{window}d'] = (
                company_data['deposit_balance'].pct_change(periods=window)
            )
    
    return df

def detect_traditional_risk_patterns_simplified(df):
    """
    Simplified traditional risk pattern detection focusing on core risk indicators.
    """
    risk_records = []
    persona_assignments = []
    
    # Get the latest date for recent risk calculation
    max_date = df['date'].max()
    recent_cutoff = max_date - pd.Timedelta(days=SIMPLIFIED_CONFIG['data']['recent_window'])
    
    # Process each company
    for company in tqdm(df['company_id'].unique(), desc="Detecting traditional risk patterns"):
        company_data = df[df['company_id'] == company].sort_values('date')
        
        if len(company_data) < 90:  # Need at least 3 months of data
            continue
        
        # Process recent data points
        recent_data = company_data[company_data['date'] >= recent_cutoff]
        
        for _, current_row in recent_data.iterrows():
            current_date = current_row['date']
            current_util = current_row['loan_utilization']
            current_deposit = current_row['deposit_balance']
            
            if pd.isna(current_util) or pd.isna(current_deposit):
                continue
            
            risk_flags = []
            risk_levels = []
            risk_descriptions = []
            persona = None
            persona_confidence = 0.0
            
            # Core Risk Pattern 1: Deteriorating Health (rising utilization + declining deposits)
            util_change_90d = current_row.get('util_change_90d', 0)
            deposit_change_90d = current_row.get('deposit_change_90d', 0)
            
            if not pd.isna(util_change_90d) and not pd.isna(deposit_change_90d):
                if util_change_90d > 0.1 and deposit_change_90d < -0.1:
                    severity = "high" if (util_change_90d > 0.2 and deposit_change_90d < -0.2) else "medium"
                    risk_flags.append('deteriorating_health')
                    risk_descriptions.append(
                        f"[{severity.upper()}] Deteriorating: +{util_change_90d:.1%} utilization, "
                        f"{deposit_change_90d:.1%} deposits"
                    )
                    risk_levels.append(severity)
                    persona = "deteriorating_health"
                    persona_confidence = 0.8
            
            # Core Risk Pattern 2: High Utilization Risk
            if current_util > 0.75:
                severity = "high" if current_util > 0.9 else "medium"
                risk_flags.append('high_utilization')
                risk_descriptions.append(
                    f"[{severity.upper()}] High utilization: {current_util:.1%}"
                )
                risk_levels.append(severity)
                
                if persona_confidence < 0.7:
                    if current_util > 0.9:
                        persona = "distressed_client"
                    else:
                        persona = "credit_dependent"
                    persona_confidence = 0.7
            
            # Core Risk Pattern 3: Rapid Deposit Decline
            if not pd.isna(deposit_change_90d) and deposit_change_90d < -0.15:
                severity = "high" if deposit_change_90d < -0.25 else "medium"
                risk_flags.append('rapid_deposit_decline')
                risk_descriptions.append(
                    f"[{severity.upper()}] Rapid deposit decline: {deposit_change_90d:.1%}"
                )
                risk_levels.append(severity)
                
                if persona_confidence < 0.75:
                    persona = "cash_constrained"
                    persona_confidence = 0.75
            
            # Assign default persona if none assigned
            if persona is None:
                if current_util < 0.4:
                    persona = "cautious_borrower"
                    persona_confidence = 0.6
                elif current_util > 0.75:
                    persona = "credit_dependent"
                    persona_confidence = 0.6
                else:
                    continue  # Skip companies without clear risk patterns
            
            if risk_flags:
                # Determine overall risk level
                overall_risk = "low"
                if "high" in risk_levels:
                    overall_risk = "high"
                elif "medium" in risk_levels:
                    overall_risk = "medium"
                
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
                    'is_recent': True
                })
                
                persona_assignments.append({
                    'company_id': company,
                    'date': current_date,
                    'persona': persona,
                    'confidence': persona_confidence,
                    'risk_level': overall_risk,
                    'is_recent': True,
                    'persona_type': 'traditional'
                })
    
    # Create dataframes
    risk_df = pd.DataFrame(risk_records) if risk_records else pd.DataFrame()
    persona_df = pd.DataFrame(persona_assignments) if persona_assignments else pd.DataFrame()
    
    return risk_df, persona_df

#######################################################
# INTEGRATED VISUALIZATION AND REPORTING
#######################################################

def plot_clustering_validation(clustering_results):
    """
    Visualize clustering analysis results to validate the optimal number of personas.
    """
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Silhouette scores
    cluster_range, silhouette_scores = zip(*clustering_results['silhouette_scores'])
    ax1.plot(cluster_range, silhouette_scores, marker='o', linewidth=2, markersize=8)
    ax1.axvline(x=clustering_results['optimal_clusters'], color='red', linestyle='--', 
                label=f'Optimal: {clustering_results["optimal_clusters"]} clusters')
    ax1.axvline(x=6, color='green', linestyle=':', 
                label='Business Logic: 6 personas')
    ax1.set_xlabel('Number of Clusters')
    ax1.set_ylabel('Silhouette Score')
    ax1.set_title('Silhouette Score vs Number of Clusters')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Elbow method (inertia)
    cluster_range, inertias = zip(*clustering_results['inertias'])
    ax2.plot(cluster_range, inertias, marker='s', linewidth=2, markersize=8)
    ax2.axvline(x=clustering_results['optimal_clusters'], color='red', linestyle='--',
                label=f'Optimal: {clustering_results["optimal_clusters"]} clusters')
    ax2.axvline(x=6, color='green', linestyle=':', 
                label='Business Logic: 6 personas')
    ax2.set_xlabel('Number of Clusters')
    ax2.set_ylabel('Inertia (Within-cluster sum of squares)')
    ax2.set_title('Elbow Method for Optimal Clusters')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Cluster size distribution
    if 'cluster_analysis' in clustering_results:
        cluster_analysis = clustering_results['cluster_analysis']
        cluster_ids = list(cluster_analysis.keys())
        cluster_sizes = [cluster_analysis[cid]['size'] for cid in cluster_ids]
        
        colors = plt.cm.tab10(np.linspace(0, 1, len(cluster_ids)))
        bars = ax3.bar(cluster_ids, cluster_sizes, color=colors)
        ax3.set_xlabel('Cluster ID')
        ax3.set_ylabel('Number of Companies')
        ax3.set_title('Cluster Size Distribution')
        
        # Add percentage labels on bars
        total_companies = sum(cluster_sizes)
        for bar, size in zip(bars, cluster_sizes):
            height = bar.get_height()
            percentage = (size / total_companies) * 100
            ax3.text(bar.get_x() + bar.get_width()/2., height + 1,
                    f'{percentage:.1f}%', ha='center', va='bottom', fontsize=9)
    
    # 4. Cluster characteristics heatmap
    if 'cluster_analysis' in clustering_results:
        cluster_analysis = clustering_results['cluster_analysis']
        
        # Prepare data for heatmap
        characteristics = ['new_client_rate', 'utilization_change_rate', 'deposit_change_rate',
                          'sufficient_util_history_rate', 'sufficient_deposit_history_rate', 'noisy_data_rate']
        
        heatmap_data = []
        cluster_labels = []
        
        for cluster_id, analysis in cluster_analysis.items():
            row = [analysis.get(char, 0) for char in characteristics]
            heatmap_data.append(row)
            cluster_labels.append(f'Cluster {cluster_id}')
        
        if heatmap_data:
            heatmap_array = np.array(heatmap_data)
            im = ax4.imshow(heatmap_array, cmap='viridis', aspect='auto')
            
            # Set ticks and labels
            ax4.set_xticks(range(len(characteristics)))
            ax4.set_xticklabels([char.replace('_', '\n') for char in characteristics], rotation=45, ha='right')
            ax4.set_yticks(range(len(cluster_labels)))
            ax4.set_yticklabels(cluster_labels)
            ax4.set_title('Cluster Characteristics Heatmap')
            
            # Add colorbar
            plt.colorbar(im, ax=ax4, label='Rate (0-1)')
            
            # Add text annotations
            for i in range(len(cluster_labels)):
                for j in range(len(characteristics)):
                    text = ax4.text(j, i, f'{heatmap_array[i, j]:.2f}',
                                   ha="center", va="center", color="white", fontweight='bold')
    
    plt.suptitle('Clustering Analysis for Sparse Data Personas', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    return fig

def plot_simplified_integrated_overview(sparse_persona_df, traditional_persona_df=None, clustering_results=None):
    """
    Create a comprehensive overview of the simplified integrated analysis.
    """
    fig = plt.figure(figsize=(20, 16))
    
    # 1. Sparse persona distribution
    ax1 = plt.subplot(2, 4, 1)
    if not sparse_persona_df.empty:
        persona_counts = sparse_persona_df['persona'].value_counts()
        colors = plt.cm.Set3(np.linspace(0, 1, len(persona_counts)))
        
        # Create horizontal bar chart for better readability
        bars = ax1.barh(range(len(persona_counts)), persona_counts.values, color=colors)
        ax1.set_yticks(range(len(persona_counts)))
        ax1.set_yticklabels([name.replace('_', '\n') for name in persona_counts.index], fontsize=10)
        ax1.set_xlabel('Number of Companies')
        ax1.set_title('Sparse Data Personas\n(Business Logic)', fontweight='bold')
        
        # Add value labels
        for i, (bar, count) in enumerate(zip(bars, persona_counts.values)):
            ax1.text(count + 0.5, i, str(count), va='center', fontweight='bold')
    
    # 2. Risk level distribution 
    ax2 = plt.subplot(2, 4, 2)
    if not sparse_persona_df.empty:
        risk_counts = sparse_persona_df['risk_level'].value_counts()
        colors = {'low': 'lightgreen', 'medium': 'orange', 'high': 'red'}
        pie_colors = [colors.get(level, 'gray') for level in risk_counts.index]
        
        wedges, texts, autotexts = ax2.pie(risk_counts.values, labels=risk_counts.index,
                                          autopct='%1.1f%%', colors=pie_colors, startangle=90)
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')
        ax2.set_title('Risk Level Distribution\n(Sparse Data)', fontweight='bold')
    
    # 3. Confidence score distribution
    ax3 = plt.subplot(2, 4, 3)
    if not sparse_persona_df.empty:
        ax3.hist(sparse_persona_df['confidence'], bins=15, alpha=0.7, color='skyblue', edgecolor='black')
        ax3.axvline(sparse_persona_df['confidence'].mean(), color='red', linestyle='--',
                   label=f'Mean: {sparse_persona_df["confidence"].mean():.2f}')
        ax3.set_xlabel('Confidence Score')
        ax3.set_ylabel('Number of Companies')
        ax3.set_title('Confidence Score\nDistribution', fontweight='bold')
        ax3.legend()
    
    # 4. New client analysis
    ax4 = plt.subplot(2, 4, 4)
    if not sparse_persona_df.empty:
        new_client_analysis = sparse_persona_df.groupby(['is_new_client', 'risk_level']).size().unstack(fill_value=0)
        new_client_analysis.plot(kind='bar', ax=ax4, color=['lightgreen', 'orange', 'red'], stacked=True)
        ax4.set_xlabel('Is New Client')
        ax4.set_ylabel('Number of Companies')
        ax4.set_title('New Client Risk\nDistribution', fontweight='bold')
        labels = ['Existing Client', 'New Client']
        ax4.set_xticks(range(len(labels)))
        ax4.set_xticklabels(labels, rotation=0)
        # ax4.set_xticklabels(['Existing Client', 'New Client'], rotation=0)
        ax4.legend(title='Risk Level', fontsize=8)
    
    # 5. Data sufficiency analysis
    ax5 = plt.subplot(2, 4, 5)
    if not sparse_persona_df.empty:
        sufficiency_data = {
            'Sufficient\nUtilization': sparse_persona_df['sufficient_utilization_history'].sum(),
            'Sufficient\nDeposits': sparse_persona_df['sufficient_deposit_history'].sum(),
            'Insufficient\nUtilization': (~sparse_persona_df['sufficient_utilization_history']).sum(),
            'Insufficient\nDeposits': (~sparse_persona_df['sufficient_deposit_history']).sum()
        }
        
        bars = ax5.bar(sufficiency_data.keys(), sufficiency_data.values(), 
                      color=['green', 'green', 'red', 'red'], alpha=0.7)
        ax5.set_ylabel('Number of Companies')
        ax5.set_title('Data Sufficiency\nAnalysis', fontweight='bold')
        ax5.tick_params(axis='x', rotation=45)
        
        # Add value labels
        for bar, value in zip(bars, sufficiency_data.values()):
            height = bar.get_height()
            ax5.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                    str(value), ha='center', va='bottom', fontweight='bold')
    
    # 6. Clustering vs Business Logic Comparison
    ax6 = plt.subplot(2, 4, 6)
    if clustering_results:
        optimal_clusters = clustering_results['optimal_clusters']
        business_logic_personas = 6
        max_silhouette = clustering_results['max_silhouette_score']
        
        comparison_data = {
            'Clustering\nSuggestion': optimal_clusters,
            'Business\nLogic': business_logic_personas
        }
        
        bars = ax6.bar(comparison_data.keys(), comparison_data.values(), 
                      color=['blue', 'green'], alpha=0.7)
        ax6.set_ylabel('Number of Personas/Clusters')
        ax6.set_title(f'Persona Count Comparison\n(Silhouette: {max_silhouette:.3f})', fontweight='bold')
        
        # Add value labels
        for bar, value in zip(bars, comparison_data.values()):
            height = bar.get_height()
            ax6.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    str(value), ha='center', va='bottom', fontweight='bold', fontsize=12)
    
    # 7. Traditional vs Sparse Coverage
    ax7 = plt.subplot(2, 4, 7)
    sparse_count = len(sparse_persona_df) if not sparse_persona_df.empty else 0
    traditional_count = len(traditional_persona_df) if traditional_persona_df is not None and not traditional_persona_df.empty else 0
    
    coverage_data = {
        'Sparse\nAnalysis': sparse_count,
        'Traditional\nAnalysis': traditional_count
    }
    
    colors = ['#CD853F', '#2E8B57']
    bars = ax7.bar(coverage_data.keys(), coverage_data.values(), color=colors, alpha=0.7)
    ax7.set_ylabel('Number of Companies')
    ax7.set_title('Analysis Coverage\nComparison', fontweight='bold')
    
    # Add value labels and percentages
    total_companies = sparse_count + traditional_count
    for bar, (key, value) in zip(bars, coverage_data.items()):
        height = bar.get_height()
        percentage = (value / total_companies * 100) if total_companies > 0 else 0
        ax7.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{value}\n({percentage:.1f}%)', ha='center', va='bottom', fontweight='bold')
    
    # 8. Summary Statistics
    ax8 = plt.subplot(2, 4, 8)
    ax8.axis('off')
    
    # Calculate summary statistics
    total_analyzed = sparse_count + traditional_count
    high_risk_sparse = len(sparse_persona_df[sparse_persona_df['risk_level'] == 'high']) if not sparse_persona_df.empty else 0
    high_risk_traditional = len(traditional_persona_df[traditional_persona_df['risk_level'] == 'high']) if traditional_persona_df is not None and not traditional_persona_df.empty else 0
    
    avg_confidence_sparse = sparse_persona_df['confidence'].mean() if not sparse_persona_df.empty else 0
    avg_confidence_traditional = traditional_persona_df['confidence'].mean() if traditional_persona_df is not None and not traditional_persona_df.empty else 0
    
    silhouette_score_display = (
    f"{clustering_results['max_silhouette_score']:.3f}"
    if clustering_results else "N/A")

    optimal_clusters_display = (
    clustering_results['optimal_clusters']
    if clustering_results else "N/A")

    summary_text = f"""
    SIMPLIFIED ANALYSIS SUMMARY

    Total Companies: {total_analyzed}

    Coverage Breakdown:
    • Sparse Analysis: {sparse_count}
    • Traditional Analysis: {traditional_count}

    Risk Distribution:
    • High Risk (Sparse): {high_risk_sparse}
    • High Risk (Traditional): {high_risk_traditional}

    Quality Metrics:
    • Avg Confidence (Sparse): {avg_confidence_sparse:.2f}
    • Avg Confidence (Traditional): {avg_confidence_traditional:.2f}

    Personas Discovered:
    • Business Logic: 6 sparse personas
    • Clustering Suggests: {optimal_clusters_display}

    Model Validation:
    • Silhouette Score: {silhouette_score_display}
    """
    
    ax8.text(0.05, 0.95, summary_text, transform=ax8.transAxes, fontsize=11,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    plt.suptitle('Simplified Integrated Bank Risk Analysis - Overview Dashboard', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    return fig

def create_simplified_priority_report(sparse_persona_df, traditional_persona_df=None, clustering_results=None):
    """
    Generate a simplified priority action report with clear business recommendations.
    """
    print("\n" + "="*80)
    print("SIMPLIFIED PRIORITY ACTION REPORT")
    print("="*80)
    
    priority_actions = []
    
    # Process sparse data personas
    if not sparse_persona_df.empty:
        for _, row in sparse_persona_df.iterrows():
            persona = row['persona']
            risk_level = row['risk_level']
            confidence = row['confidence']
            company_id = row['company_id']
            
            # Define actions based on simplified personas
            action_mapping = {
                'new_client': {
                    'action': 'New client onboarding review and engagement strategy',
                    'priority': 2.0 + confidence,
                    'category': 'Client Onboarding',
                    'urgency': 'Medium - within 30 days'
                },
                'no_utilization_change': {
                    'action': 'Monitor for potential credit needs or relationship changes',
                    'priority': 1.0 + confidence,
                    'category': 'Relationship Monitoring',
                    'urgency': 'Low - quarterly review'
                },
                'no_deposit_change': {
                    'action': 'Assess deposit growth opportunities and cash management needs',
                    'priority': 1.0 + confidence,
                    'category': 'Deposit Growth',
                    'urgency': 'Low - quarterly review'
                },
                'insufficient_utilization_history': {
                    'action': 'Credit product education and needs assessment',
                    'priority': 1.5 + confidence,
                    'category': 'Product Development',
                    'urgency': 'Medium - within 60 days'
                },
                'insufficient_deposit_history': {
                    'action': 'Deposit relationship development and treasury services review',
                    'priority': 1.5 + confidence,
                    'category': 'Relationship Development',
                    'urgency': 'Medium - within 60 days'
                },
                'noisy_history': {
                    'action': 'Data quality review and specialized risk assessment',
                    'priority': 2.5 + confidence,
                    'category': 'Risk Assessment',
                    'urgency': 'High - within 15 days'
                }
            }
            
            action_info = action_mapping.get(persona, {
                'action': 'Standard relationship review',
                'priority': 1.0,
                'category': 'General Review',
                'urgency': 'Low - annual review'
            })
            
            # Adjust priority based on risk level
            if risk_level == 'high':
                action_info['priority'] += 1.0
            elif risk_level == 'medium':
                action_info['priority'] += 0.5
            
            priority_actions.append({
                'company_id': company_id,
                'persona': persona,
                'risk_level': risk_level,
                'confidence': confidence,
                'analysis_type': 'Sparse Data',
                'priority_score': action_info['priority'],
                'action_category': action_info['category'],
                'recommended_action': action_info['action'],
                'urgency': action_info['urgency'],
                'reasoning': row.get('reasoning', 'Based on sparse data analysis')
            })
    
    # Process traditional personas
    if traditional_persona_df is not None and not traditional_persona_df.empty:
        traditional_summary = traditional_persona_df.groupby(['company_id', 'persona']).agg({
            'confidence': 'mean',
            'risk_level': lambda x: x.mode().iloc[0] if not x.empty else 'low'
        }).reset_index()
        
        for _, row in traditional_summary.iterrows():
            priority_score = 3.0 + row['confidence']  # Traditional analysis gets higher base priority
            if row['risk_level'] == 'high':
                priority_score += 1.0
            elif row['risk_level'] == 'medium':
                priority_score += 0.5
            
            priority_actions.append({
                'company_id': row['company_id'],
                'persona': row['persona'],
                'risk_level': row['risk_level'],
                'confidence': row['confidence'],
                'analysis_type': 'Traditional',
                'priority_score': priority_score,
                'action_category': 'Credit Risk Management',
                'recommended_action': 'Immediate credit risk review and intervention',
                'urgency': 'High - within 7 days',
                'reasoning': 'High-activity client showing risk patterns'
            })
    
    # Create priority dataframe and sort
    priority_df = pd.DataFrame(priority_actions)
    if not priority_df.empty:
        priority_df = priority_df.sort_values('priority_score', ascending=False)
    
    # Print summary
    print(f"\nTOTAL PRIORITY ACTIONS: {len(priority_df)}")
    print("-" * 50)
    
    if not priority_df.empty:
        # Group by category
        by_category = priority_df.groupby('action_category').agg({
            'company_id': 'count',
            'priority_score': 'mean'
        }).sort_values('priority_score', ascending=False)
        
        print("ACTIONS BY CATEGORY:")
        for category, data in by_category.iterrows():
            print(f"  {category}: {data['company_id']} companies (avg priority: {data['priority_score']:.2f})")
        
        print(f"\nTOP 10 PRIORITY ACTIONS:")
        print("-" * 50)
        
        for i, (_, row) in enumerate(priority_df.head(10).iterrows(), 1):
            print(f"{i:2d}. Company {row['company_id']} ({row['analysis_type']})")
            print(f"    Persona: {row['persona']} | Risk: {row['risk_level'].upper()}")
            print(f"    Action: {row['recommended_action']}")
            print(f"    Urgency: {row['urgency']}")
            print(f"    Priority Score: {row['priority_score']:.2f}")
            print()
    
    # Clustering validation summary
    if clustering_results:
        print(f"\nCLUSTERING VALIDATION RESULTS:")
        print("-" * 50)
        print(f"Business Logic Personas: 6")
        print(f"Optimal Clusters (Data-Driven): {clustering_results['optimal_clusters']}")
        print(f"Silhouette Score: {clustering_results['max_silhouette_score']:.3f}")
        
        if clustering_results['optimal_clusters'] == 6:
            print("✅ Clustering analysis CONFIRMS business logic approach!")
        elif clustering_results['optimal_clusters'] < 6:
            print(f"⚠️  Clustering suggests FEWER personas ({clustering_results['optimal_clusters']}) - consider consolidation")
        else:
            print(f"⚠️  Clustering suggests MORE personas ({clustering_results['optimal_clusters']}) - consider refinement")
        
        # Analyze cluster characteristics
        if 'cluster_analysis' in clustering_results:
            print(f"\nCLUSTER CHARACTERISTICS:")
            print("-" * 30)
            cluster_analysis = clustering_results['cluster_analysis']
            for cluster_id, characteristics in cluster_analysis.items():
                print(f"Cluster {cluster_id} ({characteristics['size']} companies, {characteristics['percentage']:.1f}%):")
                patterns = characteristics.get('dominant_patterns', [])
                if patterns:
                    print(f"  Primary patterns: {', '.join(patterns)}")
                else:
                    print(f"  Mixed patterns")
    
    print(f"\n" + "="*80)
    print("RECOMMENDATIONS:")
    print("="*80)
    print("1. IMMEDIATE: Address high-priority traditional risk cases")
    print("2. SHORT-TERM: Implement new client onboarding improvements")
    print("3. MEDIUM-TERM: Develop targeted engagement strategies for sparse data clients")
    print("4. LONG-TERM: Monitor clustering results to refine persona definitions")
    
    if clustering_results and clustering_results['optimal_clusters'] != 6:
        print(f"5. VALIDATION: Consider adjusting to {clustering_results['optimal_clusters']} personas based on clustering analysis")
    
    return priority_df

#######################################################
# MAIN SIMPLIFIED INTEGRATED WORKFLOW
#######################################################

def simplified_integrated_bank_risk_analysis(df):
    """
    Main simplified workflow that combines traditional and sparse analysis
    with clustering validation for persona optimization.
    """
    print("="*80)
    print("SIMPLIFIED INTEGRATED BANK RISK ANALYSIS")
    print("="*80)
    print("Business Logic: 6 clear sparse personas + clustering validation")
    
    results = {}
    
    # 1. Simplified data cleaning
    print("\n1. Simplified data cleaning and categorization...")
    df_traditional, df_sparse, cleaning_stats = simplified_clean_data(df, min_nonzero_pct=0.8)
    
    results['cleaning_stats'] = cleaning_stats
    
    print(f"  Traditional companies: {len(cleaning_stats['traditional_companies'])}")
    print(f"  Sparse companies: {len(cleaning_stats['sparse_companies'])}")
    
    # 2. Sparse data analysis with clustering validation
    sparse_results = {}
    if not df_sparse.empty:
        print("\n2. Sparse data analysis with clustering validation...")
        
        # Calculate sparse features
        sparse_features_df = calculate_sparse_features(df_sparse)
        
        # Perform clustering analysis for validation
        clustering_results = perform_clustering_analysis(sparse_features_df)
        
        # Assign business logic personas
        sparse_persona_df = assign_simplified_sparse_personas(sparse_features_df)
        
        sparse_results = {
            'features_df': sparse_features_df,
            'persona_df': sparse_persona_df,
            'clustering_results': clustering_results
        }
        
        print(f"  Sparse features calculated: {len(sparse_features_df)}")
        print(f"  Business logic personas assigned: {len(sparse_persona_df)}")
        print(f"  Clustering suggests: {clustering_results['optimal_clusters']} clusters")
    
    results['sparse_analysis'] = sparse_results
    
    # 3. Traditional analysis (simplified)
    traditional_results = {}
    if not df_traditional.empty:
        print("\n3. Traditional risk analysis (simplified)...")
        
        # Add simplified traditional metrics
        df_traditional_with_metrics = add_traditional_derived_metrics_simplified(df_traditional)
        
        # Detect traditional risk patterns
        traditional_risk_df, traditional_persona_df = detect_traditional_risk_patterns_simplified(df_traditional_with_metrics)
        
        traditional_results = {
            'risk_df': traditional_risk_df,
            'persona_df': traditional_persona_df
        }
        
        print(f"  Traditional risk events: {len(traditional_risk_df)}")
        print(f"  Traditional personas: {len(traditional_persona_df)}")
    
    results['traditional_analysis'] = traditional_results
    
    # 4. Generate visualizations
    print("\n4. Creating visualizations...")
    
    # Clustering validation plot
    if 'clustering_results' in sparse_results:
        clustering_fig = plot_clustering_validation(sparse_results['clustering_results'])
        clustering_fig.savefig('clustering_validation_analysis.png', dpi=300, bbox_inches='tight')
        print("  Saved: clustering_validation_analysis.png")
    
    # Integrated overview
    sparse_persona_df = sparse_results.get('persona_df', pd.DataFrame())
    traditional_persona_df = traditional_results.get('persona_df', pd.DataFrame())
    clustering_results = sparse_results.get('clustering_results')
    
    overview_fig = plot_simplified_integrated_overview(
        sparse_persona_df, traditional_persona_df, clustering_results
    )
    overview_fig.savefig('simplified_integrated_overview.png', dpi=300, bbox_inches='tight')
    print("  Saved: simplified_integrated_overview.png")
    
    # 5. Generate priority action report
    print("\n5. Generating priority action report...")
    priority_df = create_simplified_priority_report(
        sparse_persona_df, traditional_persona_df, clustering_results
    )
    
    results['priority_actions'] = priority_df
    results['visualizations'] = {
        'clustering_validation': 'clustering_validation_analysis.png',
        'integrated_overview': 'simplified_integrated_overview.png'
    }
    
    print("\n" + "="*80)
    print("SIMPLIFIED ANALYSIS COMPLETE!")
    print("="*80)
    
    total_companies = len(cleaning_stats['traditional_companies']) + len(cleaning_stats['sparse_companies'])
    print(f"✅ Total companies analyzed: {total_companies}")
    print(f"✅ Sparse personas (business logic): 6 clear categories")
    print(f"✅ Clustering validation: {clustering_results['optimal_clusters'] if clustering_results else 'N/A'} optimal clusters")
    print(f"✅ Priority actions identified: {len(priority_df) if priority_df is not None else 0}")
    print(f"✅ Visualizations: 2 comprehensive reports")
    
    return results

#######################################################
# ENHANCED DATA GENERATION FOR TESTING
#######################################################

def generate_realistic_test_data(num_companies=100, days=730):
    """
    Generate realistic test data that will demonstrate both traditional and sparse patterns.
    """
    print(f"Generating realistic test data: {num_companies} companies, {days} days...")
    
    np.random.seed(42)
    
    end_date = pd.Timestamp('2023-12-31')
    start_date = end_date - pd.Timedelta(days=days)
    date_range = pd.date_range(start=start_date, end=end_date)
    
    company_ids = [f'COMP{str(i).zfill(4)}' for i in range(num_companies)]
    
    data = []
    
    for i, company_id in enumerate(tqdm(company_ids, desc="Generating realistic data")):
        # Determine company pattern (40% traditional, 60% sparse)
        company_type = np.random.choice([
            'traditional_active', 'new_client', 'no_util_change', 'no_deposit_change',
            'insufficient_util', 'insufficient_deposit', 'noisy_data'
        ], p=[0.4, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
        
        if company_type == 'traditional_active':
            # Generate rich, active data suitable for traditional analysis
            base_deposit = np.random.lognormal(12, 1)  # Higher base values
            base_loan = np.random.lognormal(11, 1.2)
            util_rate = np.random.uniform(0.4, 0.8)
            
            # Regular activity with some risk patterns
            has_risk = np.random.random() < 0.3
            risk_start = int(len(date_range) * 0.6) if has_risk else len(date_range)
            
            for j, date in enumerate(date_range):
                # High activity rate (90%+ of days have data)
                if np.random.random() < 0.95:
                    # Trend and seasonal components
                    trend = 1 + 0.001 * j
                    seasonal = 1 + 0.05 * np.sin(2 * np.pi * date.dayofyear / 365)
                    noise = np.random.normal(1, 0.05)
                    
                    # Risk pattern
                    if j > risk_start:
                        risk_factor = 1 - 0.002 * (j - risk_start)
                        util_increase = 0.001 * (j - risk_start)
                    else:
                        risk_factor = 1
                        util_increase = 0
                    
                    deposit = base_deposit * trend * seasonal * noise * risk_factor
                    utilization = min(0.95, max(0.1, util_rate + util_increase + np.random.normal(0, 0.02)))
                    used_loan = base_loan * utilization
                    unused_loan = base_loan - used_loan
                else:
                    deposit = used_loan = unused_loan = 0
                
                data.append({
                    'company_id': company_id,
                    'date': date,
                    'deposit_balance': max(0, deposit),
                    'used_loan': max(0, used_loan),
                    'unused_loan': max(0, unused_loan)
                })
        
        elif company_type == 'new_client':
            # New client (< 1 year) with moderate activity
            start_active = len(date_range) - 300  # Active for last 300 days
            base_deposit = np.random.lognormal(10, 1)
            base_loan = np.random.lognormal(9, 1)
            
            for j, date in enumerate(date_range):
                if j >= start_active and np.random.random() < 0.7:
                    deposit = base_deposit * np.random.normal(1, 0.1)
                    used_loan = base_loan * np.random.uniform(0.3, 0.7)
                    unused_loan = base_loan - used_loan
                else:
                    deposit = used_loan = unused_loan = 0
                
                data.append({
                    'company_id': company_id,
                    'date': date,
                    'deposit_balance': max(0, deposit),
                    'used_loan': max(0, used_loan),
                    'unused_loan': max(0, unused_loan)
                })
        
        elif company_type == 'no_util_change':
            # Stable utilization pattern
            base_deposit = np.random.lognormal(11, 0.8)
            base_loan = np.random.lognormal(10, 0.8)
            stable_util = np.random.uniform(0.5, 0.7)  # Fixed utilization
            
            for j, date in enumerate(date_range):
                if np.random.random() < 0.8:  # 80% activity rate
                    deposit = base_deposit * np.random.normal(1, 0.2)  # Deposits vary
                    used_loan = base_loan * (stable_util + np.random.normal(0, 0.01))  # Very stable utilization
                    unused_loan = base_loan - used_loan
                else:
                    deposit = used_loan = unused_loan = 0
                
                data.append({
                    'company_id': company_id,
                    'date': date,
                    'deposit_balance': max(0, deposit),
                    'used_loan': max(0, used_loan),
                    'unused_loan': max(0, unused_loan)
                })
        
        elif company_type == 'no_deposit_change':
            # Stable deposit pattern
            base_deposit = np.random.lognormal(11, 0.5)  # More stable
            base_loan = np.random.lognormal(10, 1)
            
            for j, date in enumerate(date_range):
                if np.random.random() < 0.8:
                    deposit = base_deposit * np.random.normal(1, 0.05)  # Very stable deposits
                    used_loan = base_loan * np.random.uniform(0.3, 0.9)  # Utilization varies
                    unused_loan = base_loan - used_loan
                else:
                    deposit = used_loan = unused_loan = 0
                
                data.append({
                    'company_id': company_id,
                    'date': date,
                    'deposit_balance': max(0, deposit),
                    'used_loan': max(0, used_loan),
                    'unused_loan': max(0, unused_loan)
                })
        
        elif company_type == 'insufficient_util':
            # Insufficient utilization history (mainly deposits)
            base_deposit = np.random.lognormal(10, 1)
            
            for j, date in enumerate(date_range):
                if np.random.random() < 0.6:
                    deposit = base_deposit * np.random.normal(1, 0.15)
                    # Very occasional loan usage
                    if np.random.random() < 0.05:
                        used_loan = np.random.lognormal(8, 1)
                        unused_loan = used_loan * 0.5
                    else:
                        used_loan = unused_loan = 0
                else:
                    deposit = used_loan = unused_loan = 0
                
                data.append({
                    'company_id': company_id,
                    'date': date,
                    'deposit_balance': max(0, deposit),
                    'used_loan': max(0, used_loan),
                    'unused_loan': max(0, unused_loan)
                })
        
        elif company_type == 'insufficient_deposit':
            # Insufficient deposit history (mainly loans)
            base_loan = np.random.lognormal(9, 1)
            
            for j, date in enumerate(date_range):
                if np.random.random() < 0.6:
                    used_loan = base_loan * np.random.uniform(0.4, 0.8)
                    unused_loan = base_loan - used_loan
                    # Very occasional deposits
                    if np.random.random() < 0.05:
                        deposit = np.random.lognormal(9, 1)
                    else:
                        deposit = 0
                else:
                    deposit = used_loan = unused_loan = 0
                
                data.append({
                    'company_id': company_id,
                    'date': date,
                    'deposit_balance': max(0, deposit),
                    'used_loan': max(0, used_loan),
                    'unused_loan': max(0, unused_loan)
                })
        
        elif company_type == 'noisy_data':
            # Very irregular, noisy patterns
            for j, date in enumerate(date_range):
                if np.random.random() < 0.4:  # Sparse activity
                    # Highly variable amounts
                    deposit = np.random.lognormal(8, 2)  # High variance
                    used_loan = np.random.lognormal(7, 2)
                    unused_loan = np.random.lognormal(6, 1.5)
                else:
                    deposit = used_loan = unused_loan = 0
                
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
    print("Starting Simplified Integrated Bank Risk Analysis...")
    
    # Generate realistic test data
    test_df = generate_realistic_test_data(num_companies=120, days=1460)
    print(f"Generated test data: {len(test_df)} records for {test_df['company_id'].nunique()} companies")
    
    # Run simplified integrated analysis
    results = simplified_integrated_bank_risk_analysis(test_df)
    
    print("\n🎉 SIMPLIFIED ANALYSIS COMPLETE!")
    print("\n📊 Generated Files:")
    print("   1. clustering_validation_analysis.png - Clustering validation results")
    print("   2. simplified_integrated_overview.png - Comprehensive dashboard")
    
    print("\n🎯 KEY INSIGHTS:")
    clustering_results = results['sparse_analysis'].get('clustering_results')
    if clustering_results:
        optimal_clusters = clustering_results['optimal_clusters']
        silhouette_score = clustering_results['max_silhouette_score']
        
        if optimal_clusters == 6:
            print(f"   ✅ Clustering CONFIRMS 6 personas (Silhouette: {silhouette_score:.3f})")
        else:
            print(f"   ⚠️  Clustering suggests {optimal_clusters} personas (Silhouette: {silhouette_score:.3f})")
            print(f"   💡 Consider adjusting business logic based on data patterns")
    
    sparse_df = results['sparse_analysis'].get('persona_df', pd.DataFrame())
    if not sparse_df.empty:
        top_persona = sparse_df['persona'].value_counts().index[0]
        top_count = sparse_df['persona'].value_counts().iloc[0]
        print(f"   📈 Most common sparse persona: {top_persona} ({top_count} companies)")
    
    priority_df = results.get('priority_actions')
    if priority_df is not None and not priority_df.empty:
        high_priority = len(priority_df[priority_df['priority_score'] > 3.0])
        print(f"   🚨 High priority actions: {high_priority} companies need immediate attention")
