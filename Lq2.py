import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta

def generate_synthetic_data(n_clients=4000, days=365):
    """Generate synthetic client data with realistic patterns"""
    np.random.seed(42)
    
    # Generate dates
    date_range = pd.date_range(start='2024-01-01', periods=days, freq='D')
    
    # Create client segments
    segments = ['Retail', 'Consumer']
    client_segments = np.random.choice(segments, size=n_clients, p=[0.6, 0.4])
    
    # Initialize lists for data
    records = []
    
    # Generate base patterns for different client types
    for client_id in range(n_clients):
        # Base patterns with seasonal components
        seasonal_pattern = np.sin(np.linspace(0, 2*np.pi, days)) * 0.2
        trend = np.linspace(0, 0.1, days)  # Slight upward trend
        
        # Different patterns for retail vs consumer
        if client_segments[client_id] == 'Retail':
            base_loan_util = np.random.normal(0.65, 0.15)
            base_deposits = np.random.normal(1000000, 300000)
            volatility = 0.3
        else:
            base_loan_util = np.random.normal(0.45, 0.10)
            base_deposits = np.random.normal(500000, 150000)
            volatility = 0.2
        
        for day in range(days):
            # Add seasonality, trend and random noise
            loan_util = base_loan_util + seasonal_pattern[day] + trend[day] + np.random.normal(0, volatility * 0.1)
            deposits = base_deposits * (1 + seasonal_pattern[day] + trend[day] + np.random.normal(0, volatility))
            
            # Ensure valid ranges
            loan_util = np.clip(loan_util, 0, 1)
            deposits = max(deposits, 0)
            
            records.append({
                'date': date_range[day],
                'client_id': f'CLIENT_{client_id:04d}',
                'segment': client_segments[client_id],
                'loan_utilization': loan_util,
                'operating_deposits': deposits
            })
    
    return pd.DataFrame(records)

def calculate_client_metrics(df):
    """Calculate key liquidity metrics for each client"""
    metrics = df.groupby('client_id').agg({
        'loan_utilization': ['mean', 'std', 'min', 'max'],
        'operating_deposits': ['mean', 'std', 'min', 'max'],
        'segment': 'first'
    })
    
    # Flatten column names
    metrics.columns = ['loan_util_mean', 'loan_util_std', 'loan_util_min', 'loan_util_max',
                      'deposits_mean', 'deposits_std', 'deposits_min', 'deposits_max', 'segment']
    metrics = metrics.reset_index()
    
    # Calculate additional ratios
    metrics['deposit_volatility'] = metrics['deposits_std'] / metrics['deposits_mean']
    metrics['loan_volatility'] = metrics['loan_util_std'] / metrics['loan_util_mean']
    
    return metrics

def perform_clustering(metrics_df, n_clusters=4):
    """Perform clustering analysis on client metrics"""
    # Select features for clustering
    features = ['loan_util_mean', 'deposit_volatility', 'loan_volatility']
    X = metrics_df[features]
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Perform clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    metrics_df['cluster'] = kmeans.fit_predict(X_scaled)
    
    # Calculate cluster quality
    silhouette_avg = silhouette_score(X_scaled, metrics_df['cluster'])
    
    return metrics_df, silhouette_avg, kmeans.cluster_centers_

def analyze_temporal_patterns(df, client_clusters):
    """Analyze how clusters behave over time"""
    # Merge cluster assignments with original data
    df_with_clusters = df.merge(client_clusters[['client_id', 'cluster']], on='client_id')
    
    # Calculate daily averages per cluster
    daily_patterns = df_with_clusters.groupby(['date', 'cluster', 'segment']).agg({
        'loan_utilization': ['mean', 'std'],
        'operating_deposits': ['mean', 'std']
    }).reset_index()
    
    # Flatten column names
    daily_patterns.columns = ['date', 'cluster', 'segment', 
                            'loan_util_mean', 'loan_util_std',
                            'deposits_mean', 'deposits_std']
    
    return daily_patterns

def plot_cluster_analysis(client_metrics, daily_patterns):
    """Create comprehensive visualizations for cluster analysis"""
    plt.style.use('seaborn')
    fig = plt.figure(figsize=(20, 15))
    
    # 1. Scatter plot of clusters
    ax1 = plt.subplot(3, 2, 1)
    scatter = ax1.scatter(client_metrics['loan_util_mean'], 
                         client_metrics['deposit_volatility'],
                         c=client_metrics['cluster'],
                         cmap='viridis',
                         alpha=0.6)
    ax1.set_xlabel('Average Loan Utilization')
    ax1.set_ylabel('Deposit Volatility')
    ax1.set_title('Client Clusters')
    plt.colorbar(scatter)
    
    # 2. Loan utilization trends by cluster
    ax2 = plt.subplot(3, 2, 2)
    for cluster in sorted(daily_patterns['cluster'].unique()):
        cluster_data = daily_patterns[daily_patterns['cluster'] == cluster]
        ax2.plot(cluster_data['date'], cluster_data['loan_util_mean'], 
                label=f'Cluster {cluster}')
    ax2.set_xlabel('Date')
    ax2.set_ylabel('Average Loan Utilization')
    ax2.set_title('Loan Utilization Trends by Cluster')
    ax2.legend()
    
    # 3. Segment distribution within clusters
    ax3 = plt.subplot(3, 2, 3)
    segment_cluster_dist = pd.crosstab(client_metrics['cluster'], 
                                     client_metrics['segment'], 
                                     normalize='index') * 100
    segment_cluster_dist.plot(kind='bar', stacked=True, ax=ax3)
    ax3.set_xlabel('Cluster')
    ax3.set_ylabel('Percentage')
    ax3.set_title('Segment Distribution within Clusters')
    
    # 4. Operating deposits trends
    ax4 = plt.subplot(3, 2, 4)
    for cluster in sorted(daily_patterns['cluster'].unique()):
        cluster_data = daily_patterns[daily_patterns['cluster'] == cluster]
        ax4.plot(cluster_data['date'], 
                cluster_data['deposits_mean'] / 1e6,
                label=f'Cluster {cluster}')
    ax4.set_xlabel('Date')
    ax4.set_ylabel('Average Operating Deposits (Millions)')
    ax4.set_title('Operating Deposits Trends by Cluster')
    ax4.legend()
    
    # 5. Volatility comparison
    ax5 = plt.subplot(3, 2, 5)
    sns.boxplot(data=client_metrics, x='cluster', y='deposit_volatility', ax=ax5)
    ax5.set_xlabel('Cluster')
    ax5.set_ylabel('Deposit Volatility')
    ax5.set_title('Deposit Volatility Distribution by Cluster')
    
    # 6. Risk profile heatmap
    ax6 = plt.subplot(3, 2, 6)
    risk_metrics = client_metrics.groupby('cluster').agg({
        'loan_util_mean': 'mean',
        'deposit_volatility': 'mean',
        'loan_volatility': 'mean'
    })
    sns.heatmap(risk_metrics, annot=True, cmap='YlOrRd', ax=ax6)
    ax6.set_title('Risk Profile Heatmap by Cluster')
    
    plt.tight_layout()
    return fig

def calculate_risk_scores(client_metrics):
    """Calculate liquidity risk scores for each client"""
    # Normalize metrics for risk calculation
    scaler = StandardScaler()
    risk_features = ['loan_util_mean', 'deposit_volatility', 'loan_volatility']
    normalized_features = scaler.fit_transform(client_metrics[risk_features])
    
    # Calculate composite risk score (weighted average of normalized features)
    weights = np.array([0.4, 0.3, 0.3])  # Weights for different risk factors
    risk_scores = np.dot(normalized_features, weights)
    
    # Convert to percentile ranks
    client_metrics['risk_score'] = pd.Series(risk_scores).rank(pct=True)
    
    # Assign risk categories
    client_metrics['risk_category'] = pd.qcut(client_metrics['risk_score'], 
                                            q=5, 
                                            labels=['Very Low', 'Low', 'Medium', 'High', 'Very High'])
    
    return client_metrics

def generate_summary_report(client_metrics, daily_patterns, silhouette_score):
    """Generate a summary report of the analysis"""
    summary = {
        'cluster_profiles': client_metrics.groupby('cluster').agg({
            'client_id': 'count',
            'loan_util_mean': 'mean',
            'deposit_volatility': 'mean',
            'risk_score': 'mean',
            'segment': lambda x: x.value_counts().index[0]
        }).round(3),
        
        'risk_distribution': client_metrics.groupby('risk_category')['client_id'].count(),
        
        'segment_risk': client_metrics.groupby('segment')['risk_score'].agg(['mean', 'std']).round(3),
        
        'model_quality': {
            'silhouette_score': round(silhouette_score, 3),
            'n_clusters': len(client_metrics['cluster'].unique())
        }
    }
    
    return summary

def main():
    # Generate synthetic data
    print("Generating synthetic data...")
    df = generate_synthetic_data()
    
    # Calculate client metrics
    print("Calculating client metrics...")
    client_metrics = calculate_client_metrics(df)
    
    # Perform clustering
    print("Performing cluster analysis...")
    client_metrics, silhouette_score_val, cluster_centers = perform_clustering(client_metrics)
    
    # Analyze temporal patterns
    print("Analyzing temporal patterns...")
    daily_patterns = analyze_temporal_patterns(df, client_metrics)
    
    # Calculate risk scores
    print("Calculating risk scores...")
    client_metrics = calculate_risk_scores(client_metrics)
    
    # Generate visualizations
    print("Creating visualizations...")
    fig = plot_cluster_analysis(client_metrics, daily_patterns)
    
    # Generate summary report
    print("Generating summary report...")
    summary = generate_summary_report(client_metrics, daily_patterns, silhouette_score_val)
    
    # Print summary statistics
    print("\nAnalysis Summary:")
    print(f"Number of clients analyzed: {len(client_metrics)}")
    print(f"Clustering quality (Silhouette score): {silhouette_score_val:.3f}")
    print("\nCluster Profiles:")
    print(summary['cluster_profiles'])
    print("\nRisk Distribution:")
    print(summary['risk_distribution'])
    
    return df, client_metrics, daily_patterns, summary, fig

if __name__ == "__main__":
    df, client_metrics, daily_patterns, summary, fig = main()

*******************

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.ensemble import IsolationForest
from statsmodels.tsa.seasonal import seasonal_decompose
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta

def generate_synthetic_data(n_clients=4000, years=4):
    """Generate synthetic client data with realistic patterns over multiple years"""
    np.random.seed(42)
    days = years * 365
    
    # Generate dates
    date_range = pd.date_range(start='2020-01-01', periods=days, freq='D')
    
    # Create client segments with more granular categorization
    segments = ['Retail-Small', 'Retail-Medium', 'Consumer-Standard', 'Consumer-Premium']
    segment_weights = [0.3, 0.3, 0.2, 0.2]
    client_segments = np.random.choice(segments, size=n_clients, p=segment_weights)
    
    records = []
    
    for client_id in range(n_clients):
        # Complex seasonal patterns (yearly, quarterly, and monthly)
        yearly_pattern = np.sin(np.linspace(0, years * 2*np.pi, days)) * 0.15
        quarterly_pattern = np.sin(np.linspace(0, years * 8*np.pi, days)) * 0.1
        monthly_pattern = np.sin(np.linspace(0, years * 24*np.pi, days)) * 0.05
        
        # Long-term trend with random direction
        trend_direction = np.random.choice([-1, 1])
        trend = np.linspace(0, 0.2 * trend_direction, days)
        
        # Segment-specific parameters
        if 'Retail-Small' in client_segments[client_id]:
            base_loan_util = np.random.normal(0.7, 0.15)
            base_deposits = np.random.normal(500000, 150000)
            volatility = 0.35
        elif 'Retail-Medium' in client_segments[client_id]:
            base_loan_util = np.random.normal(0.6, 0.12)
            base_deposits = np.random.normal(1500000, 300000)
            volatility = 0.25
        elif 'Consumer-Standard' in client_segments[client_id]:
            base_loan_util = np.random.normal(0.5, 0.10)
            base_deposits = np.random.normal(300000, 100000)
            volatility = 0.3
        else:  # Consumer-Premium
            base_loan_util = np.random.normal(0.4, 0.08)
            base_deposits = np.random.normal(800000, 200000)
            volatility = 0.2
        
        for day in range(days):
            # Combine patterns
            seasonal_component = (yearly_pattern[day] + 
                               quarterly_pattern[day] + 
                               monthly_pattern[day])
            
            # Add random shocks (rare events)
            shock = 0
            if np.random.random() < 0.01:  # 1% chance of shock
                shock = np.random.normal(0, 0.2)
            
            loan_util = (base_loan_util + 
                        seasonal_component + 
                        trend[day] + 
                        shock + 
                        np.random.normal(0, volatility * 0.1))
            
            deposits = (base_deposits * 
                       (1 + seasonal_component + 
                        trend[day] + 
                        shock + 
                        np.random.normal(0, volatility)))
            
            # Ensure valid ranges
            loan_util = np.clip(loan_util, 0, 1)
            deposits = max(deposits, 0)
            
            records.append({
                'date': date_range[day],
                'client_id': f'CLIENT_{client_id:04d}',
                'segment': client_segments[client_id],
                'loan_utilization': loan_util,
                'operating_deposits': deposits
            })
    
    return pd.DataFrame(records)

def detect_anomalies(df, feature_cols):
    """Detect anomalies in client behavior using Isolation Forest"""
    scaler = RobustScaler()
    X = scaler.fit_transform(df[feature_cols])
    
    iso_forest = IsolationForest(contamination=0.1, random_state=42)
    anomalies = iso_forest.fit_predict(X)
    
    return anomalies == -1

def calculate_advanced_metrics(df):
    """Calculate advanced liquidity metrics with statistical measures"""
    # Group by client
    metrics = df.groupby('client_id').agg({
        'loan_utilization': ['mean', 'std', 'min', 'max', 'skew', 'kurt'],
        'operating_deposits': ['mean', 'std', 'min', 'max', 'skew', 'kurt'],
        'segment': 'first'
    })
    
    # Flatten column names
    metrics.columns = ['loan_util_mean', 'loan_util_std', 'loan_util_min', 'loan_util_max',
                      'loan_util_skew', 'loan_util_kurt', 'deposits_mean', 'deposits_std',
                      'deposits_min', 'deposits_max', 'deposits_skew', 'deposits_kurt',
                      'segment']
    metrics = metrics.reset_index()
    
    # Calculate additional ratios and metrics
    metrics['deposit_volatility'] = metrics['deposits_std'] / metrics['deposits_mean']
    metrics['loan_volatility'] = metrics['loan_util_std'] / metrics['loan_util_mean']
    metrics['deposit_stability'] = (metrics['deposits_max'] - metrics['deposits_min']) / metrics['deposits_mean']
    
    # Detect anomalies
    anomaly_features = ['loan_util_mean', 'deposit_volatility', 'loan_volatility']
    metrics['is_anomaly'] = detect_anomalies(metrics, anomaly_features)
    
    return metrics

def analyze_seasonal_patterns(df):
    """Analyze seasonal patterns in the data"""
    # Calculate daily averages
    daily_avg = df.groupby('date').agg({
        'loan_utilization': 'mean',
        'operating_deposits': 'mean'
    }).reset_index()
    
    # Perform seasonal decomposition for both metrics
    seasonal_results = {}
    for metric in ['loan_utilization', 'operating_deposits']:
        decomposition = seasonal_decompose(daily_avg[metric], 
                                        period=365,  # Yearly seasonality
                                        extrapolate_trend='freq')
        seasonal_results[metric] = {
            'trend': decomposition.trend,
            'seasonal': decomposition.seasonal,
            'residual': decomposition.resid
        }
    
    return daily_avg, seasonal_results

def calculate_transition_matrices(df, client_metrics, window_size=30):
    """Calculate transition probabilities between risk states"""
    # Define risk states based on percentiles
    client_metrics['risk_state'] = pd.qcut(client_metrics['risk_score'], 
                                         q=5, 
                                         labels=['VL', 'L', 'M', 'H', 'VH'])
    
    # Create windows for analysis
    df['window'] = (df['date'] - df['date'].min()).dt.days // window_size
    
    transitions = []
    for client_id in df['client_id'].unique():
        client_data = df[df['client_id'] == client_id].copy()
        client_risk = client_metrics[client_metrics['client_id'] == client_id]['risk_state'].iloc[0]
        
        # Calculate changes in loan utilization and deposits
        client_data['loan_util_change'] = client_data['loan_utilization'].diff()
        client_data['deposits_change'] = client_data['operating_deposits'].diff()
        
        # Record transitions between states
        for w in client_data['window'].unique()[:-1]:
            current_window = client_data[client_data['window'] == w]
            next_window = client_data[client_data['window'] == w + 1]
            
            if len(current_window) > 0 and len(next_window) > 0:
                transitions.append({
                    'client_id': client_id,
                    'from_window': w,
                    'to_window': w + 1,
                    'risk_state': client_risk,
                    'loan_util_change': next_window['loan_utilization'].mean() - current_window['loan_utilization'].mean(),
                    'deposits_change': next_window['operating_deposits'].mean() - current_window['operating_deposits'].mean()
                })
    
    return pd.DataFrame(transitions)

def analyze_cohort_behavior(df, client_metrics):
    """Analyze behavior patterns by cohort"""
    # Create cohorts based on initial deposit levels
    client_metrics['deposit_cohort'] = pd.qcut(client_metrics['deposits_mean'], 
                                             q=5, 
                                             labels=['C1', 'C2', 'C3', 'C4', 'C5'])
    
    # Merge cohort information with main dataset
    df_with_cohorts = df.merge(client_metrics[['client_id', 'deposit_cohort']], 
                              on='client_id')
    
    # Calculate metrics by cohort over time
    cohort_metrics = df_with_cohorts.groupby(['date', 'deposit_cohort']).agg({
        'loan_utilization': ['mean', 'std'],
        'operating_deposits': ['mean', 'std']
    }).reset_index()
    
    return cohort_metrics

def calculate_stress_scenarios(client_metrics, daily_patterns):
    """Calculate stress scenarios based on historical patterns"""
    stress_scenarios = {}
    
    # Severe deposit outflow scenario
    deposit_percentiles = daily_patterns.groupby('cluster')['deposits_mean'].quantile([0.01, 0.05, 0.10])
    stress_scenarios['severe_deposit_outflow'] = deposit_percentiles
    
    # High utilization scenario
    util_percentiles = daily_patterns.groupby('cluster')['loan_util_mean'].quantile([0.90, 0.95, 0.99])
    stress_scenarios['high_utilization'] = util_percentiles
    
    # Combined stress scenario
    stress_scenarios['combined_stress'] = client_metrics.groupby('cluster').agg({
        'deposits_mean': lambda x: x.quantile(0.05),
        'loan_util_mean': lambda x: x.quantile(0.95)
    })
    
    return stress_scenarios

def generate_enhanced_report(client_metrics, daily_patterns, seasonal_results, 
                           transitions, stress_scenarios):
    """Generate comprehensive analysis report"""
    report = {
        'client_segmentation': {
            'cluster_profiles': client_metrics.groupby('cluster').agg({
                'client_id': 'count',
                'loan_util_mean': ['mean', 'std'],
                'deposit_volatility': ['mean', 'std'],
                'risk_score': ['mean', 'std']
            }),
            'segment_distribution': pd.crosstab(client_metrics['cluster'], 
                                              client_metrics['segment'])
        },
        
        'risk_analysis': {
            'risk_distribution': client_metrics.groupby('risk_category')['client_id'].count(),
            'high_risk_clients': client_metrics[client_metrics['risk_score'] > 0.8].shape[0],
            'anomalies_detected': client_metrics['is_anomaly'].sum()
        },
        
        'seasonal_patterns': {
            'loan_utilization': {
                'yearly_amplitude': seasonal_results['loan_utilization']['seasonal'].max() - 
                                  seasonal_results['loan_utilization']['seasonal'].min(),
                'trend_direction': 'Increasing' if seasonal_results['loan_utilization']['trend'][-1] > 
                                  seasonal_results['loan_utilization']['trend'][0] else 'Decreasing'
            },
            'operating_deposits': {
                'yearly_amplitude': seasonal_results['operating_deposits']['seasonal'].max() - 
                                  seasonal_results['operating_deposits']['seasonal'].min(),
                'trend_direction': 'Increasing' if seasonal_results['operating_deposits']['trend'][-1] > 
                                  seasonal_results['operating_deposits']['trend'][0] else 'Decreasing'
            }
        },
        
        'stress_scenarios': stress_scenarios
    }
    
    return report

def plot_enhanced_analysis(client_metrics, daily_patterns, seasonal_results, 
                         transitions, cohort_metrics):
    """Create enhanced visualizations for analysis"""
    plt.style.use('seaborn')
    fig = plt.figure(figsize=(25, 20))
    
    # 1. Cluster visualization with risk overlay
    ax1 = plt.subplot(4, 2, 1)
    scatter = ax1.scatter(client_metrics['loan_util_mean'], 
                         client_metrics['deposit_volatility'],
                         c=client_metrics['risk_score'],
                         cmap='RdYlGn_r',
                         alpha=0.6)
    ax1.set_xlabel('Average Loan Utilization')
    ax1.set_ylabel('Deposit Volatility')
    ax1.set_title('Client Risk Profile Distribution')
    plt.colorbar(scatter, label='Risk Score')
    
    # 2. Seasonal patterns
    ax2 = plt.subplot(4, 2, 2)
    ax2.plot(seasonal_results['loan_utilization']['seasonal'], 
             label='Loan Utilization')
    ax2.plot(seasonal_results['operating_deposits']['seasonal'] / 
             seasonal_results['operating_deposits']['seasonal'].max(),
             label='Deposits (Normalized)')
    ax2.set_xlabel('Day of Year')
    ax2.set_title('Seasonal Patterns')
    ax2.legend()
    
    # Additional plots...
    
    plt.tight_layout()
    return fig

def main():
    # Generate extended synthetic data
    print("Generating synthetic data...")
    df = generate_synthetic_data(years=4)
    
    # Calculate advanced metrics
    print("Calculating advanced metrics...")
    client_metrics = calculate_advanced_metrics(df)
    
    # Perform clustering and risk scoring
    print("Performing cluster analysis...")
    client_metrics, silhouette_score_val, _ = perform_clustering(client_metrics)
    client_metrics = calculate_risk_scores(client_metrics)
    
    # Analyze patterns
    print("Analyzing patterns...")
    daily_patterns = analyze_temporal_patterns(df, client_metrics)
    _, seasonal_results = analyze_seasonal_patterns(df)
    transitions = calculate_transition_matrices(df, client_metrics)
    cohort_metrics = analyze_cohort_behavior(df, client_metrics)
    
    # Calculate stress scenarios
    print("Calculating stress scenarios...")
    stress_scenarios = calculate_stress_scenarios(client_metrics, daily_patterns)

def main():
    # Generate extended synthetic data
    print("Generating synthetic data...")
    df = generate_synthetic_data(years=4)
    
    # Calculate advanced metrics
    print("Calculating advanced metrics...")
    client_metrics = calculate_advanced_metrics(df)
    
    # Perform clustering and risk scoring
    print("Performing cluster analysis...")
    client_metrics, silhouette_score_val, _ = perform_clustering(client_metrics)
    client_metrics = calculate_risk_scores(client_metrics)
    
    # Analyze patterns
    print("Analyzing patterns...")
    daily_patterns = analyze_temporal_patterns(df, client_metrics)
    _, seasonal_results = analyze_seasonal_patterns(df)
    transitions = calculate_transition_matrices(df, client_metrics)
    cohort_metrics = analyze_cohort_behavior(df, client_metrics)
    
    # Calculate stress scenarios
    print("Calculating stress scenarios...")
    stress_scenarios = calculate_stress_scenarios(client_metrics, daily_patterns)
    
    # Generate enhanced report
    print("Generating comprehensive report...")
    report = generate_enhanced_report(client_metrics, daily_patterns, 
                                    seasonal_results, transitions, 
                                    stress_scenarios)
    
    # Create visualizations
    print("Creating enhanced visualizations...")
    fig = plot_enhanced_analysis(client_metrics, daily_patterns, 
                               seasonal_results, transitions, 
                               cohort_metrics)
    
    # Print key findings
    print("\nKey Analysis Findings:")
    print(f"Number of clients analyzed: {len(client_metrics)}")
    print(f"Clustering quality (Silhouette score): {silhouette_score_val:.3f}")
    print(f"High-risk clients identified: {report['risk_analysis']['high_risk_clients']}")
    print(f"Anomalies detected: {report['risk_analysis']['anomalies_detected']}")
    
    # Export results
    client_metrics.to_csv('client_metrics_enhanced.csv', index=False)
    daily_patterns.to_csv('daily_patterns_enhanced.csv', index=False)
    
    return {
        'data': df,
        'client_metrics': client_metrics,
        'daily_patterns': daily_patterns,
        'seasonal_results': seasonal_results,
        'transitions': transitions,
        'cohort_metrics': cohort_metrics,
        'stress_scenarios': stress_scenarios,
        'report': report,
        'visualizations': fig
    }

def validate_results(results):
    """Validate the analysis results and data quality"""
    validations = {
        'data_completeness': True,
        'data_quality': True,
        'analysis_validity': True,
        'warnings': []
    }
    
    # Check for missing values
    if results['data'].isnull().any().any():
        validations['data_completeness'] = False
        validations['warnings'].append("Missing values detected in raw data")
    
    # Check for outliers in client metrics
    metrics = results['client_metrics']
    for col in ['loan_util_mean', 'deposit_volatility']:
        z_scores = np.abs(stats.zscore(metrics[col]))
        if (z_scores > 3).any():
            validations['warnings'].append(f"Outliers detected in {col}")
    
    # Validate clustering quality
    if hasattr(results['report'], 'silhouette_score') and results['report'].silhouette_score < 0.3:
        validations['analysis_validity'] = False
        validations['warnings'].append("Poor clustering quality detected")
    
    # Check for temporal consistency
    daily_patterns = results['daily_patterns']
    if daily_patterns['date'].nunique() != len(daily_patterns['date'].unique()):
        validations['data_quality'] = False
        validations['warnings'].append("Duplicate dates detected in patterns")
    
    return validations

def generate_monitoring_alerts(results, thresholds=None):
    """Generate monitoring alerts based on analysis results"""
    if thresholds is None:
        thresholds = {
            'high_risk_ratio': 0.1,  # 10% of clients
            'deposit_volatility': 0.3,
            'loan_utilization': 0.8,
            'anomaly_ratio': 0.05
        }
    
    alerts = []
    metrics = results['client_metrics']
    
    # Risk concentration alerts
    high_risk_ratio = (metrics['risk_score'] > 0.8).mean()
    if high_risk_ratio > thresholds['high_risk_ratio']:
        alerts.append({
            'level': 'HIGH',
            'type': 'Risk Concentration',
            'message': f"High risk client concentration: {high_risk_ratio:.1%}"
        })
    
    # Volatility alerts
    high_vol_clients = (metrics['deposit_volatility'] > thresholds['deposit_volatility']).sum()
    if high_vol_clients > 0:
        alerts.append({
            'level': 'MEDIUM',
            'type': 'Deposit Volatility',
            'message': f"{high_vol_clients} clients show high deposit volatility"
        })
    
    # Loan utilization alerts
    high_util_clients = (metrics['loan_util_mean'] > thresholds['loan_utilization']).sum()
    if high_util_clients > 0:
        alerts.append({
            'level': 'MEDIUM',
            'type': 'Loan Utilization',
            'message': f"{high_util_clients} clients show high loan utilization"
        })
    
    # Anomaly alerts
    anomaly_ratio = metrics['is_anomaly'].mean()
    if anomaly_ratio > thresholds['anomaly_ratio']:
        alerts.append({
            'level': 'HIGH',
            'type': 'Anomaly Detection',
            'message': f"High anomaly ratio detected: {anomaly_ratio:.1%}"
        })
    
    return alerts

if __name__ == "__main__":
    # Run main analysis
    results = main()
    
    # Validate results
    validations = validate_results(results)
    if validations['warnings']:
        print("\nValidation Warnings:")
        for warning in validations['warnings']:
            print(f"- {warning}")
    
    # Generate monitoring alerts
    alerts = generate_monitoring_alerts(results)
    if alerts:
        print("\nMonitoring Alerts:")
        for alert in alerts:
            print(f"{alert['level']} - {alert['type']}: {alert['message']}")
    
    # Save visualizations
    results['visualizations'].savefig('liquidity_analysis_enhanced.png')
    
    print("\nAnalysis completed successfully!")

