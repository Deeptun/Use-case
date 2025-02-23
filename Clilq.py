import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score
from lightgbm import LGBMClassifier
import shap
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from statsmodels.stats.diagnostic import acorr_ljungbox
from scipy import stats
from scipy.stats import pearsonr, spearmanr, kendalltau
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

def generate_synthetic_data(n_clients=400, n_days=730):
    """
    Enhanced synthetic data generation with more realistic patterns and seasonality
    """
    np.random.seed(42)
    dates = pd.date_range(end='2024-02-21', periods=n_days)
    clients = range(1, n_clients + 1)
    
    data = []
    for client in clients:
        # Multiple seasonal components
        annual_seasonality = np.sin(np.linspace(0, 2*np.pi, 365))
        quarterly_seasonality = np.sin(np.linspace(0, 8*np.pi, 365))
        monthly_seasonality = np.sin(np.linspace(0, 24*np.pi, 365))
        
        # Combine seasonalities with different weights
        seasonal_pattern = (0.5 * annual_seasonality + 
                          0.3 * quarterly_seasonality + 
                          0.2 * monthly_seasonality)
        seasonal_pattern = np.repeat(seasonal_pattern, 2)[:n_days]
        
        # Add trend and noise
        trend = np.linspace(0, np.random.uniform(-0.5, 0.5), n_days)
        noise = np.random.normal(0, 0.05, n_days)
        
        # Generate base metrics
        base_loan_util = np.random.uniform(0.4, 0.8)
        loan_util = base_loan_util + 0.2 * seasonal_pattern + trend + noise
        loan_util = np.clip(loan_util, 0, 1)
        
        # Generate correlated deposits
        correlation = np.random.uniform(-0.8, 0.8)
        base_deposits = np.random.uniform(1e6, 1e7)
        deposits = base_deposits * (1 + correlation * (loan_util - base_loan_util) + 
                                  seasonal_pattern + np.random.normal(0, 0.1, n_days))
        
        # Add sudden changes for some clients
        if np.random.random() < 0.1:
            change_point = np.random.randint(n_days//2, n_days)
            deposits[change_point:] *= (1 - np.random.uniform(0.2, 0.4))
            loan_util[change_point:] *= (1 + np.random.uniform(0.1, 0.3))
        
        # Generate more detailed financial metrics
        for i, date in enumerate(dates):
            data.append({
                'date': date,
                'client_id': client,
                'loan_utilization': loan_util[i],
                'deposits': deposits[i],
                'risk_rating': np.random.choice(['1+', '1', '1-', '2+', '2', '2-', '3+', '3', '3-']),
                'sector': np.random.choice(['Retail', 'Manufacturing', 'Technology', 'Healthcare']),
                'revenue': np.random.uniform(1e7, 1e8) * (1 + seasonal_pattern[i]),
                'ebitda_margin': np.random.uniform(0.05, 0.25),
                'debt_service_ratio': np.random.uniform(1.1, 2.5),
                'working_capital_ratio': np.random.uniform(0.8, 2.0),
                'current_ratio': np.random.uniform(1.0, 3.0),
                'quick_ratio': np.random.uniform(0.8, 2.5),
                'inventory_turnover': np.random.uniform(4, 12),
                'receivables_days': np.random.uniform(30, 90)
            })
    
    return pd.DataFrame(data)

def perform_statistical_tests(df, client_id, metric, window=90):
    """
    Perform comprehensive statistical tests on time series data
    """
    client_data = df[df['client_id'] == client_id][metric].tail(window)
    
    # Stationarity test (Augmented Dickey-Fuller)
    adf_result = adfuller(client_data)
    
    # Autocorrelation test (Ljung-Box)
    lb_result = acorr_ljungbox(client_data, lags=[10], return_df=True)
    
    # Normality test (Shapiro-Wilk)
    _, sw_pvalue = stats.shapiro(client_data)
    
    # Trend analysis
    trend_coef = np.polyfit(range(len(client_data)), client_data, 1)[0]
    
    return {
        'adf_pvalue': adf_result[1],
        'lb_pvalue': lb_result['lb_pvalue'].iloc[0],
        'sw_pvalue': sw_pvalue,
        'trend_coefficient': trend_coef
    }

def detect_anomalies(series, window=30, thresh=3):
    """
    Detect anomalies using rolling statistics and Z-score
    """
    rolling_mean = series.rolling(window=window).mean()
    rolling_std = series.rolling(window=window).std()
    z_scores = (series - rolling_mean) / rolling_std
    
    return np.abs(z_scores) > thresh

def identify_advanced_patterns(df, lookback_periods=[30, 90, 180]):
    """
    Enhanced pattern detection with statistical validation
    """
    pattern_results = []
    
    for client in df['client_id'].unique():
        client_data = df[df['client_id'] == client].sort_values('date')
        
        for period in lookback_periods:
            recent_data = client_data.tail(period)
            
            # Statistical tests
            loan_stats = perform_statistical_tests(df, client, 'loan_utilization', period)
            deposit_stats = perform_statistical_tests(df, client, 'deposits', period)
            
            # Correlation analysis
            pearson_corr, _ = pearsonr(recent_data['loan_utilization'], recent_data['deposits'])
            spearman_corr, _ = spearmanr(recent_data['loan_utilization'], recent_data['deposits'])
            
            # Trend analysis
            loan_trend = np.polyfit(range(len(recent_data)), recent_data['loan_utilization'], 1)[0]
            deposit_trend = np.polyfit(range(len(recent_data)), recent_data['deposits'], 1)[0]
            
            # Volatility analysis
            loan_volatility = recent_data['loan_utilization'].std()
            deposit_volatility = recent_data['deposits'].std() / recent_data['deposits'].mean()
            
            # Pattern detection with statistical significance
            patterns = {
                'increasing_loan_decreasing_deposit': (
                    loan_trend > 0 and deposit_trend < 0 and 
                    loan_stats['adf_pvalue'] < 0.05 and 
                    deposit_stats['adf_pvalue'] < 0.05
                ),
                'high_volatility': (
                    loan_volatility > np.percentile(df['loan_utilization'].std(), 75) or
                    deposit_volatility > np.percentile(df['deposits'].std(), 75)
                ),
                'negative_correlation': pearson_corr < -0.3 and spearman_corr < -0.3,
                'deposit_deterioration': (
                    deposit_stats['trend_coefficient'] < 0 and 
                    deposit_stats['adf_pvalue'] < 0.05
                ),
                'sudden_changes': any(detect_anomalies(recent_data['deposits']))
            }
            
            # Additional financial metrics analysis
            financial_patterns = {
                'deteriorating_ratios': (
                    recent_data['current_ratio'].iloc[-1] < recent_data['current_ratio'].mean() and
                    recent_data['quick_ratio'].iloc[-1] < recent_data['quick_ratio'].mean()
                ),
                'increasing_receivables': (
                    recent_data['receivables_days'].iloc[-1] > 
                    recent_data['receivables_days'].mean() * 1.2
                )
            }
            
            pattern_results.append({
                'client_id': client,
                'period': period,
                **patterns,
                **financial_patterns,
                'pearson_correlation': pearson_corr,
                'spearman_correlation': spearman_corr,
                'loan_trend': loan_trend,
                'deposit_trend': deposit_trend,
                'risk_score': sum(patterns.values()) + sum(financial_patterns.values())
            })
    
    return pd.DataFrame(pattern_results)

def perform_cohort_analysis(df, pattern_df):
    """
    Advanced cohort analysis with temporal transitions
    """
    # Create initial cohorts based on entry characteristics
    client_entry_data = df.groupby('client_id').agg({
        'date': 'min',
        'loan_utilization': 'first',
        'deposits': 'first',
        'sector': 'first'
    })
    
    # Define cohort categories
    client_entry_data['loan_cohort'] = pd.qcut(client_entry_data['loan_utilization'], 
                                              q=3, labels=['Low', 'Medium', 'High'])
    client_entry_data['deposit_cohort'] = pd.qcut(client_entry_data['deposits'], 
                                                 q=3, labels=['Low', 'Medium', 'High'])
    
    # Track cohort transitions
    transition_data = []
    for client in df['client_id'].unique():
        client_patterns = pattern_df[pattern_df['client_id'] == client]
        
        if len(client_patterns) >= 2:
            initial_risk = client_patterns.iloc[0]['risk_score']
            final_risk = client_patterns.iloc[-1]['risk_score']
            
            transition_data.append({
                'client_id': client,
                'initial_risk': initial_risk,
                'final_risk': final_risk,
                'risk_change': final_risk - initial_risk,
                'sector': client_entry_data.loc[client, 'sector'],
                'loan_cohort': client_entry_data.loc[client, 'loan_cohort'],
                'deposit_cohort': client_entry_data.loc[client, 'deposit_cohort']
            })
    
    return pd.DataFrame(transition_data)

def create_advanced_visualizations(df, pattern_df, cohort_df):
    """
    Create sophisticated visualizations for analysis
    """
    # 1. Risk Score Evolution Dashboard
    risk_fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=("Risk Score Distribution", "Risk Score by Sector",
                       "Risk Score vs Correlation", "Risk Score Timeline")
    )
    
    # Risk score distribution
    risk_fig.add_trace(
        go.Histogram(x=pattern_df['risk_score'], name="Risk Score"),
        row=1, col=1
    )
    
    # Risk score by sector
    sector_risk = cohort_df.groupby('sector')['final_risk'].mean().sort_values()
    risk_fig.add_trace(
        go.Bar(x=sector_risk.index, y=sector_risk.values, name="Sector Risk"),
        row=1, col=2
    )
    
    # Risk score vs correlation
    risk_fig.add_trace(
        go.Scatter(x=pattern_df['pearson_correlation'], 
                  y=pattern_df['risk_score'],
                  mode='markers',
                  name="Risk vs Correlation"),
        row=2, col=1
    )
    
    # Risk score timeline
    timeline_data = pattern_df.groupby('period')['risk_score'].mean()
    risk_fig.add_trace(
        go.Scatter(x=timeline_data.index, 
                  y=timeline_data.values,
                  mode='lines+markers',
                  name="Risk Timeline"),
        row=2, col=2
    )
    
    # 2. Cohort Transition Sankey Diagram
    cohort_transitions = cohort_df.groupby(['loan_cohort', 'deposit_cohort', 'sector']).size()
    
    sankey_fig = go.Figure(data=[go.Sankey(
        node = {
            "label": list(cohort_transitions.index.get_level_values(0).unique()) + 
                    list(cohort_transitions.index.get_level_values(1).unique()) +
                    list(cohort_transitions.index.get_level_values(2).unique())
        },
        link = {
            "source": [i for i, _ in enumerate(cohort_transitions)],
            "target": [len(cohort_transitions) + i for i, _ in enumerate(cohort_transitions)],
            "value": cohort_transitions.values
        }
    )])
    
    # 3. Pattern Detection Heatmap
    pattern_cols = ['increasing_loan_decreasing_deposit', 'high_volatility', 
                   'negative_correlation', 'deposit_deterioration', 'sudden_changes']
    pattern_corr = pattern_df[pattern_cols].corr()
    
    heatmap_fig = go.Figure(data=go.Heatmap(
        z=pattern_corr.values,
        x=pattern_corr.columns,
        y=pattern_corr.columns,
        colorscale='RdBu'
    ))
    
    return risk_fig, sankey_fig, heatmap_fig

def main():
    # Generate synthetic data
    df = generate_synthetic_data()
    
    # Perform advanced pattern detection
    pattern_df = identify_advanced_patterns(df)
    
    # Perform cohort analysis
    cohort_df = perform_cohort_analysis(df, pattern_df)
    
    # Create visualizations
    risk_fig, sankey_fig, heatmap_fig = create_advanced_visualizations(df, pattern_df, cohort_df)
    
    # Save visualizations
    risk_fig.write_html("risk_analysis_dashboard.html")
    sankey_fig.write_html("cohort_transitions.html")
    heatmap_fig.write_html("pattern_correlations.html")
    
    # Print summary statistics
    print("\nAdvanced Analysis Summary:")
    print(f"Total clients analyzed: {df['client_id'].nunique()}")
    print(f"High risk clients (score >= 3): {len(pattern_df[pattern_df['risk_score'] >= 3])}")

def calculate_risk_metrics(pattern_df, cohort_df):
    """
    Calculate comprehensive risk metrics and statistical insights
    """
    risk_metrics = {
        # Pattern Statistics
        'pattern_stats': {
            'increasing_loan_decreasing_deposit': pattern_df['increasing_loan_decreasing_deposit'].mean(),
            'high_volatility': pattern_df['high_volatility'].mean(),
            'negative_correlation': pattern_df['negative_correlation'].mean(),
            'deposit_deterioration': pattern_df['deposit_deterioration'].mean(),
            'sudden_changes': pattern_df['sudden_changes'].mean()
        },
        
        # Risk Score Statistics
        'risk_score_stats': {
            'mean': pattern_df['risk_score'].mean(),
            'median': pattern_df['risk_score'].median(),
            'std': pattern_df['risk_score'].std(),
            'skew': pattern_df['risk_score'].skew(),
            'kurtosis': pattern_df['risk_score'].kurtosis()
        },
        
        # Correlation Statistics
        'correlation_stats': {
            'mean_pearson': pattern_df['pearson_correlation'].mean(),
            'mean_spearman': pattern_df['spearman_correlation'].mean(),
            'correlation_volatility': pattern_df['pearson_correlation'].std()
        },
        
        # Cohort Transition Statistics
        'cohort_stats': {
            'mean_risk_change': cohort_df['risk_change'].mean(),
            'positive_transitions': (cohort_df['risk_change'] > 0).mean(),
            'negative_transitions': (cohort_df['risk_change'] < 0).mean()
        }
    }
    
    return risk_metrics

def generate_risk_report(risk_metrics, pattern_df, cohort_df):
    """
    Generate a comprehensive risk report with statistical insights
    """
    report = []
    
    # Overall Risk Assessment
    report.append("RISK ASSESSMENT REPORT")
    report.append("=" * 50)
    
    # Pattern Analysis
    report.append("\n1. Pattern Detection Analysis")
    report.append("-" * 30)
    for pattern, freq in risk_metrics['pattern_stats'].items():
        report.append(f"{pattern}: {freq:.2%} of observations")
    
    # Risk Score Distribution
    report.append("\n2. Risk Score Statistics")
    report.append("-" * 30)
    for metric, value in risk_metrics['risk_score_stats'].items():
        report.append(f"{metric}: {value:.3f}")
    
    # Correlation Analysis
    report.append("\n3. Correlation Analysis")
    report.append("-" * 30)
    for metric, value in risk_metrics['correlation_stats'].items():
        report.append(f"{metric}: {value:.3f}")
    
    # Cohort Analysis
    report.append("\n4. Cohort Transition Analysis")
    report.append("-" * 30)
    for metric, value in risk_metrics['cohort_stats'].items():
        report.append(f"{metric}: {value:.3f}")
    
    # High Risk Client Analysis
    high_risk_clients = pattern_df[pattern_df['risk_score'] >= 3]
    report.append("\n5. High Risk Client Analysis")
    report.append("-" * 30)
    report.append(f"Number of high risk clients: {len(high_risk_clients)}")
    report.append(f"Percentage of portfolio: {len(high_risk_clients)/len(pattern_df):.2%}")
    
    # Sector Risk Analysis
    sector_risk = cohort_df.groupby('sector')['final_risk'].agg(['mean', 'std'])
    report.append("\n6. Sector Risk Analysis")
    report.append("-" * 30)
    for sector in sector_risk.index:
        report.append(f"{sector}:")
        report.append(f"  - Mean Risk: {sector_risk.loc[sector, 'mean']:.3f}")
        report.append(f"  - Risk Std: {sector_risk.loc[sector, 'std']:.3f}")
    
    return "\n".join(report)

def main():
    # Generate synthetic data
    print("Generating synthetic data...")
    df = generate_synthetic_data()
    
    # Perform advanced pattern detection
    print("Performing pattern detection...")
    pattern_df = identify_advanced_patterns(df)
    
    # Perform cohort analysis
    print("Performing cohort analysis...")
    cohort_df = perform_cohort_analysis(df, pattern_df)
    
    # Calculate risk metrics
    print("Calculating risk metrics...")
    risk_metrics = calculate_risk_metrics(pattern_df, cohort_df)
    
    # Generate risk report
    print("Generating risk report...")
    risk_report = generate_risk_report(risk_metrics, pattern_df, cohort_df)
    
    # Create visualizations
    print("Creating visualizations...")
    risk_fig, sankey_fig, heatmap_fig = create_advanced_visualizations(df, pattern_df, cohort_df)
    
    # Save outputs
    risk_fig.write_html("risk_analysis_dashboard.html")
    sankey_fig.write_html("cohort_transitions.html")
    heatmap_fig.write_html("pattern_correlations.html")
    
    with open("risk_report.txt", "w") as f:
        f.write(risk_report)
    
    # Print summary statistics
    print("\nAnalysis Complete!")
    print("\nKey Findings:")
    print(f"- Total clients analyzed: {df['client_id'].nunique()}")
    print(f"- High risk clients: {len(pattern_df[pattern_df['risk_score'] >= 3])}")
    print(f"- Average risk score: {risk_metrics['risk_score_stats']['mean']:.2f}")
    print(f"- Most common pattern: {max(risk_metrics['pattern_stats'].items(), key=lambda x: x[1])[0]}")
    print(f"- Sectors with highest risk: {cohort_df.groupby('sector')['final_risk'].mean().nlargest(2).index.tolist()}")
    
    print("\nOutputs generated:")
    print("1. risk_analysis_dashboard.html - Interactive dashboard of risk metrics")
    print("2. cohort_transitions.html - Sankey diagram of client transitions")
    print("3. pattern_correlations.html - Heatmap of pattern correlations")
    print("4. risk_report.txt - Detailed risk assessment report")

if __name__ == "__main__":
    main()
