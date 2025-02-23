import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from lightgbm import LGBMClassifier
import shap
from statsmodels.tsa.seasonal import seasonal_decompose
from scipy.stats import pearsonr
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

def generate_synthetic_data(n_clients=100, n_days=1460):
    # Previous data generation code remains the same
    # [Previous implementation]
    np.random.seed(42)
    dates = pd.date_range(end='2024-02-21', periods=n_days)
    clients = range(1, n_clients + 1)
    
    data = []
    for client in clients:
        seasonal_pattern = np.sin(np.linspace(0, 4*np.pi, n_days)) * np.random.uniform(0.1, 0.3)
        trend = np.linspace(0, np.random.uniform(-0.5, 0.5), n_days)
        
        base_loan_util = np.random.uniform(0.4, 0.8)
        loan_util = base_loan_util + seasonal_pattern + trend
        loan_util = np.clip(loan_util, 0, 1)
        
        correlation = np.random.uniform(-0.8, 0.8)
        base_deposits = np.random.uniform(1e6, 1e7)
        deposits = base_deposits * (1 + correlation * (loan_util - base_loan_util) + 
                                  np.random.normal(0, 0.1, n_days))
        
        base_rating = np.random.choice(['1+', '1', '1-', '2+', '2', '2-', '3+', '3', '3-', 
                                      '4+', '4', '4-', '5+', '5', '5-', 'NG'])
        
        if np.random.random() < 0.15:
            rating_trend = np.linspace(0, np.random.uniform(2, 4), n_days)
            deposits = deposits * (1 - rating_trend/10)
            loan_util = loan_util * (1 + rating_trend/10)
        
        for i, date in enumerate(dates):
            data.append({
                'date': date,
                'client_id': client,
                'loan_utilization': loan_util[i],
                'deposits': deposits[i],
                'risk_rating': base_rating,
                'sector': np.random.choice(['Retail', 'Manufacturing', 'Technology', 'Healthcare']),
                'revenue': np.random.uniform(1e7, 1e8),
                'ebitda_margin': np.random.uniform(0.05, 0.25),
                'debt_service_ratio': np.random.uniform(1.1, 2.5),
                'working_capital_ratio': np.random.uniform(0.8, 2.0)
            })
    
    return pd.DataFrame(data)

def identify_advanced_patterns(df, lookback_periods=[90, 180]):
    """
    Identify complex patterns in client behavior over different time periods
    """
    pattern_results = []
    
    for client in df['client_id'].unique():
        client_data = df[df['client_id'] == client].sort_values('date')
        
        for period in lookback_periods:
            recent_data = client_data.iloc[-period:]
            
            # Calculate trends
            loan_trend = np.polyfit(range(len(recent_data)), recent_data['loan_utilization'], 1)[0]
            deposit_trend = np.polyfit(range(len(recent_data)), recent_data['deposits'], 1)[0]
            
            # Pattern 1: Loan up, deposit down
            pattern1 = loan_trend > 0.001 and deposit_trend < -0.001
            
            # Pattern 2: Loan steady (small change), deposit down
            pattern2 = abs(loan_trend) < 0.001 and deposit_trend < -0.001
            
            # Pattern 3: Loan down, deposit down faster
            pattern3 = loan_trend < -0.001 and deposit_trend < loan_trend
            
            # Pattern 4: High volatility in deposits
            deposit_volatility = recent_data['deposits'].std() / recent_data['deposits'].mean()
            pattern4 = deposit_volatility > 0.15
            
            # Pattern 5: Declining deposit to loan ratio
            ratio_trend = np.polyfit(range(len(recent_data)), 
                                   recent_data['deposits'] / (recent_data['loan_utilization'] * 1e6), 1)[0]
            pattern5 = ratio_trend < -0.01
            
            pattern_results.append({
                'client_id': client,
                'period': period,
                'loan_up_deposit_down': pattern1,
                'loan_steady_deposit_down': pattern2,
                'deposit_falling_faster': pattern3,
                'high_deposit_volatility': pattern4,
                'declining_deposit_loan_ratio': pattern5,
                'risk_score': sum([pattern1, pattern2, pattern3, pattern4, pattern5])
            })
    
    return pd.DataFrame(pattern_results)

def categorize_levels(series):
    """Categorize values into High, Medium, Low based on percentiles"""
    thresholds = series.quantile([0.33, 0.67])
    categories = pd.cut(series, 
                       bins=[-np.inf, thresholds[0.33], thresholds[0.67], np.inf],
                       labels=['Low', 'Medium', 'High'])
    return categories

def create_client_segments(df):
    """Create client segments based on loan utilization and deposit levels"""
    # Calculate average metrics for recent period (last 90 days)
    recent_metrics = df.groupby('client_id').agg({
        'loan_utilization': 'mean',
        'deposits': 'mean'
    })
    
    # Categorize into High, Medium, Low
    recent_metrics['loan_category'] = categorize_levels(recent_metrics['loan_utilization'])
    recent_metrics['deposit_category'] = categorize_levels(recent_metrics['deposits'])
    
    # Create segments
    recent_metrics['segment'] = recent_metrics['loan_category'] + '_loan_' + \
                               recent_metrics['deposit_category'] + '_deposit'
    
    return recent_metrics

def analyze_client_population(df):
    """Analyze client population for waterfall chart"""
    total_clients = df['client_id'].nunique()
    
    # Check for minimum history
    clients_with_history = df.groupby('client_id')['date'].agg(['min', 'max'])
    clients_with_2years = clients_with_history[
        (clients_with_history['max'] - clients_with_history['min']).dt.days >= 365*2
    ].index
    
    # Get clients with both loans and deposits
    clients_with_both = df[
        (df['loan_utilization'] > 0) & (df['deposits'] > 0)
    ]['client_id'].unique()
    
    # Combine criteria
    final_clients = set(clients_with_2years) & set(clients_with_both)
    
    return {
        'total_clients': total_clients,
        'clients_with_2years': len(clients_with_2years),
        'clients_with_both': len(clients_with_both),
        'final_clients': len(final_clients)
    }

def create_waterfall_chart(population_metrics):
    """Create waterfall chart showing client population breakdown"""
    fig = go.Figure(go.Waterfall(
        name="Client Population",
        orientation="v",
        measure=["absolute", "relative", "relative", "total"],
        x=["Total Clients", "With 2yr History", "With Loans & Deposits", "Final Population"],
        y=[population_metrics['total_clients'],
           -(population_metrics['total_clients'] - population_metrics['clients_with_2years']),
           -(population_metrics['clients_with_2years'] - population_metrics['final_clients']),
           population_metrics['final_clients']],
        connector={"line": {"color": "rgb(63, 63, 63)"}},
    ))
    
    fig.update_layout(
        title="Client Population Waterfall Analysis",
        showlegend=False
    )
    
    return fig

def create_segment_analysis_plots(df, segments):
    """Create visualization for segment analysis"""
    # Merge segments with original data
    df_with_segments = df.merge(
        segments[['segment']], 
        left_on='client_id', 
        right_index=True
    )
    
    # Create subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=("Segment Distribution", "Average Loan Utilization by Segment",
                       "Average Deposits by Segment", "Risk Score Distribution by Segment")
    )
    
    # Segment distribution
    segment_counts = df_with_segments['segment'].value_counts()
    fig.add_trace(
        go.Bar(x=segment_counts.index, y=segment_counts.values, name="Count"),
        row=1, col=1
    )
    
    # Average metrics by segment
    segment_metrics = df_with_segments.groupby('segment').agg({
        'loan_utilization': 'mean',
        'deposits': 'mean',
        'risk_rating': lambda x: (pd.to_numeric(x.str[0], errors='coerce')).mean()
    })
    
    fig.add_trace(
        go.Bar(x=segment_metrics.index, y=segment_metrics['loan_utilization'], name="Loan Util"),
        row=1, col=2
    )
    
    fig.add_trace(
        go.Bar(x=segment_metrics.index, y=segment_metrics['deposits'], name="Deposits"),
        row=2, col=1
    )
    
    fig.add_trace(
        go.Bar(x=segment_metrics.index, y=segment_metrics['risk_rating'], name="Risk Rating"),
        row=2, col=2
    )
    
    fig.update_layout(height=800, title_text="Segment Analysis Dashboard")
    return fig

def main():
    # Generate synthetic data
    df = generate_synthetic_data()
    
    # Identify advanced patterns
    patterns_df = identify_advanced_patterns(df)
    
    # Create client segments
    segments_df = create_client_segments(df)
    
    # Analyze population
    population_metrics = analyze_client_population(df)
    
    # Create visualizations
    waterfall_chart = create_waterfall_chart(population_metrics)
    segment_analysis = create_segment_analysis_plots(df, segments_df)
    
    # Print summary
    print("\nAnalysis Summary:")
    print(f"Total clients analyzed: {population_metrics['total_clients']}")
    print(f"Clients in final analysis: {population_metrics['final_clients']}")
    print("\nPattern Detection Results:")
    print(f"High risk clients (3+ patterns): {len(patterns_df[patterns_df['risk_score'] >= 3])}")
    print("\nSegment Distribution:")
    print(segments_df['segment'].value_counts())
    
    # Save visualizations
    waterfall_chart.write_html("waterfall_chart.html")
    segment_analysis.write_html("segment_analysis.html")

if __name__ == "__main__":
    main()
