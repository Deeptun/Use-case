import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import TimeSeriesSplit
from statsmodels.tsa.seasonal import STL
import xgboost as xgb
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# Configuration
np.random.seed(42)
N_CLIENTS = 2
START_DATE = datetime(2020, 1, 1)
END_DATE = datetime(2023, 12, 31)
DAYS = (END_DATE - START_DATE).days

# Enhanced Data Simulation with More Realistic Patterns
def simulate_client_data():
    dates = pd.date_range(START_DATE, END_DATE, freq='D')  # Ensure daily frequency
    clients = [f'C{str(i).zfill(3)}' for i in range(N_CLIENTS)]
    
    data = []
    for client in clients:
        # Base patterns with multiple regimes
        base_loan = np.random.normal(0.5, 0.1)
        base_deposit = np.random.normal(0.6, 0.15)
        
        # Multiple trend regimes
        n_regimes = np.random.randint(2, 5)
        regime_dates = sorted(np.random.choice(dates, n_regimes, replace=False))
        
        loan_trend = np.zeros(len(dates))
        deposit_trend = np.zeros(len(dates))
        
        current_regime = 0
        for i in range(len(dates)):
            if current_regime < len(regime_dates) and dates[i] > regime_dates[current_regime]:
                current_regime += 1
            loan_trend[i] = base_loan + np.random.choice([-0.2, 0, 0.2]) * current_regime
            deposit_trend[i] = base_deposit + np.random.choice([-0.3, 0, 0.1]) * current_regime
        
        # Seasonality with varying amplitude
        season_amp = np.random.uniform(0.05, 0.2)
        season = season_amp * np.sin(2 * np.pi * np.arange(len(dates)) / 365)
        
        # Shock events with different magnitudes
        shocks = np.zeros(len(dates))
        shock_days = np.random.choice(len(dates), size=np.random.randint(3, 7), replace=False)
        shocks[shock_days] = np.random.uniform(-0.4, 0.4, len(shock_days))
        
        # Combine components
        loan_util = loan_trend + season + shocks + np.random.normal(0, 0.05, len(dates))
        op_deposit = deposit_trend + -season * 0.5 + shocks * 0.7 + np.random.normal(0, 0.07, len(dates))
        
        # Create risk scenarios with multiple patterns
        if np.random.rand() < 0.25:  # Risky clients
            pattern_type = np.random.choice(['divergence', 'crossing', 'accelerated_decline'])
            cross_point = np.random.randint(300, len(dates) - 180)
            
            if pattern_type == 'divergence':
                loan_util[cross_point:] = np.minimum(loan_util[cross_point:] + 0.4, 1)
                op_deposit[cross_point:] = np.maximum(op_deposit[cross_point:] - 0.3, 0)
            elif pattern_type == 'crossing':
                loan_util[cross_point:] = np.linspace(loan_util[cross_point], 1, len(dates) - cross_point)
                op_deposit[cross_point:] = np.linspace(op_deposit[cross_point], 0, len(dates) - cross_point)
            else:  # accelerated decline
                loan_util[cross_point:] = loan_util[cross_point] - np.exp(np.linspace(0, 2, len(dates) - cross_point)) / 10
                op_deposit[cross_point:] = op_deposit[cross_point] - np.exp(np.linspace(0, 3, len(dates) - cross_point)) / 8
        
        # Create DataFrame
        df = pd.DataFrame({
            'date': dates,
            'client_id': client,
            'loan_utilization': np.clip(loan_util, 0, 1),
            'operating_deposits': np.clip(op_deposit, 0, 1),
            'credit_rating': np.random.choice([f"{i}{s}" for i in range(1, 11) for s in ['-', '', '+']], len(dates)),
            'default_flag': np.zeros(len(dates))
        })
        
        # Add defaults with multiple warning signs
        if np.random.rand() < 0.07:
            default_day = np.random.randint(len(dates) - 180, len(dates))
            df.loc[default_day:, 'default_flag'] = 1
            # Create pre-default patterns
            warning_start = default_day - np.random.randint(90, 180)
            df.loc[warning_start:default_day, 'loan_utilization'] = np.minimum(
                df.loc[warning_start:default_day, 'loan_utilization'] + 0.4, 1)
            df.loc[warning_start:default_day, 'operating_deposits'] = np.maximum(
                df.loc[warning_start:default_day, 'operating_deposits'] - 0.3, 0)
        
        data.append(df)
    
    return pd.concat(data).sort_values(['client_id', 'date'])

# Enhanced KPI Calculation with Pattern Detection
def calculate_kpis(df):
    df = df.sort_values(['client_id', 'date'])
    
    # 1. Rolling Trend Analysis
    def calculate_trend(series, window):
        def get_slope(x):
            x = x[~np.isnan(x)]
            if len(x) < 2: return np.nan
            return LinearRegression().fit(np.arange(len(x)).reshape(-1,1), x).coef_[0]
        return series.rolling(window, min_periods=int(window*0.5)).apply(get_slope)
    
    windows = [90, 180]  # 3 and 6 months
    for window in windows:
        df[f'loan_trend_{window}'] = df.groupby('client_id')['loan_utilization'].transform(
            lambda x: calculate_trend(x, window))
        df[f'deposit_trend_{window}'] = df.groupby('client_id')['operating_deposits'].transform(
            lambda x: calculate_trend(x, window))
    
    # 2. Pattern Flags
    def detect_patterns(group):
        # Recent 6 months trends
        loan_trend = group[f'loan_trend_180'].iloc[-1]
        deposit_trend = group[f'deposit_trend_180'].iloc[-1]
        
        # Pattern 1: Loan ↗ & Deposit ↘
        group['pattern1'] = (loan_trend > 0.01) & (deposit_trend < -0.01)
        
        # Pattern 2: Loan ↔ & Deposit ↘ 
        group['pattern2'] = (abs(loan_trend) < 0.005) & (deposit_trend < -0.01)
        
        # Pattern 3: Loan ↘ & Deposit ↘↘
        group['pattern3'] = (loan_trend < -0.005) & (deposit_trend < loan_trend*1.5)
        
        # Pattern 4: Volatility increase
        loan_vol = group['loan_utilization'].rolling(90).std().iloc[-1]
        dep_vol = group['operating_deposits'].rolling(90).std().iloc[-1]
        group['pattern4'] = (loan_vol > 0.15) | (dep_vol > 0.2)
        
        return group
    
    df = df.groupby('client_id').apply(detect_patterns)
    
    # 3. Liquidity Score
    df['liquidity_score'] = (df['operating_deposits'] * 0.6 + 
                            (1 - df['loan_utilization']) * 0.4)
    
    return df

# Enhanced Dynamic Clustering with Behavior Patterns
def dynamic_clustering(df, n_clusters=5):
    # Feature Engineering
    features = df.groupby('client_id').agg({
        'loan_utilization': ['mean', 'std', 'max', lambda x: x.ewm(span=90).mean().iloc[-1]],
        'operating_deposits': ['mean', 'std', 'min', lambda x: x.ewm(span=90).mean().iloc[-1]],
        'liquidity_score': ['mean', 'std'],
        'pattern1': 'sum',
        'pattern2': 'sum',
        'pattern3': 'sum',
        'pattern4': 'sum'
    })
    # Flatten multi-level column names and ensure they are strings
    features.columns = ['_'.join(col).strip() for col in features.columns.values]
    
    # Temporal Features using STL Decomposition
    def calculate_stl_features(series):
        try:
            res = STL(series, period=365).fit()
            return pd.Series({
                'trend_strength': res.trend.max() - res.trend.min(),
                'seasonality_strength': res.seasonal.max() - res.seasonal.min(),
                'residual_strength': res.resid.max() - res.resid.min()
            })
        except:
            return pd.Series({
                'trend_strength': np.nan,
                'seasonality_strength': np.nan,
                'residual_strength': np.nan
            })
    
    # Apply STL decomposition for each client
    stl_results = df.groupby('client_id').apply(
        lambda x: pd.concat([
            calculate_stl_features(x['loan_utilization']).add_prefix('loan_'),
            calculate_stl_features(x['operating_deposits']).add_prefix('deposit_')
        ], axis=0)
    )
    stl_results = stl_results.unstack().reset_index()
    
    # Merge STL features with other features
    features = features.merge(stl_results, on='client_id', how='left')
    
    # Ensure all column names are strings
    # features.columns = features.columns.astype(str)
    
    # Drop non-numeric columns (e.g., 'client_id') before clustering
    numeric_features = features.drop(columns=['client_id']).fillna(0)
    
    # Clustering
    scaler = StandardScaler()
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(scaler.fit_transform(numeric_features))
    
    # Add cluster labels to the original DataFrame
    df = df.merge(pd.Series(clusters, index=features.index, name='cluster'), 
                 left_on='client_id', right_index=True)
    return df

# Waterfall Chart and Cohort Analysis
def create_cohort_analysis(df):
    # Filter clients with 2 years of data
    client_counts = df.groupby('client_id')['date'].nunique()
    valid_clients = client_counts[client_counts >= 730].index
    filtered_df = df[df['client_id'].isin(valid_clients)]
    
    # Define thresholds using quantiles
    loan_thresholds = filtered_df['loan_utilization'].quantile([0.33, 0.66]).values
    deposit_thresholds = filtered_df['operating_deposits'].quantile([0.33, 0.66]).values
    
    # Categorization function
    def categorize(value, thresholds, type_):
        if type_ == 'loan':
            if value <= thresholds[0]: return 'Low'
            elif value <= thresholds[1]: return 'Medium'
            else: return 'High'
        else:
            if value <= thresholds[0]: return 'Low'
            elif value <= thresholds[1]: return 'Medium'
            else: return 'High'
    
    # Calculate recent averages (last 6 months)
    recent_data = filtered_df.groupby('client_id').apply(lambda x: x.tail(180))
    client_stats = recent_data.groupby('client_id').agg({
        'loan_utilization': 'mean',
        'operating_deposits': 'mean'
    })
    
    # Apply categorization
    client_stats['loan_category'] = client_stats['loan_utilization'].apply(
        lambda x: categorize(x, loan_thresholds, 'loan'))
    client_stats['deposit_category'] = client_stats['operating_deposits'].apply(
        lambda x: categorize(x, deposit_thresholds, 'deposit'))
    
    # Create cohort matrix
    cohort_matrix = pd.crosstab(
        client_stats['loan_category'],
        client_stats['deposit_category'],
        margins=True
    )
    
    # Waterfall Chart
    waterfall_data = {
        'Initial Population': len(client_counts),
        'With 2+ Years History': len(valid_clients),
        **{f"{l}-{d}": cohort_matrix.loc[l,d] 
           for l in ['High','Medium','Low'] 
           for d in ['High','Medium','Low']}
    }
    
    fig = go.Figure(go.Waterfall(
        name="20",
        orientation="v",
        measure=["absolute"] + ["relative"]*(len(waterfall_data)-2) + ["total"],
        x=list(waterfall_data.keys()),
        textposition="auto",
        text=[f"{v:,}" for v in waterfall_data.values()],
        y=list(waterfall_data.values()),
        connector={"line":{"color":"rgb(63, 63, 63)"}},
    ))
    
    fig.update_layout(
        title="Client Cohort Analysis",
        showlegend=False
    )
    
    # Heatmap
    heatmap_fig = px.imshow(cohort_matrix.iloc[:-1, :-1],
                           labels=dict(x="Deposit Category", y="Loan Category", color="Count"),
                           x=['High','Medium','Low'],
                           y=['High','Medium','Low'],
                           text_auto=True)
    heatmap_fig.update_layout(title="Loan vs Deposit Category Distribution")
    
    return fig, heatmap_fig, cohort_matrix

# Enhanced Visualization with Patterns
def create_pattern_visualization(df):
    # Select sample clients with different patterns
    sample_clients = []
    patterns = ['pattern1', 'pattern2', 'pattern3', 'pattern4']
    for pattern in patterns:
        clients = df[df[pattern]].groupby('client_id').size().nlargest(2).index
        sample_clients.extend(clients)
    
    # Create subplots
    fig = make_subplots(rows=len(sample_clients), cols=1, 
                       subplot_titles=[f"Client {c}" for c in sample_clients])
    
    for i, client in enumerate(sample_clients):
        client_data = df[df['client_id'] == client]
        
        # Loan Utilization
        fig.add_trace(go.Scatter(
            x=client_data['date'],
            y=client_data['loan_utilization'],
            name='Loan Utilization',
            line=dict(color='red'),
        ), row=i+1, col=1)
        
        # Operating Deposits
        fig.add_trace(go.Scatter(
            x=client_data['date'],
            y=client_data['operating_deposits'],
            name='Operating Deposits',
            line=dict(color='blue'),
        ), row=i+1, col=1)
        
        # Add pattern annotations
        patterns_present = client_data.iloc[-1][['pattern1','pattern2','pattern3','pattern4']]
        pattern_text = "Patterns: " + ", ".join([f"P{i+1}" for i, p in enumerate(patterns_present) if p])
        fig.add_annotation(
            xref=f"x{i+1}",
            yref=f"y{i+1}",
            x=0.05,
            y=0.95,
            text=pattern_text,
            showarrow=False,
            font=dict(size=10)
        )
    
    fig.update_layout(height=300*len(sample_clients), title_text="Client Patterns Analysis")
    return fig

# Main Execution
if __name__ == "__main__":
    # Generate and process data
    df = simulate_client_data()
    df = calculate_kpis(df)
    df = dynamic_clustering(df)
    
    # Cohort analysis and visualization
    waterfall_fig, heatmap_fig, cohort_matrix = create_cohort_analysis(df)
    pattern_fig = create_pattern_visualization(df)
    
    # Show results
    waterfall_fig.show()
    heatmap_fig.show()
    pattern_fig.show()
    
    # Print cohort insights
    print("\nCohort Analysis Matrix:")
    print(cohort_matrix)
    
    # Risk client identification
    risk_clients = df.groupby('client_id').apply(lambda x: x[['pattern1','pattern2','pattern3','pattern4']].any().any())
    print(f"\nClients with risk patterns: {risk_clients.sum()} out of {len(risk_clients)}")