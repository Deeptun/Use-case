import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from statsmodels.tsa.seasonal import STL
from sklearn.pipeline import make_pipeline

# Set random seed for reproducibility
np.random.seed(42)

# 1. Enhanced Data Generation
def generate_synthetic_data(num_clients=100, years=4):
    dates = pd.date_range(end=pd.Timestamp.today(), periods=365*years, freq='D')
    clients = [f'C{i:03d}' for i in range(num_clients)]
    
    data = []
    for client in clients:
        # Generate base patterns with realistic financial behaviors
        base_deposit = np.random.lognormal(mean=10, sigma=0.5) + 1000
        base_loan = base_deposit * np.random.uniform(0.2, 0.8)
        
        # Generate trend components with regime changes
        trend = np.concatenate([
            np.linspace(0, np.random.normal(0, 0.01), 300),
            np.linspace(0, np.random.normal(0, 0.03), 365*years-300)
        ])
        
        # Generate seasonality with multiple cycles
        seasonality = (np.sin(np.linspace(0, 4*np.pi*years, len(dates))) * 0.2 + \
                     (np.sin(np.linspace(0, 12*np.pi*years, len(dates))) * 0.1
        
        # Generate deposits with structural breaks
        deposits = (base_deposit + 
                   base_deposit * trend + 
                   base_deposit * seasonality + 
                   np.random.normal(0, 0.05*base_deposit, len(dates)))
        
        # Generate loan utilization patterns with correlation regimes
        loan_corr = np.random.choice([0.3, 0.8], p=[0.7, 0.3])
        loans = (base_loan + 
                loan_corr * deposits * 0.5 +
                (1 - loan_corr) * (base_loan * trend * np.random.uniform(-1, 1)) + 
                np.random.normal(0, 0.05*base_loan, len(dates)))
        
        # Ensure non-zero/non-negative values
        deposits = np.abs(deposits) + 1
        loans = np.clip(np.abs(loans), 1, None)
        
        df = pd.DataFrame({
            'Date': dates,
            'Client': client,
            'Deposits': deposits,
            'Loans': loans
        })
        data.append(df)
    
    return pd.concat(data).reset_index(drop=True)

# 2. Advanced Feature Engineering
def compute_risk_features(df, window=90):
    def process_client(x):
        x = x.sort_values('Date')
        # Rolling features
        x['Deposit_MA'] = x.Deposits.rolling(window).mean()
        x['Loan_MA'] = x.Loans.rolling(window).mean()
        x['Loan_Utilization'] = x.Loans / x.Deposits
        x['Rolling_Corr'] = x.Loans.rolling(window).corr(x.Deposits)
        x['Deposit_Change'] = x.Deposits.pct_change(window)
        x['Loan_Change'] = x.Loans.pct_change(window)
        x['Volatility'] = x.Deposits.rolling(window).std()
        
        # STL Decomposition
        stl = STL(x.Deposits, period=365, robust=True)
        res = stl.fit()
        x['Deposit_Trend'] = res.trend
        x['Deposit_Seasonal'] = res.seasonal
        
        return x.dropna()
    
    features = df.groupby('Client', group_keys=False).apply(process_client)
    
    # Risk flags
    features['High_Utilization'] = (features.Loan_Utilization > 0.8).astype(int)
    features['Deposit_Drop'] = (features.Deposit_Change < -0.2).astype(int)
    features['Loan_Surge'] = (features.Loan_Change > 0.3).astype(int)
    
    return features

# 3. Anomaly Detection
def detect_anomalies(features):
    model = IsolationForest(contamination=0.05, random_state=42)
    
    anomalies = features.groupby('Client', group_keys=False).apply(
        lambda x: x.assign(
            Deposit_Anomaly = model.fit_predict(x[['Deposits']]),
            Loan_Anomaly = model.fit_predict(x[['Loans']])
        )
    )
    anomalies['Deposit_Anomaly'] = (anomalies['Deposit_Anomaly'] == -1).astype(int)
    anomalies['Loan_Anomaly'] = (anomalies['Loan_Anomaly'] == -1).astype(int)
    return anomalies

# 4. Enhanced Clustering with Correlation Cluster
def dynamic_clustering(features, n_clusters=3, corr_threshold=0.7):
    monthly_data = features.set_index('Date').groupby(['Client', pd.Grouper(freq='M')]).last().reset_index()
    
    clusters = []
    for period, group in monthly_data.groupby(pd.Grouper(key='Date', freq='M')):
        if len(group) == 0:
            continue
        
        # Identify strong correlation clients
        corr_clients = group[group.Rolling_Corr >= corr_threshold].copy()
        corr_clients['Cluster'] = 3  # Special cluster
        
        # Cluster remaining clients
        remaining = group[~group.Client.isin(corr_clients.Client)]
        if len(remaining) > 0:
            scaler = StandardScaler()
            scaled_data = scaler.fit_transform(remaining[['Loan_Utilization', 'Deposit_Change']])
            kmeans = KMeans(n_clusters=n_clusters, n_init=10)
            remaining['Cluster'] = kmeans.fit_predict(scaled_data)
        
        clusters.append(pd.concat([corr_clients, remaining], ignore_index=True))
    
    return pd.concat(clusters).sort_values(['Client', 'Date'])

# 5. Early Warning System
def detect_early_warnings(clustered_data):
    warnings = clustered_data.groupby('Client', group_keys=False).apply(
        lambda x: x.assign(
            Warning_1 = (x.Loan_Utilization.rolling(90).mean().diff() > 0.1) &
                       (x.Deposit_MA.rolling(90).mean().diff() < -0.05),
            Warning_2 = (x.Loan_Change.rolling(60).mean().abs() < 0.05) &
                       (x.Deposit_Change.rolling(60).mean() < -0.03),
            Warning_3 = (x.Loans.diff(90) < -0.1) & 
                       (x.Deposits.diff(90) < -0.15),
            Warning_4 = (x.Rolling_Corr.rolling(60).mean() > 0.7) &
                       (x.Deposit_Change < -0.1)
        )
    )
    warnings['Total_Warnings'] = warnings.filter(like='Warning').sum(axis=1)
    return warnings

# 6. Credit Risk Prediction
def build_risk_model(features, horizon=90):
    # Create synthetic target variable
    features['Target'] = features.groupby('Client')['High_Utilization'].shift(-horizon).fillna(0)
    
    # Prepare data
    train = features.dropna()
    X = train[['Loan_Utilization', 'Deposit_Change', 'Rolling_Corr', 'Volatility']]
    y = train['Target']
    
    # Build pipeline
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(class_weight='balanced')
    )
    model.fit(X, y)
    
    # Add predictions
    features['Downgrade_Probability'] = model.predict_proba(
        features[['Loan_Utilization', 'Deposit_Change', 'Rolling_Corr', 'Volatility']]
    )[:,1]
    return features

# 7. Visualization Functions
def plot_enhanced_waterfall(clustered_data):
    plt.figure(figsize=(14, 7))
    metrics = [
        ('Total Clients', len(clustered_data.Client.unique())),
        ('Data Quality', clustered_data.groupby('Client').size().mean()),
        ('High Utilization', clustered_data[clustered_data.High_Utilization == 1].Client.nunique()),
        ('Deposit Drop', clustered_data[clustered_data.Deposit_Drop == 1].Client.nunique()),
        ('Strong Correlation', clustered_data[clustered_data.Cluster == 3].Client.nunique()),
        ('At Risk', clustered_data[clustered_data.Total_Warnings > 1].Client.nunique())
    ]
    values = [m[1] for m in metrics]
    labels = [m[0] for m in metrics]
    
    plt.bar(range(len(metrics)), values, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b'])
    plt.xticks(range(len(metrics)), labels, rotation=45)
    plt.title('Enhanced Client Risk Segmentation')
    plt.ylabel('Number of Clients')
    plt.tight_layout()
    plt.show()

def plot_correlation_cluster_trends(clustered_data):
    plt.figure(figsize=(12, 6))
    corr_cluster = clustered_data[clustered_data.Cluster == 3]
    
    for client in corr_cluster.Client.unique()[:5]:
        client_data = corr_cluster[corr_cluster.Client == client]
        plt.plot(client_data.Date, client_data.Rolling_Corr, label=client)
    
    plt.axhline(0.7, color='r', linestyle='--', label='Correlation Threshold')
    plt.title('Strong Correlation Cluster Members')
    plt.ylabel('Rolling Correlation')
    plt.legend()
    plt.show()

# Main Analysis Pipeline
def full_analysis():
    # Generate data
    raw_data = generate_synthetic_data()
    
    # Feature engineering
    features = compute_risk_features(raw_data)
    
    # Anomaly detection
    features = detect_anomalies(features)
    
    # Clustering
    clustered = dynamic_clustering(features)
    
    # Early warnings
    clustered = detect_early_warnings(clustered)
    
    # Risk modeling
    final_data = build_risk_model(clustered)
    
    # Visualization
    plot_enhanced_waterfall(final_data)
    plot_correlation_cluster_trends(final_data)
    
    # Cluster analysis
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=final_data, x='Cluster', y='Downgrade_Probability')
    plt.title('Risk Downgrade Probability by Cluster')
    plt.show()
    
    return final_data

# Execute analysis
if __name__ == "__main__":
    final_data = full_analysis()
    
    # Display results
    print("\nStrong Correlation Cluster Statistics:")
    strong_corr = final_data[final_data.Cluster == 3]
    print(f"Number of clients: {strong_corr.Client.nunique()}")
    print(f"Average correlation: {strong_corr.Rolling_Corr.mean():.2f}")
    print(f"Downgrade probability: {strong_corr.Downgrade_Probability.mean():.2f}")
    
    print("\nTop At-Risk Clients:")
    at_risk = final_data[final_data.Total_Warnings >= 2].groupby('Client').agg(
        Warnings=('Total_Warnings', 'max'),
        Downgrade_Prob=('Downgrade_Probability', 'max')
    ).sort_values(['Warnings', 'Downgrade_Prob'], ascending=False)
    print(at_risk.head(10))
