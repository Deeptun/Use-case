import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from datetime import datetime, timedelta
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Set styles
plt.style.use('ggplot')
sns.set_palette("Set2")

# Configuration dictionary for easy parameter tuning
CONFIG = {
    'data': {
        'min_nonzero_pct': 0.8
    },
    'risk': {
        'trend_windows': [30, 90, 180],  # days
        'change_thresholds': {
            'sharp': 0.2,
            'moderate': 0.1,
            'gradual': 0.05
        },
        'persona_patterns': {
            'cautious_borrower': 'Low utilization, stable deposits',
            'aggressive_expansion': 'Rising utilization, volatile deposits',
            'distressed_client': 'High utilization, declining deposits',
            'seasonal_operator': 'Cyclical utilization and deposits',
            'deteriorating_health': 'Rising utilization, declining deposits',
            'cash_constrained': 'Stable utilization, rapidly declining deposits',
            'credit_dependent': 'High utilization, low deposit ratio'
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
    2. Removing NaN and infinite values
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
    
    # Replace any remaining zeros with NaN for calculations
    # (keeping zeros intact in the original data)
    df_calc = df_clean.copy()
    df_calc.loc[df_calc['deposit_balance'] == 0, 'deposit_balance'] = np.nan
    df_calc.loc[df_calc['used_loan'] == 0, 'used_loan'] = np.nan
    df_calc.loc[df_calc['unused_loan'] == 0, 'unused_loan'] = np.nan
    
    # Remove infinite values
    df_calc.replace([np.inf, -np.inf], np.nan, inplace=True)
    
    print(f"Cleaned data shape: {df_clean.shape}")
    print(f"Removed {len(df['company_id'].unique()) - len(valid_companies)} companies")
    
    return df_clean, df_calc

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
    df['loan_utilization'].fillna(0, inplace=True)
    df.loc[df['total_loan'] == 0, 'loan_utilization'] = 0
    
    # Calculate deposit to loan ratio
    df['deposit_loan_ratio'] = df['deposit_balance'] / df['used_loan']
    df['deposit_loan_ratio'].replace([np.inf, -np.inf], np.nan, inplace=True)
    
    # Calculate rolling metrics for each company
    for company in tqdm(df['company_id'].unique(), desc="Calculating rolling metrics"):
        company_data = df[df['company_id'] == company].sort_values('date')
        
        # Skip if too few data points
        if len(company_data) < 30:
            continue
            
        # Calculate rolling averages for different windows
        for window in [7, 30, 90]:
            df.loc[company_data.index, f'util_ma_{window}d'] = company_data['loan_utilization'].rolling(window, min_periods=3).mean()
            df.loc[company_data.index, f'deposit_ma_{window}d'] = company_data['deposit_balance'].rolling(window, min_periods=3).mean()
            
        # Calculate rates of change
        for window in [30, 90]:
            df.loc[company_data.index, f'util_change_{window}d'] = company_data['loan_utilization'].pct_change(periods=window)
            df.loc[company_data.index, f'deposit_change_{window}d'] = company_data['deposit_balance'].pct_change(periods=window)
            
        # Calculate volatility measures
        df.loc[company_data.index, 'util_volatility_30d'] = company_data['loan_utilization'].rolling(30, min_periods=10).std()
        df.loc[company_data.index, 'deposit_volatility_30d'] = company_data['deposit_balance'].pct_change().rolling(30, min_periods=10).std()
    
    return df

def detect_risk_patterns_efficient(df):
    """
    Efficient implementation of risk pattern detection based on rolling metrics
    and predefined risk rules. Assign each company to a persona based on their pattern.
    """
    risk_records = []
    persona_assignments = []
    
    # Time windows to analyze
    windows = CONFIG['risk']['trend_windows']
    
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
            
            # Extract current metrics
            current_util = current_row['loan_utilization']
            current_deposit = current_row['deposit_balance']
            
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
                    risk_descriptions.append(f"[{severity.upper()}] 90d: Rising utilization (+{util_change_90d:.1%}) with declining deposits ({deposit_change_90d:.1%})")
                    risk_levels.append(severity)
                    
                    if persona_confidence < 0.8:
                        persona = "deteriorating_health"
                        persona_confidence = 0.8
            
            # ---- RISK PATTERN 2: High utilization with low deposit ratio ----
            if current_util > 0.8 and current_row.get('deposit_loan_ratio', float('inf')) < 1.0:
                severity = "high" if current_util > 0.9 else "medium"
                risk_flags.append('credit_dependent')
                risk_descriptions.append(f"[{severity.upper()}] Current: High loan utilization ({current_util:.1%}) with low deposit coverage")
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
                    risk_descriptions.append(f"[{severity.upper()}] 30d: Rapid deposit decline ({deposit_change_30d:.1%}) with stable utilization")
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
                            risk_descriptions.append(f"[MEDIUM] Significant increase in volatility for both metrics")
                            risk_levels.append("medium")
                            
                            if persona_confidence < 0.6:
                                persona = "aggressive_expansion"
                                persona_confidence = 0.6
            
            # ---- RISK PATTERN 5: Seasonal pattern detection ----
            if len(company_data) > 365:  # Need at least a year of data
                # Check for seasonal patterns using autocorrelation
                try:
                    util_series = company_data['loan_utilization'].iloc[max(0, i-365):i+1]
                    deposit_series = company_data['deposit_balance'].iloc[max(0, i-365):i+1]
                    
                    # Resample to monthly to check for seasonality
                    monthly_util = util_series.resample('M', on='date').mean()
                    monthly_deposit = deposit_series.resample('M', on='date').mean()
                    
                    # Calculate autocorrelation
                    if len(monthly_util) > 6 and len(monthly_deposit) > 6:
                        util_autocorr = pd.Series(monthly_util).autocorr(lag=3)
                        deposit_autocorr = pd.Series(monthly_deposit).autocorr(lag=3)
                        
                        if abs(util_autocorr) > 0.5 or abs(deposit_autocorr) > 0.5:
                            risk_flags.append('seasonal_pattern')
                            risk_descriptions.append(f"[LOW] Detected seasonal patterns in financial metrics")
                            risk_levels.append("low")
                            
                            if persona_confidence < 0.65:
                                persona = "seasonal_operator"
                                persona_confidence = 0.65
                except:
                    pass  # Skip if autocorrelation fails
            
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
                    if current_util < 0.3:
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
                    'current_util': current_util,
                    'current_deposit': current_deposit
                })
                
                # Record persona assignment for cohort analysis
                persona_assignments.append({
                    'company_id': company,
                    'date': current_date,
                    'persona': persona,
                    'confidence': persona_confidence,
                    'risk_level': overall_risk
                })
    
    # Create risk dataframe
    if risk_records:
        risk_df = pd.DataFrame(risk_records)
        persona_df = pd.DataFrame(persona_assignments)
        return risk_df, persona_df
    else:
        # Return empty dataframes with correct columns
        risk_df = pd.DataFrame(columns=['company_id', 'date', 'risk_flags', 'risk_description', 
                                        'risk_level', 'persona', 'current_util', 'current_deposit'])
        persona_df = pd.DataFrame(columns=['company_id', 'date', 'persona', 'confidence', 'risk_level'])
        return risk_df, persona_df

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

def plot_persona_cohort(cohort_data):
    """
    Plot persona-based cohort analysis showing how personas evolve over time.
    """
    if cohort_data is None or cohort_data.empty:
        print("No cohort data available for plotting.")
        return None
    
    plt.figure(figsize=(14, 8))
    
    # Plot area chart
    ax = cohort_data.plot(kind='area', stacked=True, alpha=0.7, figsize=(14, 8))
    
    # Formatting
    plt.title('Client Personas Over Time', fontsize=16)
    plt.xlabel('Quarter', fontsize=12)
    plt.ylabel('Number of Companies', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend(title='Personas', title_fontsize=12)
    
    # Add annotations for percentages
    for i, quarter in enumerate(cohort_data.index):
        total = cohort_data.loc[quarter].sum()
        if total == 0:
            continue
            
        y_pos = 0
        for persona in cohort_data.columns:
            count = cohort_data.loc[quarter, persona]
            if count > 0:
                percentage = count / total * 100
                if percentage > 5:  # Only show labels for significant percentages
                    plt.annotate(f"{percentage:.0f}%", 
                                xy=(i, y_pos + count/2), 
                                ha='center', 
                                va='center',
                                fontsize=9,
                                fontweight='bold')
            y_pos += count
    
    plt.tight_layout()
    return ax

def plot_persona_transitions(quarterly_persona_df):
    """
    Plot Sankey diagram or other visualization showing transitions between personas over time.
    """
    if quarterly_persona_df is None or quarterly_persona_df.empty:
        print("No transition data available for plotting.")
        return None
    
    # Create a DataFrame to track persona transitions between quarters
    transitions = []
    
    for company_id, company_data in quarterly_persona_df.groupby('company_id'):
        company_data = company_data.sort_values('quarter')
        
        # Skip if only one quarter of data
        if len(company_data) < 2:
            continue
            
        # Track transitions between consecutive quarters
        for i in range(len(company_data) - 1):
            from_persona = company_data.iloc[i]['persona']
            to_persona = company_data.iloc[i+1]['persona']
            from_quarter = company_data.iloc[i]['quarter']
            to_quarter = company_data.iloc[i+1]['quarter']
            
            transitions.append({
                'company_id': company_id,
                'from_quarter': from_quarter,
                'to_quarter': to_quarter,
                'from_persona': from_persona,
                'to_persona': to_persona
            })
    
    transitions_df = pd.DataFrame(transitions)
    
    if transitions_df.empty:
        print("No transitions found between quarters.")
        return None
    
    # Count transitions between personas
    transition_counts = transitions_df.groupby(['from_persona', 'to_persona']).size().reset_index(name='count')
    
    # Create a heatmap of transitions
    pivot_transitions = transition_counts.pivot_table(
        index='from_persona', 
        columns='to_persona', 
        values='count',
        fill_value=0
    )
    
    plt.figure(figsize=(12, 8))
    sns.heatmap(pivot_transitions, annot=True, fmt='d', cmap='YlGnBu')
    plt.title('Persona Transitions Between Quarters', fontsize=16)
    plt.xlabel('To Persona', fontsize=12)
    plt.ylabel('From Persona', fontsize=12)
    plt.tight_layout()
    
    return plt.gcf()

def plot_risk_company(company_id, df, risk_df):
    """
    Create a detailed plot of a company's risk patterns with persona information.
    Shows loan utilization and deposit balance on two axes with risk annotations.
    """
    # Filter data for company
    company_data = df[df['company_id'] == company_id].sort_values('date')
    company_risks = risk_df[risk_df['company_id'] == company_id].sort_values('date')
    
    if company_data.empty:
        print(f"No data found for company {company_id}")
        return None
    
    # Create figure with two y-axes
    fig, ax1 = plt.subplots(figsize=(15, 8))
    
    # First axis - Loan Utilization
    color = 'tab:blue'
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Loan Utilization (%)', color=color)
    ax1.plot(company_data['date'], company_data['loan_utilization'] * 100, color=color, linewidth=2)
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.set_ylim(0, 100)
    
    # Second axis - Deposits
    color = 'tab:red'
    ax2 = ax1.twinx()
    ax2.set_ylabel('Deposit Balance', color=color)
    ax2.plot(company_data['date'], company_data['deposit_balance'], color=color, linewidth=2)
    ax2.tick_params(axis='y', labelcolor=color)
    
    # Format dates on x-axis
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.xticks(rotation=45)
    
    # Add risk markers
    if not company_risks.empty:
        risk_dates = company_risks['date'].tolist()
        risk_descriptions = company_risks['risk_description'].tolist()
        risk_levels = company_risks['risk_level'].tolist()
        personas = company_risks['persona'].tolist()
        
        # Create color map for risk levels
        risk_colors = {'high': 'red', 'medium': 'orange', 'low': 'yellow'}
        
        # Add vertical lines for each risk event
        for i, (date, desc, level, persona) in enumerate(zip(risk_dates, risk_descriptions, risk_levels, personas)):
            # Limit to plotting only a reasonable number of risk markers
            if i < 8:  # Only show first 8 risk events to avoid cluttering
                ax1.axvline(x=date, color=risk_colors[level], linestyle='--', alpha=0.7)
                
                # Add descriptions (shortened if too long)
                short_desc = desc.split('|')[0] if len(desc) > 50 else desc
                y_pos = 90 - (i % 4) * 20  # Stagger text vertically
                
                ax1.annotate(f"{short_desc} [{persona}]", 
                             xy=(date, y_pos), 
                             xytext=(10, 0), 
                             textcoords="offset points", 
                             color=risk_colors[level],
                             rotation=0, 
                             ha='left', 
                             va='center', 
                             fontsize=8,
                             bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8))
    
    # Add title and company details
    persona_text = ""
    if not company_risks.empty:
        # Get most recent persona
        recent_persona = company_risks.iloc[-1]['persona']
        persona_desc = CONFIG['risk']['persona_patterns'].get(recent_persona, "Unknown")
        persona_text = f"\nCurrent Persona: {recent_persona} - {persona_desc}"
    
    plt.title(f"Risk Analysis for {company_id}{persona_text}", fontsize=14)
    
    # Add key metrics in text box
    avg_util = company_data['loan_utilization'].mean() * 100
    avg_deposit = company_data['deposit_balance'].mean()
    risk_count = len(company_risks)
    
    metrics_text = (
        f"Average Utilization: {avg_util:.1f}%\n"
        f"Average Deposit: ${avg_deposit:,.2f}\n"
        f"Risk Events: {risk_count}"
    )
    
    # Add text box with metrics
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax1.text(0.02, 0.05, metrics_text, transform=ax1.transAxes, fontsize=10,
            verticalalignment='bottom', bbox=props)
    
    plt.tight_layout()
    return fig

def plot_persona_distribution(persona_df):
    """
    Plot distribution of personas across the dataset.
    """
    if persona_df is None or persona_df.empty:
        print("No persona data available for distribution analysis.")
        return None
    
    # Count each persona
    persona_counts = persona_df.groupby('persona').size().reset_index(name='count')
    persona_counts = persona_counts.sort_values('count', ascending=False)
    
    if persona_counts.empty:
        return None
    
    # Create plot
    plt.figure(figsize=(12, 6))
    
    # Create bar plot
    ax = sns.barplot(x='persona', y='count', data=persona_counts, palette='viridis')
    
    # Add percentage labels
    total = persona_counts['count'].sum()
    for i, p in enumerate(ax.patches):
        percentage = 100 * p.get_height() / total
        ax.annotate(f"{percentage:.1f}%", 
                   (p.get_x() + p.get_width() / 2., p.get_height()),
                   ha='center', 
                   va='bottom', 
                   fontsize=10,
                   fontweight='bold')
    
    # Add titles and labels
    plt.title('Distribution of Client Personas', fontsize=16)
    plt.xlabel('Persona', fontsize=12)
    plt.ylabel('Number of Risk Events', fontsize=12)
    plt.xticks(rotation=45)
    
    # Add persona descriptions as a table
    persona_desc = pd.DataFrame(list(CONFIG['risk']['persona_patterns'].items()), 
                                columns=['Persona', 'Description'])
    
    # Create a table at the bottom
    plt.table(cellText=persona_desc.values,
              colLabels=persona_desc.columns,
              loc='bottom',
              cellLoc='center',
              bbox=[0, -0.55, 1, 0.3])
    
    plt.subplots_adjust(bottom=0.35)
    plt.tight_layout()
    
    return plt.gcf()

def perform_clustering(df):
    """
    Use unsupervised learning to cluster companies based on behavioral patterns
    """
    # Create company-level features from time series data
    company_features = []
    
    for company_id, company_data in tqdm(df.groupby('company_id'), desc="Creating clustering features"):
        if len(company_data) < 180:  # Need at least 6 months of data
            continue
        
        # Sort by date
        company_data = company_data.sort_values('date')
        
        try:
            # Utilization statistics
            util_mean = company_data['loan_utilization'].mean()
            util_std = company_data['loan_utilization'].std()
            util_trend = np.polyfit(range(len(company_data)), company_data['loan_utilization'].values, 1)[0]
            
            # Deposit statistics 
            deposit_mean = company_data['deposit_balance'].mean()
            deposit_std = company_data['deposit_balance'].std()
            deposit_trend = np.polyfit(range(len(company_data)), company_data['deposit_balance'].fillna(0).values, 1)[0]
            
            # Volatility and correlation
            volatility_metric = company_data['loan_utilization'].diff().abs().mean()
            correlation = company_data['loan_utilization'].corr(company_data['deposit_balance'])
            
            # Ratio statistics
            deposit_loan_ratio = company_data['deposit_loan_ratio'].mean()
            
            # Feature vector
            company_features.append({
                'company_id': company_id,
                'util_mean': util_mean,
                'util_std': util_std,
                'util_trend': util_trend,
                'deposit_mean': deposit_mean,
                'deposit_std': deposit_std,
                'deposit_trend': deposit_trend,
                'volatility': volatility_metric,
                'correlation': correlation if not np.isnan(correlation) else 0,
                'deposit_loan_ratio': deposit_loan_ratio if not np.isnan(deposit_loan_ratio) else 0
            })
        except:
            continue  # Skip if feature calculation fails
    
    feature_df = pd.DataFrame(company_features)
    
    if len(feature_df) < 2:
        print("Not enough data for clustering")
        return None
    
    # Standardize features
    feature_cols = [col for col in feature_df.columns if col != 'company_id']
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(feature_df[feature_cols].fillna(0))
    
    # Apply PCA for dimensionality reduction
    pca = PCA(n_components=min(5, len(feature_cols)))
    pca_result = pca.fit_transform(scaled_features)
    
    # Apply KMeans clustering
    kmeans = KMeans(n_clusters=CONFIG['clustering']['n_clusters'], 
                   random_state=CONFIG['clustering']['random_state'])
    clusters = kmeans.fit_predict(pca_result)
    
    # Add results to feature dataframe
    feature_df['cluster'] = clusters
    
    # Add PCA components
    for i in range(pca_result.shape[1]):
        feature_df[f'pca_{i+1}'] = pca_result[:, i]
    
    # Analyze cluster characteristics
    cluster_profiles = feature_df.groupby('cluster').agg({
        'util_mean': 'mean',
        'util_trend': 'mean',
        'deposit_trend': 'mean',
        'volatility': 'mean',
        'correlation': 'mean',
        'deposit_loan_ratio': 'mean',
        'company_id': 'count'
    }).reset_index()
    
    # Assign descriptive names to clusters
    cluster_names = []
    for _, row in cluster_profiles.iterrows():
        cluster_id = row['cluster']
        
        # Define a mapping rule for naming clusters
        if row['util_mean'] > 0.7 and row['deposit_loan_ratio'] < 1:
            name = "Credit_Dependent"
        elif row['util_trend'] > 0 and row['deposit_trend'] < 0:
            name = "Deteriorating_Health"
        elif row['util_mean'] < 0.4 and row['deposit_loan_ratio'] > 2:
            name = "Cautious_Borrower"
        elif row['volatility'] > cluster_profiles['volatility'].median() * 1.5:
            name = "Volatile_Behavior"
        elif row['deposit_trend'] < 0 and abs(row['util_trend']) < 0.001:
            name = "Cash_Constrained"
        else:
            name = f"Cluster_{cluster_id}"
        
        cluster_names.append({
            'cluster': cluster_id,
            'cluster_name': name,
            'size': row['company_id']
        })
    
    cluster_names_df = pd.DataFrame(cluster_names)
    
    return feature_df, cluster_profiles, cluster_names_df

def plot_clusters(feature_df):
    """
    Visualize clusters in PCA space
    """
    if feature_df is None or 'pca_1' not in feature_df.columns:
        print("No clustering data available for visualization.")
        return None
    
    plt.figure(figsize=(12, 8))
    
    # Create scatter plot of first two PCA components
    ax = sns.scatterplot(
        x='pca_1',
        y='pca_2',
        hue='cluster',
        data=feature_df,
        palette='viridis',
        s=100,
        alpha=0.7
    )
    
    # Add labels
    plt.title('Client Clusters in PCA Space', fontsize=16)
    plt.xlabel('Principal Component 1', fontsize=12)
    plt.ylabel('Principal Component 2', fontsize=12)
    
    plt.tight_layout()
    return plt.gcf()

def compare_personas_and_clusters(persona_df, feature_df):
    """
    Compare rule-based personas with unsupervised clusters
    """
    if persona_df is None or feature_df is None or persona_df.empty or 'cluster' not in feature_df.columns:
        print("Cannot compare personas and clusters: Missing data")
        return None
    
    # Get the latest persona for each company
    latest_personas = persona_df.sort_values('date').groupby('company_id').last()[['persona']]
    
    # Merge with cluster data
    comparison_df = feature_df[['company_id', 'cluster']].merge(
        latest_personas,
        on='company_id',
        how='inner'
    )
    
    if comparison_df.empty:
        print("No overlap between personas and clusters")
        return None
    
    # Create contingency table
    contingency = pd.crosstab(
        comparison_df['persona'],
        comparison_df['cluster'],
        margins=True
    )
    
    # Plot heatmap
    plt.figure(figsize=(12, 8))
    sns.heatmap(contingency.iloc[:-1, :-1], annot=True, fmt='d', cmap='Blues')
    plt.title('Comparison of Rule-Based Personas vs. Unsupervised Clusters', fontsize=14)
    plt.xlabel('Cluster')
    plt.ylabel('Persona')
    
    plt.tight_layout()
    return plt.gcf()

def main(df):
    """
    Main function to execute the entire analysis workflow with improved risk detection
    and persona-based cohort analysis.
    """
    print("Starting enhanced bank client risk analysis...")
    
    # 1. Clean data
    print("\nCleaning data...")
    df_clean, df_calc = clean_data(df, min_nonzero_pct=CONFIG['data']['min_nonzero_pct'])
    
    # 2. Add derived metrics with enhanced features
    print("\nAdding derived metrics and calculating trends...")
    df_with_metrics = add_derived_metrics(df_clean)
    
    # 3. Detect risk patterns with efficient implementation and persona assignment
    print("\nDetecting risk patterns and assigning personas...")
    risk_df, persona_df = detect_risk_patterns_efficient(df_with_metrics)
    print(f"Found {len(risk_df)} risk events across {risk_df['company_id'].nunique()} companies")
    print(f"Assigned {persona_df['persona'].nunique() if not persona_df.empty else 0} different personas")
    
    # 4. Perform clustering analysis
    print("\nPerforming clustering analysis...")
    cluster_results = perform_clustering(df_with_metrics)
    if cluster_results is not None:
        feature_df, cluster_profiles, cluster_names_df = cluster_results
        print(f"Created {cluster_profiles['cluster'].nunique()} clusters")
        
        # Plot clusters
        print("\nVisualizing clusters...")
        cluster_fig = plot_clusters(feature_df)
        if cluster_fig:
            plt.savefig('client_clusters.png')
            print("Saved cluster visualization to client_clusters.png")
        
        # Compare personas and clusters
        print("\nComparing personas and clusters...")
        comparison_fig = compare_personas_and_clusters(persona_df, feature_df)
        if comparison_fig:
            plt.savefig('persona_cluster_comparison.png')
            print("Saved comparison to persona_cluster_comparison.png")
    else:
        feature_df = None
    
    # 5. Create persona-based cohort analysis
    print("\nCreating persona-based cohort analysis...")
    cohort_results = create_personas_cohort_analysis(persona_df)
    if cohort_results[0] is not None:
        cohort_data, risk_cohort_data, quarterly_persona_df = cohort_results
        
        # 6. Plot persona cohort analysis
        print("\nPlotting persona cohort analysis...")
        cohort_fig = plot_persona_cohort(cohort_data)
        if cohort_fig:
            plt.savefig('persona_cohort_analysis.png')
            print("Saved persona cohort analysis to persona_cohort_analysis.png")
        
        # 7. Plot persona distribution
        print("\nPlotting persona distribution...")
        persona_dist_fig = plot_persona_distribution(persona_df)
        if persona_dist_fig:
            plt.savefig('persona_distribution.png')
            print("Saved persona distribution to persona_distribution.png")
        
        # 8. Plot persona transitions
        print("\nPlotting persona transitions...")
        transition_fig = plot_persona_transitions(quarterly_persona_df)
        if transition_fig:
            plt.savefig('persona_transitions.png')
            print("Saved persona transitions to persona_transitions.png")
    else:
        quarterly_persona_df = None
    
    # 9. Plot top risky companies with persona information
    print("\nPlotting top risky companies with persona information...")
    if not risk_df.empty:
        top_risky_companies = risk_df['company_id'].value_counts().head(5).index.tolist()
        
        for company_id in top_risky_companies:
            print(f"Plotting risk analysis for {company_id}...")
            company_fig = plot_risk_company(company_id, df_with_metrics, risk_df)
            if company_fig:
                company_fig.savefig(f'risk_analysis_{company_id}.png')
                print(f"Saved company risk analysis to risk_analysis_{company_id}.png")
    
    print("\nAnalysis complete! All visualization files saved.")
    
    return {
        'data': df_with_metrics,
        'risk_df': risk_df,
        'persona_df': persona_df,
        'cluster_df': feature_df,
        'quarterly_persona_df': quarterly_persona_df
    }

if __name__ == "__main__":
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Simple data generator to test the algorithm
    def generate_test_data(num_companies=100, days=1460):  # 4 years of data
        print("Generating test data...")
        
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
                
                # Calculate final values
                deposit = base_deposit * deposit_trend_component * seasonal_component * deposit_random * risk_factor_deposit
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
    
    # Generate test data
    df = generate_test_data(num_companies=100, days=1460)
    
    # Run the main analysis pipeline
    results = main(df)
