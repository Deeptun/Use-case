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

# Configuration dictionary for easy parameter tuning
CONFIG = {
    'data': {
        'min_nonzero_pct': 0.8,
        'min_continuous_days': 365,  # Minimum continuous days of data required
        'recent_window': 30  # Recent risk window in days
    },
    'risk': {
        'trend_windows': [30, 90, 180],  # days
        'change_thresholds': {
            'sharp': 0.2,    # 20% change
            'moderate': 0.1, # 10% change
            'gradual': 0.05  # 5% change
        },
        'persona_patterns': {
            'cautious_borrower': 'Low utilization (<40%), stable deposits',
            'aggressive_expansion': 'Rising utilization (>10% increase), volatile deposits',
            'distressed_client': 'High utilization (>80%), declining deposits (>5% decrease)',
            'seasonal_loan_user': 'Cyclical utilization with >15% amplitude',
            'seasonal_deposit_pattern': 'Cyclical deposits with >20% amplitude',
            'deteriorating_health': 'Rising utilization (>15% increase), declining deposits (>10% decrease)',
            'cash_constrained': 'Stable utilization, rapidly declining deposits (>15% decrease)',
            'credit_dependent': 'High utilization (>75%), low deposit ratio (<0.8)'
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

def clean_data(df, min_nonzero_pct=0.8):
    """
    Clean the data by:
    1. Removing companies where less than min_nonzero_pct of their values are non-zero
    2. Removing companies without the minimum continuous days of data
    3. Applying sophisticated imputation for missing values
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
    
    # Check for continuous data requirement
    continuous_companies = []
    for company in valid_companies:
        company_data = df_clean[df_clean['company_id'] == company].sort_values('date')
        
        # Check if company has data for the most recent period
        max_date = df_clean['date'].max()
        min_required_date = max_date - pd.Timedelta(days=CONFIG['data']['min_continuous_days'])
        
        # Get recent data
        recent_data = company_data[company_data['date'] >= min_required_date]
        
        # Ensure both deposit and loan data are available
        if (recent_data['deposit_balance'] > 0).sum() >= CONFIG['data']['min_continuous_days'] * 0.8 and \
           (recent_data['used_loan'] > 0).sum() >= CONFIG['data']['min_continuous_days'] * 0.8:
            continuous_companies.append(company)
    
    # Filter for companies with continuous data
    df_clean = df_clean[df_clean['company_id'].isin(continuous_companies)].copy()
    print(f"After continuous data filter: {df_clean.shape}")
    
    # Apply advanced imputation using KNN imputer for each company
    df_imputed = df_clean.copy()
    
    for company in tqdm(continuous_companies, desc="Applying KNN imputation"):
        company_data = df_clean[df_clean['company_id'] == company].sort_values('date')
        
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
            df_imputed.loc[company_data.index, 'deposit_balance'] = imputed_values[:, 0]
            df_imputed.loc[company_data.index, 'used_loan'] = imputed_values[:, 1]
            df_imputed.loc[company_data.index, 'unused_loan'] = imputed_values[:, 2]
    
    # Create a copy for calculations that preserves NaN values
    df_calc = df_imputed.copy()
    
    print(f"Cleaned and imputed data shape: {df_imputed.shape}")
    print(f"Retained {len(continuous_companies)}/{len(df['company_id'].unique())} companies")
    
    return df_imputed, df_calc

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
    df.loc[df['total_loan'] == 0, 'loan_utilization'] = np.nan
    
    # Calculate deposit to loan ratio
    df['deposit_loan_ratio'] = df['deposit_balance'] / df['used_loan']
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    
    # Calculate rolling metrics for each company
    for company in tqdm(df['company_id'].unique(), desc="Calculating rolling metrics"):
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
                                                                  .rolling(window=7, min_periods=3).mean())  # Smooth changes
            df.loc[company_data.index, f'deposit_change_{window}d'] = (company_data['deposit_balance'].pct_change(periods=window)
                                                                     .rolling(window=7, min_periods=3).mean())
            
        # Calculate volatility measures
        df.loc[company_data.index, 'util_volatility_30d'] = company_data['loan_utilization'].rolling(
            30, min_periods=10).std()
        df.loc[company_data.index, 'deposit_volatility_30d'] = company_data['deposit_balance'].pct_change().rolling(
            30, min_periods=10).std()
        
        # Detect seasonality
        detect_seasonality(df, company_data)
    
    return df

def detect_seasonality(df, company_data, min_periods=365):
    """
    Detect seasonality in both loan utilization and deposits.
    Adds seasonal metrics to the dataframe.
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
                if 350 <= period <= 380 or 170 <= period <= 190 or 80 <= period <= 100:  # Annual, semi-annual, quarterly
                    df.loc[company_data.index, 'util_seasonal_period'] = period
                    df.loc[company_data.index, 'util_seasonal_amplitude'] = normalized_amplitude
                    
                    # Flag as seasonal if amplitude is significant
                    df.loc[company_data.index, 'util_is_seasonal'] = normalized_amplitude > 0.15
        
        # For deposits
        if company_data['deposit_balance'].notna().sum() >= min_periods * 0.8:
            deposit_series = company_data['deposit_balance'].interpolate(method='linear')
            
            # Use FFT to detect seasonality
            fft_values = np.fft.fft(deposit_series.values - deposit_series.mean())
            fft_freq = np.fft.fftfreq(len(deposit_series))
            
            # Find peaks excluding the DC component
            peaks = find_peaks(np.abs(fft_values[1:len(fft_values)//2]), height=np.std(np.abs(fft_values))*2)[0] + 1
            
            if len(peaks) > 0:
                # Calculate period of strongest seasonal component
                strongest_idx = peaks[np.argmax(np.abs(fft_values)[peaks])]
                period = int(1 / np.abs(fft_freq[strongest_idx]))
                
                # Calculate seasonal amplitude
                seasonal_amplitude = np.abs(fft_values[strongest_idx]) * 2 / len(deposit_series)
                normalized_amplitude = seasonal_amplitude / deposit_series.mean() if deposit_series.mean() != 0 else 0
                
                # Record seasonality metrics
                if 350 <= period <= 380 or 170 <= period <= 190 or 80 <= period <= 100:  # Annual, semi-annual, quarterly
                    df.loc[company_data.index, 'deposit_seasonal_period'] = period
                    df.loc[company_data.index, 'deposit_seasonal_amplitude'] = normalized_amplitude
                    
                    # Flag as seasonal if amplitude is significant
                    df.loc[company_data.index, 'deposit_is_seasonal'] = normalized_amplitude > 0.20
    except:
        # If seasonality detection fails, continue without it
        pass

def detect_risk_patterns_efficient(df):
    """
    Efficient implementation of risk pattern detection based on rolling metrics
    and predefined risk rules. Assign each company to a persona based on their pattern.
    """
    risk_records = []
    persona_assignments = []
    
    # Time windows to analyze
    windows = CONFIG['risk']['trend_windows']
    
    # Get the latest date for recent risk calculation
    max_date = df['date'].max()
    recent_cutoff = max_date - pd.Timedelta(days=CONFIG['data']['recent_window'])
    
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
            persona =             from_persona = company_data.iloc[i]['persona']
            to_persona = company_data.iloc[i+1]['persona']
            from_date = company_data.iloc[i]['date']
            to_date = company_data.iloc[i+1]['date']
            
            # Get risk scores
            from_risk = persona_risk_scores.get(from_persona, 0)
            to_risk = persona_risk_scores.get(to_persona, 0)
            
            # Record transition
            transitions.append({
                'company_id': company_id,
                'from_date': from_date,
                'to_date': to_date,
                'from_persona': from_persona,
                'to_persona': to_persona,
                'risk_change': to_risk - from_risk
            })
            
            # Record significant risk increases
            if to_risk > from_risk and (to_date - from_date).days <= 90:
                risk_increases.append({
                    'company_id': company_id,
                    'from_date': from_date,
                    'to_date': to_date,
                    'from_persona': from_persona,
                    'to_persona': to_persona,
                    'risk_change': to_risk - from_risk,
                    'days_between': (to_date - from_date).days
                })
    
    transitions_df = pd.DataFrame(transitions)
    risk_increase_df = pd.DataFrame(risk_increases)
    
    return transitions_df, risk_increase_df

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

def plot_persona_cohort_enhanced(cohort_data):
    """
    Create an enhanced visualization of persona-based cohort analysis with improved aesthetics.
    """
    if cohort_data is None or cohort_data.empty:
        print("No cohort data available for plotting.")
        return None
    
    # Prepare data
    cohort_pct = cohort_data.div(cohort_data.sum(axis=1), axis=0)
    
    # Create figure with subplots - absolute counts and percentages
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 12), sharex=True)
    
    # Define a visually appealing color palette
    palette = sns.color_palette("viridis", len(cohort_data.columns))
    
    # Plot absolute values as stacked area chart
    cohort_data.plot(kind='area', stacked=True, alpha=0.8, ax=ax1, color=palette)
    
    # Plot percentages as stacked area chart
    cohort_pct.plot(kind='area', stacked=True, alpha=0.8, ax=ax2, color=palette)
    
    # Format the percentage y-axis
    ax2.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    
    # Add titles and labels
    ax1.set_title('Client Personas Over Time (Absolute Count)', fontsize=16, fontweight='bold')
    ax2.set_title('Client Personas Over Time (Percentage)', fontsize=16, fontweight='bold')
    ax1.set_ylabel('Number of Companies', fontsize=14)
    ax2.set_ylabel('Percentage of Companies', fontsize=14)
    ax2.set_xlabel('Quarter', fontsize=14)
    
    # Improve grid appearance
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax2.grid(True, alpha=0.3, linestyle='--')
    
    # Add annotations for absolute values
    for i, quarter in enumerate(cohort_data.index):
        y_pos = 0
        for persona in cohort_data.columns:
            count = cohort_data.loc[quarter, persona]
            if count > 0 and count > cohort_data.sum(axis=1).max() * 0.05:  # Only annotate significant values
                ax1.annotate(f"{int(count)}", 
                            xy=(i, y_pos + count/2), 
                            ha='center', 
                            va='center',
                            fontsize=9,
                            fontweight='bold',
                            color='white',
                            bbox=dict(boxstyle="round,pad=0.2", fc='black', alpha=0.6))
            y_pos += count
    
    # Add annotations for percentages
    for i, quarter in enumerate(cohort_pct.index):
        y_pos = 0
        for persona in cohort_pct.columns:
            pct = cohort_pct.loc[quarter, persona]
            if pct > 0.05:  # Only annotate if > 5%
                ax2.annotate(f"{pct:.1%}", 
                            xy=(i, y_pos + pct/2), 
                            ha='center', 
                            va='center',
                            fontsize=9,
                            fontweight='bold',
                            color='white',
                            bbox=dict(boxstyle="round,pad=0.2", fc='black', alpha=0.6))
            y_pos += pct
    
    # Add a legend with persona descriptions
    handles, labels = ax2.get_legend_handles_labels()
    
    # Create a custom legend with descriptions
    legend_entries = []
    for label in labels:
        description = CONFIG['risk']['persona_patterns'].get(label, "")
        legend_entries.append(f"{label}: {description}")
    
    # Place the legend outside the plot
    fig.legend(handles, legend_entries, loc='upper center', bbox_to_anchor=(0.5, 0), 
              fontsize=11, ncol=2, frameon=True, fancybox=True, shadow=True)
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.25)  # Make room for the legend
    
    return fig

def plot_persona_transitions(transitions_df, risk_increase_df):
    """
    Create visualizations for persona transitions, with focus on movements to riskier personas.
    """
    if transitions_df is None or transitions_df.empty:
        print("No transition data available for plotting.")
        return None, None
    
    # 1. Create a transition heatmap
    transition_counts = transitions_df.groupby(['from_persona', 'to_persona']).size().reset_index(name='count')
    pivot_transitions = transition_counts.pivot_table(
        index='from_persona', 
        columns='to_persona', 
        values='count',
        fill_value=0
    )
    
    # Create the heatmap
    fig1, ax1 = plt.subplots(figsize=(12, 10))
    
    # Plot heatmap with custom colormap that emphasizes higher values
    cmap = sns.color_palette("YlGnBu", as_cmap=True)
    heatmap = sns.heatmap(pivot_transitions, annot=True, fmt='d', cmap=cmap, 
                         cbar_kws={'label': 'Number of Transitions'})
    
    # Improve appearance
    ax1.set_title('Persona Transitions Between Time Periods', fontsize=16, fontweight='bold')
    ax1.set_xlabel('To Persona', fontsize=14)
    ax1.set_ylabel('From Persona', fontsize=14)
    
    # 2. Create a visualization of risk increasing transitions
    if not risk_increase_df.empty:
        fig2, ax2 = plt.subplots(figsize=(14, 8))
        
        # Group by from/to personas and count transitions
        risk_trans_counts = risk_increase_df.groupby(['from_persona', 'to_persona']).size().reset_index(name='count')
        
        # Create a mapping of persona pairs to coordinates for the plot
        persona_list = list(set(risk_trans_counts['from_persona'].tolist() + risk_trans_counts['to_persona'].tolist()))
        persona_coords = {persona: idx for idx, persona in enumerate(persona_list)}
        
        # Create a scatter plot of transitions
        for _, row in risk_trans_counts.iterrows():
            x = persona_coords[row['from_persona']]
            y = persona_coords[row['to_persona']]
            count = row['count']
            
            # Calculate point size based on count
            size = 100 + (count * 30)
            
            # Calculate color based on risk change
            avg_change = risk_increase_df[(risk_increase_df['from_persona'] == row['from_persona']) & 
                                         (risk_increase_df['to_persona'] == row['to_persona'])]['risk_change'].mean()
            
            # Plot the transition
            scatter = ax2.scatter(x, y, s=size, alpha=0.7, 
                                 c=[avg_change], cmap='Reds', 
                                 edgecolors='black', linewidths=1)
            
            # Add count label
            ax2.annotate(str(count), xy=(x, y), xytext=(0, 0), 
                        textcoords="offset points", ha='center', va='center',
                        fontweight='bold', color='white')
        
        # Add a colorbar
        cbar = plt.colorbar(scatter)
        cbar.set_label('Risk Increase Magnitude', fontsize=12)
        
        # Set axis labels and ticks
        ax2.set_title('Risk-Increasing Persona Transitions', fontsize=16, fontweight='bold')
        ax2.set_xlabel('From Persona', fontsize=14)
        ax2.set_ylabel('To Persona', fontsize=14)
        ax2.set_xticks(range(len(persona_list)))
        ax2.set_yticks(range(len(persona_list)))
        ax2.set_xticklabels(persona_list, rotation=45, ha='right')
        ax2.set_yticklabels(persona_list)
        
        # Add grid
        ax2.grid(True, linestyle='--', alpha=0.3)
        
        plt.tight_layout()
        
        return fig1, fig2
    
    return fig1, None

def plot_risk_company(company_id, df, risk_df, recent_risks_df):
    """
    Create a detailed plot of a company's risk patterns with persona information.
    Shows loan utilization and deposit balance on two axes with risk annotations.
    Only shows recent risks for better clarity.
    """
    # Check if the company is in the recent risks list
    if recent_risks_df is not None and not recent_risks_df.empty:
        recent_company = recent_risks_df[recent_risks_df['company_id'] == company_id]
        if recent_company.empty:
            print(f"Company {company_id} does not have recent risk events.")
            return None
    
    # Filter data for company
    company_data = df[df['company_id'] == company_id].sort_values('date')
    
    # Get recent date range (focus on last year)
    end_date = company_data['date'].max()
    start_date = end_date - pd.Timedelta(days=365)
    
    # Focus on recent data for better visibility
    company_data = company_data[(company_data['date'] >= start_date)]
    
    # Get recent risk events
    company_risks = risk_df[(risk_df['company_id'] == company_id) & 
                            (risk_df['date'] >= start_date)].sort_values('date')
    
    if company_data.empty:
        print(f"No data found for company {company_id}")
        return None
    
    # Create figure with two y-axes
    fig, ax1 = plt.subplots(figsize=(16, 10))
    
    # First axis - Loan Utilization
    color = '#1f77b4'  # Blue
    ax1.set_xlabel('Date', fontsize=12)
    ax1.set_ylabel('Loan Utilization (%)', color=color, fontsize=12)
    
    # Plot line with markers at data points
    ax1.plot(company_data['date'], company_data['loan_utilization'] * 100, 
             color=color, linewidth=2.5, marker='o', markersize=4)
    
    # Add translucent range for better visualization
    ax1.fill_between(company_data['date'], company_data['loan_utilization'] * 100, 
                    color=color, alpha=0.2)
    
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.set_ylim(0, 110)  # Slightly higher to make room for annotations
    
    # Second axis - Deposits
    color = '#ff7f0e'  # Orange
    ax2 = ax1.twinx()
    ax2.set_ylabel('Deposit Balance', color=color, fontsize=12)
    
    # Plot line with markers at data points
    ax2.plot(company_data['date'], company_data['deposit_balance'], 
             color=color, linewidth=2.5, marker='o', markersize=4)
    
    # Add translucent range for better visualization
    ax2.fill_between(company_data['date'], company_data['deposit_balance'], 
                    color=color, alpha=0.2)
    
    ax2.tick_params(axis='y', labelcolor=color)
    
    # Format dates on x-axis
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    plt.xticks(rotation=45)
    
    # Add risk markers
    if not company_risks.empty:
        risk_dates = company_risks['date'].tolist()
        risk_descriptions = company_risks['risk_description'].tolist()
        risk_levels = company_risks['risk_level'].tolist()
        personas = company_risks['persona'].tolist()
        
        # Create color map for risk levels
        risk_colors = {'high': '#d62728', 'medium': '#ff7f0e', 'low': '#2ca02c'}
        
        # Add vertical lines for each risk event
        for i, (date, desc, level, persona) in enumerate(zip(risk_dates, risk_descriptions, risk_levels, personas)):
            # Only show recent risk events
            ax1.axvline(x=date, color=risk_colors[level], linestyle='--', alpha=0.7, linewidth=1.5)
            
            # Add label with risk level and persona
            ax1.annotate(f"{level.upper()} | {persona}", 
                         xy=(date, 103),  # Position at top of plot
                         xytext=(0, 0),
                         textcoords="offset points",
                         ha='center', 
                         va='center',
                         fontsize=9,
                         fontweight='bold',
                         color='white',
                         bbox=dict(boxstyle="round,pad=0.2", fc=risk_colors[level], alpha=0.8))
            
            # Add descriptions (shortened if too long)
            short_desc = desc.split('|')[0] if len(desc) > 50 else desc
            y_pos = 95 - (i % 4) * 10  # Stagger text vertically
            
            ax1.annotate(short_desc, 
                         xy=(date, y_pos), 
                         xytext=(5, 0), 
                         textcoords="offset points", 
                         color='black',
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
        
        # Get most common risk flag from recent_risks_df
        if recent_risks_df is not None and not recent_risks_df.empty:
            company_recent = recent_risks_df[recent_risks_df['company_id'] == company_id]
            if not company_recent.empty:
                most_common_flag = company_recent.iloc[0]['most_common_flag']
                persona_text = f"\nCurrent Persona: {recent_persona} - {persona_desc}\nMost Common Risk: {most_common_flag}"
    
    plt.title(f"Risk Analysis for {company_id}{persona_text}", fontsize=16, fontweight='bold')
    
    # Add key metrics in text box
    avg_util = company_data['loan_utilization'].mean() * 100
    avg_deposit = company_data['deposit_balance'].mean()
    recent_util_change = company_data['loan_utilization'].pct_change(periods=30).iloc[-1] * 100 if len(company_data) > 30 else np.nan
    recent_deposit_change = company_data['deposit_balance'].pct_change(periods=30).iloc[-1] * 100 if len(company_data) > 30 else np.nan
    
    metrics_text = (
        f"Average Utilization: {avg_util:.1f}%\n"
        f"Average Deposit: ${avg_deposit:,.2f}\n"
        f"30-day Utilization Change: {recent_util_change:.1f}%\n"
        f"30-day Deposit Change: {recent_deposit_change:.1f}%\n"
        f"Risk Events: {len(company_risks)}"
    )
    
    # Add text box with metrics
    props = dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8, edgecolor='gray')
    ax1.text(0.02, 0.04, metrics_text, transform=ax1.transAxes, fontsize=10,
            verticalalignment='bottom', bbox=props)
    
    # Add seasonal indicators if available
    if 'util_is_seasonal' in company_data.columns and 'deposit_is_seasonal' in company_data.columns:
        is_util_seasonal = company_data['util_is_seasonal'].any()
        is_deposit_seasonal = company_data['deposit_is_seasonal'].any()
        
        if is_util_seasonal or is_deposit_seasonal:
            seasonal_text = "Seasonal Patterns:\n"
            
            if is_util_seasonal:
                util_amplitude = company_data['util_seasonal_amplitude'].dropna().mean() * 100
                seasonal_text += f"• Loan Utilization: {util_amplitude:.1f}% amplitude\n"
                
            if is_deposit_seasonal:
                deposit_amplitude = company_data['deposit_seasonal_amplitude'].dropna().mean() * 100
                seasonal_text += f"• Deposits: {deposit_amplitude:.1f}% amplitude"
            
            # Add seasonal text box
            seasonal_props = dict(boxstyle='round,pad=0.5', facecolor='lightyellow', alpha=0.9, edgecolor='orange')
            ax1.text(0.98, 0.04, seasonal_text, transform=ax1.transAxes, fontsize=10,
                    ha='right', va='bottom', bbox=seasonal_props)
    
    plt.tight_layout()
    return fig

def plot_persona_distribution(persona_df, affinity_df=None):
    """
    Plot distribution of personas across the dataset with affinity information.
    """
    if persona_df is None or persona_df.empty:
        print("No persona data available for distribution analysis.")
        return None
    
    # Only consider recent persona assignments
    recent_persona_df = persona_df[persona_df['is_recent'] == True]
    
    if recent_persona_df.empty:
        recent_persona_df = persona_df  # Fallback to all data if no recent data
    
    # Count each persona
    persona_counts = recent_persona_df.groupby('persona').size().reset_index(name='count')
    persona_counts = persona_counts.sort_values('count', ascending=False)
    
    if persona_counts.empty:
        return None
    
    # Create plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
    
    # Create bar plot of counts
    sns.barplot(x='persona', y='count', data=persona_counts, palette='viridis', ax=ax1)
    
    # Add percentage labels
    total = persona_counts['count'].sum()
    for i, p in enumerate(ax1.patches):
        percentage = 100 * p.get_height() / total
        ax1.annotate(f"{percentage:.1f}%", 
                   (p.get_x() + p.get_width() / 2., p.get_height() + 0.5),
                   ha='center', 
                   va='bottom', 
                   fontsize=10,
                   fontweight='bold')
    
    # Add titles and labels
    ax1.set_title('Distribution of Client Personas', fontsize=16, fontweight='bold')
    ax1.set_xlabel('Persona', fontsize=12)
    ax1.set_ylabel('Number of Companies', fontsize=12)
    ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45, ha='right')
    
    # Add affinity distribution if available
    if affinity_df is not None and not affinity_df.empty:
        # Group by persona and calculate average affinity
        affinity_stats = affinity_df.groupby('persona').agg({
            'affinity_score': ['mean', 'std', 'min', 'max', 'count']
        }).reset_index()
        
        affinity_stats.columns = ['persona', 'mean_affinity', 'std_affinity', 'min_affinity', 'max_affinity', 'count']
        affinity_stats = affinity_stats.sort_values('mean_affinity', ascending=False)
        
        # Create affinity boxplot
        sns.boxplot(x='persona', y='affinity_score', data=affinity_df, 
                   palette='viridis', ax=ax2)
        
        # Add jittered points for individual company affinities
        sns.stripplot(x='persona', y='affinity_score', data=affinity_df, 
                     color='black', size=3, alpha=0.4, jitter=True, ax=ax2)
        
        # Add mean labels
        for i, row in affinity_stats.iterrows():
            persona = row['persona']
            mean_affinity = row['mean_affinity']
            x_pos = i
            
            # Find y position of corresponding box
            ax2.annotate(f"Avg: {mean_affinity:.2f}", 
                       (x_pos, mean_affinity),
                       xytext=(0, 5),  # 5 points vertical offset
                       textcoords="offset points",
                       ha='center', 
                       va='bottom',
                       fontsize=9,
                       fontweight='bold',
                       bbox=dict(boxstyle="round,pad=0.2", fc='white', alpha=0.8))
        
        ax2.set_title('Persona Affinity Scores', fontsize=16, fontweight='bold')
        ax2.set_xlabel('Persona', fontsize=12)
        ax2.set_ylabel('Affinity Score', fontsize=12)
        ax2.set_xticklabels(ax2.get_xticklabels(), rotation=45, ha='right')
    
    # Add persona descriptions as a table
    persona_desc = pd.DataFrame(list(CONFIG['risk']['persona_patterns'].items()), 
                                columns=['Persona', 'Description'])
    
    # Create a table at the bottom
    table = plt.table(cellText=persona_desc.values,
              colLabels=persona_desc.columns,
              loc='bottom',
              cellLoc='center',
              bbox=[0, -0.50, 1, 0.3])
    
    # Style the table
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.5)
    
    plt.subplots_adjust(bottom=0.35)  # Make room for the table
    plt.tight_layout()
    
    return fig

def perform_clustering(df):
    """
    Use unsupervised learning to cluster companies based on behavioral patterns.
    Provide more user-friendly descriptions of clusters.
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
            
            # Calculate linear trend using more robust method
            dates_numeric = (company_data['date'] - company_data['date'].min()).dt.days
            util_trend_model = sm.OLS(company_data['loan_utilization'].fillna(method='ffill'), 
                                     sm.add_constant(dates_numeric)).fit()
            util_trend = util_trend_model.params[1]  # Slope coefficient
            
            # Deposit statistics 
            deposit_mean = company_data['deposit_balance'].mean()
            deposit_std = company_data['deposit_balance'].std()
            
            deposit_trend_model = sm.OLS(company_data['deposit_balance'].fillna(method='ffill'), 
                                        sm.add_constant(dates_numeric)).fit()
            deposit_trend = deposit_trend_model.params[1]  # Slope coefficient
            
            # Normalize trend by average value
            util_trend_pct = util_trend * 30 / (util_mean + 1e-10)  # 30-day change as percentage
            deposit_trend_pct = deposit_trend * 30 / (deposit_mean + 1e-10)  # 30-day change as percentage
            
            # Volatility and correlation
            volatility_metric = company_data['loan_utilization'].diff().abs().mean()
            correlation = company_data['loan_utilization'].corr(company_data['deposit_balance'])
            
            # Ratio statistics
            deposit_loan_ratio = company_data['deposit_loan_ratio'].mean()
            
            # Seasonal metrics
            util_seasonal = company_data.get('util_is_seasonal', pd.Series([False] * len(company_data))).max()
            deposit_seasonal = company_data.get('deposit_is_seasonal', pd.Series([False] * len(company_data))).max()
            
            util_amplitude = company_data.get('util_seasonal_amplitude', pd.Series([0] * len(company_data))).mean()
            deposit_amplitude = company_data.get('deposit_seasonal_amplitude', pd.Series([0] * len(company_data))).mean()
            
            # Feature vector
            company_features.append({
                'company_id': company_id,
                'util_mean': util_mean,
                'util_std': util_std,
                'util_trend': util_trend,
                'util_trend_pct': util_trend_pct,
                'deposit_mean': deposit_mean,
                'deposit_std': deposit_std,
                'deposit_trend': deposit_trend,
                'deposit_trend_pct': deposit_trend_pct,
                'volatility': volatility_metric,
                'correlation': correlation if not np.isnan(correlation) else 0,
                'deposit_loan_ratio': deposit_loan_ratio if not np.isnan(deposit_loan_ratio) else 0,
                'util_seasonal': util_seasonal,
                'deposit_seasonal': deposit_seasonal,
                'util_amplitude': util_amplitude,
                'deposit_amplitude': deposit_amplitude
            })
        except:
            continue  # Skip if feature calculation fails
    
    feature_df = pd.DataFrame(company_features)
    
    if len(feature_df) < 2:
        print("Not enough data for clustering")
        return None
    
    # Create a more informative feature representation for clustering
    feature_cols = [
        'util_mean', 'util_trend_pct', 'deposit_trend_pct', 
        'volatility', 'correlation', 'deposit_loan_ratio',
        'util_amplitude', 'deposit_amplitude'
    ]
    
    # Standardize features
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
    
    # Analyze cluster characteristics with detailed descriptions
    cluster_profiles = []
    
    for cluster_id in range(CONFIG['clustering']['n_clusters']):
        cluster_data = feature_df[feature_df['cluster'] == cluster_id]
        
        # Skip if empty cluster
        if cluster_data.empty:
            continue
            
        # Calculate mean values for key metrics
        util_mean = cluster_data['util_mean'].mean()
        util_trend_pct = cluster_data['util_trend_pct'].mean() * 100  # Convert to percentage
        deposit_trend_pct = cluster_data['deposit_trend_pct'].mean() * 100  # Convert to percentage
        volatility = cluster_data['volatility'].mean()
        correlation = cluster_data['correlation'].mean()
        deposit_loan_ratio = cluster_data['deposit_loan_ratio'].mean()
        util_seasonal_pct = (cluster_data['util_seasonal'] == True).mean() * 100
        deposit_seasonal_pct = (cluster_data['deposit_seasonal'] == True).mean() * 100
        count = len(cluster_data)
        
        # Create human-readable descriptions
        util_description = (
            f"average {util_mean:.1%}" + 
            (f", increasing by {util_trend_pct:.1f}% monthly" if util_trend_pct > 0.5 else
             f", decreasing by {-util_trend_pct:.1f}% monthly" if util_trend_pct < -0.5 else
             ", stable")
        )
        
        deposit_description = (
            f"increasing by {deposit_trend_pct:.1f}% monthly" if deposit_trend_pct > 0.5 else
            f"decreasing by {-deposit_trend_pct:.1f}% monthly" if deposit_trend_pct < -0.5 else
            "stable"
        )
        
        ratio_description = (
            f"high ({deposit_loan_ratio:.1f})" if deposit_loan_ratio > 2 else
            f"moderate ({deposit_loan_ratio:.1f})" if deposit_loan_ratio > 1 else
            f"low ({deposit_loan_ratio:.1f})"
        )
        
        seasonal_description = ""
        if util_seasonal_pct > 30 or deposit_seasonal_pct > 30:
            seasonal_parts = []
            if util_seasonal_pct > 30:
                seasonal_parts.append(f"{util_seasonal_pct:.0f}% show loan seasonality")
            if deposit_seasonal_pct > 30:
                seasonal_parts.append(f"{deposit_seasonal_pct:.0f}% show deposit seasonality")
            seasonal_description = f" - {' and '.join(seasonal_parts)}"
        
        # Create cluster summary
        description = (
            f"Loan utilization {util_description}, deposits {deposit_description}, "
            f"deposit-to-loan ratio {ratio_description}{seasonal_description}"
        )
        
        # Map to closest persona
        if util_mean > 0.75 and deposit_loan_ratio < 0.8:
            closest_persona = "credit_dependent"
        elif util_trend_pct > 10 and deposit_trend_pct < -5:
            closest_persona = "deteriorating_health"
        elif util_mean < 0.4 and deposit_loan_ratio > 2:
            closest_persona = "cautious_borrower"
        elif volatility > feature_df['volatility'].median() * 1.5:
            closest_persona = "aggressive_expansion"
        elif deposit_trend_pct < -10 and abs(util_trend_pct) < 3:
            closest_persona = "cash_constrained"
        elif util_seasonal_pct > 50:
            closest_persona = "seasonal_loan_user"
        elif deposit_seasonal_pct > 50:
            closest_persona = "seasonal_deposit_pattern"
        else:
            closest_persona = "stable_client"
        
        cluster_profiles.append({
            'cluster': cluster_id,
            'size': count,
            'util_mean': util_mean,
            'util_trend_pct': util_trend_pct,
            'deposit_trend_pct': deposit_trend_pct,
            'volatility': volatility,
            'correlation': correlation,
            'deposit_loan_ratio': deposit_loan_ratio,
            'util_seasonal_pct': util_seasonal_pct,
            'deposit_seasonal_pct': deposit_seasonal_pct,
            'description': description,
            'closest_persona': closest_persona
        })
    
    cluster_profiles_df = pd.DataFrame(cluster_profiles)
    
    return feature_df, cluster_profiles_df

def plot_clusters(feature_df, cluster_profiles_df):
    """
    Visualize clusters in PCA space with detailed descriptions
    """
    if feature_df is None or 'pca_1' not in feature_df.columns:
        print("No clustering data available for visualization.")
        return None
    
    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
    
    # Create scatter plot of first two PCA components
    scatter = ax1.scatter(
        feature_df['pca_1'],
        feature_df['pca_2'],
        c=feature_df['cluster'],
        cmap='viridis',
        s=100,
        alpha=0.7,
        edgecolors='white'
    )
    
    # Add labels for each cluster center
    for cluster_id, profile in cluster_profiles_df.iterrows():
        cluster_points = feature_df[feature_df['cluster'] == profile['cluster']]
        center_x = cluster_points['pca_1'].mean()
        center_y = cluster_points['pca_2'].mean()
        
        # Add a star marker at the cluster center
        ax1.scatter(center_x, center_y, marker='*', s=300, color='red', edgecolors='black')
        
        # Add cluster label
        ax1.text(center_x, center_y + 0.2, f"Cluster {profile['cluster']}", 
                 ha='center', va='bottom', fontsize=12, fontweight='bold',
                 bbox=dict(boxstyle="round,pad=0.3", fc='white', alpha=0.8))
    
    # Add labels and title
    ax1.set_title('Client Clusters in PCA Space', fontsize=16, fontweight='bold')
    ax1.set_xlabel('Principal Component 1', fontsize=12)
    ax1.set_ylabel('Principal Component 2', fontsize=12)
    ax1.grid(True, linestyle='--', alpha=0.7)
    
    # Create a legend for the scatter plot
    legend1 = ax1.legend(*scatter.legend_elements(),
                        loc="upper right", title="Clusters")
    ax1.add_artist(legend1)
    
    # Create key metrics visualization for each cluster
    cluster_ids = cluster_profiles_df['cluster'].astype(int).tolist()
    metrics = ['util_mean', 'util_trend_pct', 'deposit_trend_pct', 'deposit_loan_ratio']
    labels = ['Utilization', 'Util Trend %/mo', 'Deposit Trend %/mo', 'Deposit-Loan Ratio']
    
    # Prepare data for visualization
    plot_data = []
    for metric, label in zip(metrics, labels):
        for cluster_id in cluster_ids:
            profile = cluster_profiles_df[cluster_profiles_df['cluster'] == cluster_id].iloc[0]
            plot_data.append({
                'cluster': f'Cluster {cluster_id}',
                'metric': label,
                'value': profile[metric]
            })
    
    plot_df = pd.DataFrame(plot_data)
    
    # Create heatmap of key metrics for each cluster
    pivot_df = plot_df.pivot(index='metric', columns='cluster', values='value')
    
    # Custom normalization for each row to highlight differences
    normalized_data = pivot_df.copy()
    for metric in pivot_df.index:
        row_min = pivot_df.loc[metric].min()
        row_max = pivot_df.loc[metric].max()
        if row_max > row_min:
            normalized_data.loc[metric] = (pivot_df.loc[metric] - row_min) / (row_max - row_min)
    
    # Create heatmap with custom formatting function
    def fmt(x):
        if 'Trend' in pivot_df.index[x[0]]:
            return f"{pivot_df.iloc[x[0], x[1]]:.1f}%"
        elif 'Ratio' in pivot_df.index[x[0]]:
            return f"{pivot_df.iloc[x[0], x[1]]:.2f}"
        elif 'Utilization' in pivot_df.index[x[0]]:
            return f"{pivot_df.iloc[x[0], x[1]]:.1%}"
        else:
            return f"{pivot_df.iloc[x[0], x[1]]:.2f}"
    
    sns.heatmap(normalized_data, annot=pivot_df, fmt="", cmap="YlGnBu", 
                linewidths=0.5, ax=ax2, annot_kws={"fontsize":10},
                cbar_kws={'label': 'Relative Value (Row-normalized)'})
    
    ax2.set_title('Key Metrics by Cluster', fontsize=16, fontweight='bold')
    
    # Add cluster descriptions in a table below
    cluster_desc = cluster_profiles_df[['cluster', 'description', 'closest_persona']].copy()
    cluster_desc['cluster'] = 'Cluster ' + cluster_desc['cluster'].astype(str)
    cluster_desc.columns = ['Cluster', 'Description', 'Similar Persona']
    
    # Create a table at the bottom
    table = plt.table(cellText=cluster_desc.values,
                     colLabels=cluster_desc.columns,
                     loc='bottom',
                     cellLoc='left',
                     colWidths=[0.1, 0.7, 0.2],
                     bbox=[0, -0.8, 1, 0.5])
    
    # Style the table
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.8)
    
    # Make the rows auto wrap
    for (row, col), cell in table.get_celld().items():
        if col == 1:  # Description column
            cell.set_text_props(wrap=True)
            cell.set_height(0.12)
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.4)  # Make room for the table
    
    return fig

def main(df):
    """
    Main function to execute the entire analysis workflow with improved risk detection,
    persona-based cohort analysis, and enhanced visualizations.
    """
    print("Starting enhanced bank client risk analysis...")
    
    # 1. Clean data with enhanced imputation techniques
    print("\nCleaning data and applying advanced imputation...")
    df_clean, df_calc = clean_data(df, min_nonzero_pct=CONFIG['data']['min_nonzero_pct'])
    
    # 2. Add derived metrics with enhanced features, including seasonality detection
    print("\nAdding derived metrics and detecting seasonality...")
    df_with_metrics = add_derived_metrics(df_clean)
    
    # 3. Detect risk patterns and assign personas
    print("\nDetecting risk patterns and assigning personas...")
    risk_df, persona_df, recent_risk_df = detect_risk_patterns_efficient(df_with_metrics)
    print(f"Found {len(risk_df)} risk events across {risk_df['company_id'].nunique()} companies")
    print(f"Identified {len(recent_risk_df)} companies with recent risk events")
    print(f"Assigned {persona_df['persona'].nunique() if not persona_df.empty else 0} different personas")
    
    # 4. Calculate persona affinity scores
    print("\nCalculating persona affinity scores...")
    affinity_df = calculate_persona_affinity(persona_df)
    print(f"Calculated affinity scores for {len(affinity_df['company_id'].unique())} companies")
    
    # 5. Track persona transitions with focus on risk increases
    print("\nTracking persona transitions...")
    transitions_df, risk_increase_df = track_persona_transitions(persona_df)
    print(f"Identified {len(risk_increase_df) if risk_increase_df is not None else 0} risk-increasing transitions")
    
    # 6. Perform clustering analysis with better descriptions
    print("\nPerforming clustering analysis...")
    clustering_results = perform_clustering(df_with_metrics)
    if clustering_results is not None:
        feature_df, cluster_profiles_df = clustering_results
        print(f"Created {len(cluster_profiles_df)} clusters with detailed descriptions")
        
        # Plot clusters
        print("\nVisualizing clusters...")
        cluster_fig = plot_clusters(feature_df, cluster_profiles_df)
        if cluster_fig:
            plt.savefig('client_clusters.png')
            print("Saved cluster visualization to client_clusters.png")
    else:
        feature_df = None
        cluster_profiles_df = None
    
    # 7. Create persona-based cohort analysis
    print("\nCreating persona-based cohort analysis...")
    cohort_results = create_personas_cohort_analysis(persona_df)
    if cohort_results[0] is not None:
        cohort_data, risk_cohort_data, quarterly_persona_df = cohort_results
        
        # 8. Plot enhanced persona cohort analysis
        print("\nPlotting enhanced persona cohort analysis...")
        cohort_fig = plot_persona_cohort_enhanced(cohort_data)
        if cohort_fig:
            plt.savefig('persona_cohort_analysis.png')
            print("Saved persona cohort analysis to persona_cohort_analysis.png")
        
        # 9. Plot persona distribution with affinity scores
        print("\nPlotting persona distribution with affinity scores...")
        persona_dist_fig = plot_persona_distribution(persona_df, affinity_df)
        if persona_dist_fig:
            plt.savefig('persona_distribution.png')
            print("Saved persona distribution to persona_distribution.png")
        
        # 10. Plot persona transitions with focus on risk increases
        print("\nPlotting persona transitions with risk focus...")
        transition_figs = plot_persona_transitions(transitions_df, risk_increase_df)
        if transition_figs[0]:
            plt.figure(transition_figs[0].number)
            plt.savefig('persona_transitions.png')
            print("Saved persona transitions to persona_transitions.png")
            
        if transition_figs[1]:
            plt.figure(transition_figs[1].number)
            plt.savefig('risk_increasing_transitions.png')
            print("Saved risk-increasing transitions to risk_increasing_transitions.png")
    else:
        quarterly_persona_df = None
    
    # 11. Plot risky companies with improved visualizations
    print("\nPlotting risky companies with enhanced visualizations...")
    if not recent_risk_df.empty:
        top_risky_companies = recent_risk_df['company_id'].head(5).tolist()
        
        for company_id in top_risky_companies:
            print(f"Plotting risk analysis for {company_id}...")
            company_fig = plot_risk_company(company_id, df_with_metrics, risk_df, recent_risk_df)
            if company_fig:
                company_fig.savefig(f'risk_analysis_{company_id}.png')
                print(f"Saved company risk analysis to risk_analysis_{company_id}.png")
    
    print("\nAnalysis complete! All visualization files saved.")
    
    return {
        'data': df_with_metrics,
        'risk_df': risk_df,
        'persona_df': persona_df,
        'recent_risk_df': recent_risk_df,
        'affinity_df': affinity_df,
        'transitions_df': transitions_df,
        'risk_increase_df': risk_increase_df,
        'cluster_df': feature_df,
        'cluster_profiles': cluster_profiles_df,
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

            persona_confidence = 0.0
            
            # ---- RISK PATTERN 1: Rising utilization with declining deposits ----
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
            
            # ---- RISK PATTERN 2: High utilization with low deposit ratio ----
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
            
            # ---- RISK PATTERN 3: Rapid deposit decline with stable utilization ----
            if not pd.isna(current_row.get('deposit_change_30d')) and not pd.isna(current_row.get('util_change_30d')):
                deposit_change_30d = current_row['deposit_change_30d']
                util_change_30d = current_row['util_change_30d']
                
                if deposit_change_30d < -0.15 and abs(util_change_30d) < 0.05:
                    severity = "high" if deposit_change_30d < -0.25 else "medium"
                    risk_flags.append('cash_drain_30d')
                    risk_descriptions.append(
                        f"[{severity.upper()}] 30d: Rapid deposit decline ({deposit_change_30d:.1%}) "
                        f"with stable utilization (change: {util_change_30d:.1%})"
                    )
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
                            risk_descriptions.append(
                                f"[MEDIUM] Significant increase in volatility for both metrics "
                                f"(util: {past_vol_u:.4f}→{current_vol_u:.4f}, deposit: {past_vol_d:.4f}→{current_vol_d:.4f})"
                            )
                            risk_levels.append("medium")
                            
                            if persona_confidence < 0.6:
                                persona = "aggressive_expansion"
                                persona_confidence = 0.6
            
            # ---- RISK PATTERN 5: Loan Utilization Seasonality ----
            if current_row.get('util_is_seasonal') == True:
                amplitude = current_row.get('util_seasonal_amplitude', 0)
                if amplitude > 0.2:  # More than 20% seasonal variation
                    risk_flags.append('seasonal_util')
                    risk_descriptions.append(
                        f"[LOW] Seasonal loan utilization with {amplitude:.1%} amplitude "
                        f"(period: {current_row.get('util_seasonal_period', 0):.0f} days)"
                    )
                    risk_levels.append("low")
                    
                    if persona_confidence < 0.65:
                        persona = "seasonal_loan_user"
                        persona_confidence = 0.65
            
            # ---- RISK PATTERN 6: Deposit Seasonality ----
            if current_row.get('deposit_is_seasonal') == True:
                amplitude = current_row.get('deposit_seasonal_amplitude', 0)
                if amplitude > 0.25:  # More than 25% seasonal variation
                    risk_flags.append('seasonal_deposit')
                    risk_descriptions.append(
                        f"[LOW] Seasonal deposit pattern with {amplitude:.1%} amplitude "
                        f"(period: {current_row.get('deposit_seasonal_period', 0):.0f} days)"
                    )
                    risk_levels.append("low")
                    
                    if persona_confidence < 0.65 and persona != "seasonal_loan_user":
                        persona = "seasonal_deposit_pattern"
                        persona_confidence = 0.65
            
            # ---- RISK PATTERN 7: Combined Seasonal Risk ----
            if (current_row.get('util_is_seasonal') == True and 
                current_row.get('deposit_is_seasonal') == True):
                util_amplitude = current_row.get('util_seasonal_amplitude', 0)
                deposit_amplitude = current_row.get('deposit_seasonal_amplitude', 0)
                
                # If loan volatility is higher than deposit volatility, potential risk
                if util_amplitude > deposit_amplitude * 1.5 and util_amplitude > 0.25:
                    risk_flags.append('seasonal_imbalance')
                    risk_descriptions.append(
                        f"[MEDIUM] Seasonal imbalance: Loan utilization amplitude ({util_amplitude:.1%}) "
                        f"exceeds deposit amplitude ({deposit_amplitude:.1%})"
                    )
                    risk_levels.append("medium")
            
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
                    'is_recent': current_date >= recent_cutoff
                })
    
    # Create risk dataframe
    if risk_records:
        risk_df = pd.DataFrame(risk_records)
        persona_df = pd.DataFrame(persona_assignments)
        
        # Create a dataframe of recent risks (last 30 days)
        recent_risks = risk_df[risk_df['is_recent'] == True].copy()
        
        # For each company, find the most frequent risk flag in the recent period
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
                                          'risk_level', 'is_recent'])
        recent_risk_summary = pd.DataFrame(columns=['company_id', 'latest_date', 'most_common_flag',
                                                   'risk_level', 'persona', 'current_util', 'current_deposit'])
        return risk_df, persona_df, recent_risk_summary

def calculate_persona_affinity(persona_df):
    """
    Calculate affinity scores for each company-persona combination.
    This measures how strongly a company is associated with each persona over time.
    """
    if persona_df.empty:
        return pd.DataFrame()
    
    # Calculate persona frequency for each company
    affinity_scores = []
    
    for company in persona_df['company_id'].unique():
        company_data = persona_df[persona_df['company_id'] == company]
        
        # Count occurrences of each persona
        persona_counts = company_data['persona'].value_counts()
        total_records = len(company_data)
        
        # Calculate affinity for each persona
        for persona, count in persona_counts.items():
            # Calculate frequency
            frequency = count / total_records
            
            # Calculate weighted confidence (average confidence for this persona)
            avg_confidence = company_data[company_data['persona'] == persona]['confidence'].mean()
            
            # Calculate recency factor (more weight to recent occurrences)
            recent_data = company_data[company_data['is_recent'] == True]
            recent_count = len(recent_data[recent_data['persona'] == persona])
            total_recent = len(recent_data) if len(recent_data) > 0 else 1
            recency_factor = recent_count / total_recent if total_recent > 0 else 0
            
            # Calculate combined affinity score
            affinity = (frequency * 0.5) + (avg_confidence * 0.3) + (recency_factor * 0.2)
            
            # Get the most recent risk level for this persona
            if len(company_data[company_data['persona'] == persona]) > 0:
                latest_entry = company_data[company_data['persona'] == persona].sort_values('date').iloc[-1]
                risk_level = latest_entry['risk_level']
            else:
                risk_level = 'low'
            
            affinity_scores.append({
                'company_id': company,
                'persona': persona,
                'frequency': frequency,
                'avg_confidence': avg_confidence,
                'recency_factor': recency_factor,
                'affinity_score': affinity,
                'risk_level': risk_level
            })
    
    affinity_df = pd.DataFrame(affinity_scores)
    return affinity_df

def track_persona_transitions(persona_df):
    """
    Track transitions between personas with focus on movements to riskier personas.
    """
    if persona_df.empty:
        return None, None
    
    # Create risk score mapping
    risk_scores = CONFIG['risk']['risk_levels']
    
    # Create a risk score for each persona based on common risk levels
    persona_risk_scores = {}
    for persona in persona_df['persona'].unique():
        risk_levels = persona_df[persona_df['persona'] == persona]['risk_level'].value_counts()
        if not risk_levels.empty:
            most_common_risk = risk_levels.index[0]
            persona_risk_scores[persona] = risk_scores.get(most_common_risk, 0)
        else:
            persona_risk_scores[persona] = 0
    
    # Track transitions between personas
    transitions = []
    risk_increases = []
    
    for company_id, company_data in persona_df.groupby('company_id'):
        company_data = company_data.sort_values('date')
        
        # Skip if only one persona record
        if len(company_data) < 2:
            continue
            
        # Track transitions between consecutive records
        for i in range(len(company_data) - 1):
            from_persona = company_data.iloc[i]['persona']
            to_persona = company_data.iloc[i+1]['persona']
            from_date
