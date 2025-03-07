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
from sklearn.impute import KNNImputer
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Set styles
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_context("talk")
sns.set_palette("viridis")

# Configuration dictionary for easy parameter tuning
CONFIG = {
    'data': {
        'min_nonzero_pct': 0.8,
        'recent_days': 30  # Recent time window to focus on
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
    },
    'visualization': {
        'colors': {
            'cautious_borrower': '#3498db',     # Blue
            'aggressive_expansion': '#e74c3c',  # Red
            'distressed_client': '#e67e22',     # Orange
            'seasonal_operator': '#2ecc71',     # Green
            'deteriorating_health': '#9b59b6',  # Purple
            'cash_constrained': '#f1c40f',      # Yellow
            'credit_dependent': '#1abc9c'       # Teal
        },
        'risk_colors': {
            'high': '#e74c3c',   # Red
            'medium': '#f39c12', # Orange
            'low': '#f1c40f'     # Yellow
        }
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
    
    # Instead of replacing zeros with NaN, keep them for now
    df_calc = df_clean.copy()
    
    # Remove infinite values
    df_calc.replace([np.inf, -np.inf], np.nan, inplace=True)
    
    print(f"Cleaned data shape: {df_clean.shape}")
    print(f"Removed {len(df['company_id'].unique()) - len(valid_companies)} companies")
    
    return df_clean, df_calc

def add_derived_metrics(df):
    """
    Add derived metrics to the dataframe including loan utilization, deposit to loan ratio,
    and rolling metrics for trend analysis. Use advanced imputation for missing values.
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
    
    # Use KNN imputation for company-specific time series data
    print("\nPerforming advanced imputation for missing values...")
    imputed_companies = 0
    
    for company in tqdm(df['company_id'].unique(), desc="Imputing missing values"):
        company_data = df[df['company_id'] == company].sort_values('date')
        
        # Only impute if we have enough data and there are missing values
        if len(company_data) > 30 and (
            company_data['deposit_balance'].isna().any() or 
            company_data['loan_utilization'].isna().any()):
            
            # Create features for imputation (including time-based features)
            impute_df = company_data.copy()
            
            # Add time-based features to help with seasonality
            impute_df['day_of_year'] = impute_df['date'].dt.dayofyear
            impute_df['month'] = impute_df['date'].dt.month
            impute_df['day_of_week'] = impute_df['date'].dt.dayofweek
            
            # Convert date to numeric for imputation
            impute_df['date_numeric'] = (impute_df['date'] - impute_df['date'].min()).dt.days
            
            # Select columns for imputation
            impute_cols = ['date_numeric', 'day_of_year', 'month', 'day_of_week', 
                          'deposit_balance', 'loan_utilization', 'total_loan']
            
            # Try KNN imputation
            try:
                imputer = KNNImputer(n_neighbors=5)
                imputed_values = imputer.fit_transform(impute_df[impute_cols])
                
                # Update original dataframe with imputed values
                impute_df[impute_cols] = imputed_values
                
                # Copy imputed values back to original dataframe
                df.loc[company_data.index, 'deposit_balance'] = impute_df['deposit_balance']
                df.loc[company_data.index, 'loan_utilization'] = impute_df['loan_utilization']
                
                imputed_companies += 1
            except:
                # Fallback to simple forward-backward filling if KNN fails
                df.loc[company_data.index, 'deposit_balance'] = company_data['deposit_balance'].fillna(method='ffill').fillna(method='bfill')
                df.loc[company_data.index, 'loan_utilization'] = company_data['loan_utilization'].fillna(method='ffill').fillna(method='bfill')
    
    print(f"Successfully imputed missing values for {imputed_companies} companies")
    
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
            persona =     #
    # First axis - Loan Utilization
    color = '#1a5276'  # Deep blue
    ax1.set_xlabel('Date', fontsize=14, labelpad=15)
    ax1.set_ylabel('Loan Utilization (%)', fontsize=14, labelpad=15, color=color)
    
    # Plot loan utilization with enhanced styling
    line1 = ax1.plot(company_data['date'], company_data['loan_utilization'] * 100, 
                     color=color, linewidth=2.5, label='Loan Utilization %')
    
    # Add moving average for utilization
    if 'util_ma_30d' in company_data.columns:
        ax1.plot(company_data['date'], company_data['util_ma_30d'] * 100, 
                color='#2980b9', linewidth=1.5, linestyle='--', alpha=0.7,
                label='30-Day Utilization MA')
    
    ax1.tick_params(axis='y', labelcolor=color, labelsize=12)
    ax1.set_ylim(0, 100)
    
    # Second axis - Deposits
    color = '#c0392b'  # Deep red
    ax2 = ax1.twinx()
    ax2.set_ylabel('Deposit Balance', fontsize=14, labelpad=15, color=color)
    
    # Plot deposit balance with enhanced styling
    line2 = ax2.plot(company_data['date'], company_data['deposit_balance'], 
                     color=color, linewidth=2.5, label='Deposit Balance')
    
    # Add moving average for deposits
    if 'deposit_ma_30d' in company_data.columns:
        ax2.plot(company_data['date'], company_data['deposit_ma_30d'], 
                color='#e74c3c', linewidth=1.5, linestyle='--', alpha=0.7,
                label='30-Day Deposit MA')
    
    ax2.tick_params(axis='y', labelcolor=color, labelsize=12)
    
    # Format dates on x-axis
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    ax1.xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))
    plt.xticks(rotation=45, ha='right', fontsize=12)
    
    # Highlight the recent period with a shaded area
    ax1.axvspan(recent_start, latest_date, alpha=0.2, color='#f1c40f', label='Recent Period')
    
    # Add risk markers
    if not recent_risks.empty:
        risk_dates = recent_risks['date'].tolist()
        risk_descriptions = recent_risks['risk_description'].tolist()
        risk_levels = recent_risks['risk_level'].tolist()
        personas = recent_risks['persona'].tolist()
        
        # Analyze risk flags to find the most common one
        all_flags = []
        for flags in recent_risks['risk_flags']:
            all_flags.extend(flags.split('|'))
        
        most_common_flag = pd.Series(all_flags).value_counts().index[0] if all_flags else "No flags"
        
        # Create color map for risk levels
        risk_colors = {
            'high': CONFIG['visualization']['risk_colors']['high'],
            'medium': CONFIG['visualization']['risk_colors']['medium'],
            'low': CONFIG['visualization']['risk_colors']['low']
        }
        
        # Add vertical lines for each risk event
        for i, (date, desc, level, persona) in enumerate(zip(risk_dates, risk_descriptions, risk_levels, personas)):
            # Limit to plotting only a reasonable number of risk markers
            if i < 8:  # Only show first 8 risk events to avoid cluttering
                ax1.axvline(x=date, color=risk_colors[level], linestyle='--', alpha=0.8, linewidth=1.5)
                
                # Add descriptions (shortened if too long)
                short_desc = desc.split('|')[0] if len(desc) > 50 else desc
                y_pos = 90 - (i % 4) * 20  # Stagger text vertically
                
                # Enhanced annotation with persona
                ax1.annotate(
                    f"{short_desc}\nPersona: {persona}",
                    xy=(date, y_pos),
                    xytext=(10, 0),
                    textcoords="offset points",
                    color=risk_colors[level],
                    rotation=0,
                    ha='left',
                    va='center',
                    fontsize=9,
                    bbox=dict(
                        boxstyle="round,pad=0.5",
                        fc="white",
                        ec=risk_colors[level],
                        alpha=0.9
                    )
                )
    
    # Add title and company details
    persona_text = ""
    if not recent_risks.empty:
        # Get most recent persona
        recent_persona = recent_risks.iloc[-1]['persona']
        persona_desc = CONFIG['risk']['persona_patterns'].get(recent_persona, "Unknown")
        persona_text = f"\nCurrent Persona: {recent_persona} - {persona_desc}"
    
    plt.title(f"Risk Analysis for {company_id} - Recent {recent_days} Days{persona_text}", 
              fontsize=16, fontweight='bold', pad=20)
    
    # Add key metrics in an enhanced text box
    avg_util = recent_company_data['loan_utilization'].mean() * 100
    avg_deposit = recent_company_data['deposit_balance'].mean()
    deposit_trend = (recent_company_data['deposit_balance'].iloc[-1] / recent_company_data['deposit_balance'].iloc[0] - 1) * 100 if len(recent_company_data) > 1 else 0
    util_trend = (recent_company_data['loan_utilization'].iloc[-1] - recent_company_data['loan_utilization'].iloc[0]) * 100 if len(recent_company_data) > 1 else 0
    risk_count = len(recent_risks)
    
    # Determine color for trends
    deposit_trend_color = '#27ae60' if deposit_trend >= 0 else '#c0392b'
    util_trend_color = '#c0392b' if util_trend >= 0 else '#27ae60'
    
    metrics_text = (
        f"Recent {recent_days}-Day Metrics:\n"
        f"Avg Utilization: {avg_util:.1f}%  "
        f"({util_trend:+.1f}pp)\n"
        f"Avg Deposit: ${avg_deposit:,.2f}  "
        f"({deposit_trend:+.1f}%)\n"
        f"Risk Events: {risk_count}"
    )
    
    # Add prominent text box with metrics
    props = dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.9, edgecolor='#bbbbbb')
    metrics_box = ax1.text(
        0.02, 0.05,
        metrics_text,
        transform=ax1.transAxes,
        fontsize=12,
        verticalalignment='bottom',
        bbox=props
    )
    
    # Add most common risk pattern in a highlighted box if risks exist
    if not recent_risks.empty and 'most_common_flag' in locals():
        risk_box_text = f"Most Common Risk Pattern: {most_common_flag.replace('_', ' ').title()}"
        risk_box = ax1.text(
            0.98, 0.05,
            risk_box_text,
            transform=ax1.transAxes,
            fontsize=12,
            color='white',
            horizontalalignment='right',
            verticalalignment='bottom',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='#d35400', alpha=0.9)
        )
    
    # Combine legends from both axes
    lines = line1 + line2
    if 'util_ma_30d' in company_data.columns and 'deposit_ma_30d' in company_data.columns:
        # Add MA lines to legend if they exist
        lines += [plt.Line2D([0], [0], color='#2980b9', linestyle='--', linewidth=1.5, alpha=0.7)]
        lines += [plt.Line2D([0], [0], color='#e74c3c', linestyle='--', linewidth=1.5, alpha=0.7)]
    
    # Add legend for highlighted area
    lines += [plt.Rectangle((0, 0), 1, 1, fc='#f1c40f', alpha=0.2)]
    
    labels = [l.get_label() for l in lines[:2]]
    if 'util_ma_30d' in company_data.columns and 'deposit_ma_30d' in company_data.columns:
        labels += ['30-Day Util. MA', '30-Day Deposit MA']
    labels += ['Recent Period']
    
    # Enhanced legend
    ax1.legend(
        lines, labels,
        loc='upper center',
        bbox_to_anchor=(0.5, -0.15),
        ncol=min(5, len(lines)),
        fontsize=12,
        frameon=True,
        facecolor='white',
        edgecolor='#dddddd'
    )
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.2)  # Make room for legend
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
    plt.figure(figsize=(14, 8))
    
    # Get custom colors for each persona
    colors = []
    for persona in persona_counts['persona']:
        color = CONFIG['visualization']['colors'].get(persona, '#7f8c8d')  # Default gray if persona not found
        colors.append(color)
    
    # Create enhanced bar plot
    ax = sns.barplot(x='persona', y='count', data=persona_counts, palette=colors)
    
    # Add percentage labels
    total = persona_counts['count'].sum()
    for i, p in enumerate(ax.patches):
        percentage = 100 * p.get_height() / total
        ax.annotate(
            f"{percentage:.1f}%",
            (p.get_x() + p.get_width() / 2., p.get_height()),
            ha='center',
            va='bottom',
            fontsize=12,
            fontweight='bold',
            color='#2c3e50'
        )
    
    # Add counts on top of bars
    for i, p in enumerate(ax.patches):
        height = p.get_height()
        ax.annotate(
            f"{int(height)}",
            (p.get_x() + p.get_width() / 2., height),
            ha='center',
            va='bottom',
            fontsize=12,
            xytext=(0, 5),
            textcoords='offset points'
        )
    
    # Add titles and labels
    plt.title('Distribution of Client Personas', fontsize=18, fontweight='bold', pad=20)
    plt.xlabel('Persona', fontsize=14, labelpad=15)
    plt.ylabel('Number of Risk Events', fontsize=14, labelpad=15)
    plt.xticks(rotation=30, ha='right', fontsize=12)
    
    # Add persona descriptions as a table
    persona_desc = pd.DataFrame(list(CONFIG['risk']['persona_patterns'].items()), 
                                columns=['Persona', 'Description'])
    
    # Create a styled table at the bottom
    table = plt.table(
        cellText=persona_desc.values,
        colLabels=persona_desc.columns,
        loc='bottom',
        cellLoc='center',
        bbox=[0, -0.45, 1, 0.25]
    )
    
    # Style the table
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 1.5)
    
    # Style the cells
    for i, key in enumerate(table._cells):
        cell = table._cells[key]
        if key[0] == 0:  # Header
            cell.set_text_props(fontweight='bold', color='white')
            cell.set_facecolor('#34495e')
        else:  # Data rows
            if key[1] == 0:  # Persona column
                cell.set_text_props(fontweight='bold')
                color = CONFIG['visualization']['colors'].get(cell.get_text().get_text(), '#7f8c8d')
                cell.set_facecolor(f"{color}33")  # Add alpha transparency
    
    plt.subplots_adjust(bottom=0.35)
    plt.tight_layout()
    
    return plt.gcf()

def plot_persona_transitions(quarterly_persona_df):
    """
    Plot Sankey diagram or other visualization showing transitions between personas over time.
    Enhanced with better styling and clearer visualization.
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
    
    # Sort personas by frequency for better visualization
    all_personas = pd.Series(
        list(pivot_transitions.index) + list(pivot_transitions.columns)
    ).value_counts().index.tolist()
    
    sorted_index = [p for p in all_personas if p in pivot_transitions.index]
    sorted_columns = [p for p in all_personas if p in pivot_transitions.columns]
    
    pivot_transitions = pivot_transitions.reindex(index=sorted_index, columns=sorted_columns)
    
    # Create enhanced heatmap
    plt.figure(figsize=(14, 12))
    
    # Calculate percentages for annotations
    row_sums = pivot_transitions.sum(axis=1)
    percentage_df = pivot_transitions.div(row_sums, axis=0).fillna(0) * 100
    
    # Create annotation texts with both count and percentage
    annot_texts = pivot_transitions.applymap(str)
    for i in range(len(pivot_transitions.index)):
        for j in range(len(pivot_transitions.columns)):
            from_persona = pivot_transitions.index[i]
            to_persona = pivot_transitions.columns[j]
            count = pivot_transitions.iloc[i, j]
            
            if count > 0:
                pct = percentage_df.iloc[i, j]
                annot_texts.iloc[i, j] = f"{int(count)}\n({pct:.1f}%)"
            else:
                annot_texts.iloc[i, j] = ""
    
    # Use custom colors for better visualization
    persona_colors = {p: CONFIG['visualization']['colors'].get(p, '#7f8c8d') for p in all_personas}
    
    # Create custom colormap for the heatmap
    cmap = sns.cubehelix_palette(
        start=.5, rot=-.75, 
        as_cmap=True, 
        light=0.95, 
        dark=0.4
    )
    
    # Create the heatmap with enhanced styling
    ax = sns.heatmap(
        pivot_transitions, 
        annot=annot_texts,
        fmt="",
        cmap=cmap,
        linewidths=0.5,
        linecolor="#dddddd",
        cbar_kws={'label': 'Number of Transitions'}
    )
    
    # Style the chart
    plt.title('Client Persona Transitions Between Quarters', fontsize=18, fontweight='bold', pad=20)
    plt.xlabel('To Persona', fontsize=14, labelpad=15)
    plt.ylabel('From Persona', fontsize=14, labelpad=15)
    
    # Enhance tick labels
    plt.xticks(rotation=45, ha='right', fontsize=12)
    plt.yticks(rotation=0, fontsize=12)
    
    # Add diagonal highlighting for stable personas
    for i in range(min(len(pivot_transitions.index), len(pivot_transitions.columns))):
        if pivot_transitions.index[i] == pivot_transitions.columns[i]:
            # Highlight cells where personas don't change
            ax.add_patch(plt.Rectangle(
                (i, i), 
                1, 1, 
                fill=False, 
                edgecolor='green', 
                linestyle='-', 
                linewidth=2
            ))
    
    # Add summary statistics in a text box
    total_transitions = pivot_transitions.sum().sum()
    diagonal_sum = sum(pivot_transitions.iloc[i, i] for i in range(min(len(pivot_transitions.index), len(pivot_transitions.columns))) 
                      if pivot_transitions.index[i] == pivot_transitions.columns[i])
    stable_percentage = (diagonal_sum / total_transitions) * 100 if total_transitions > 0 else 0
    
    most_common_transition = transition_counts.loc[transition_counts['count'].idxmax()]
    most_common_pct = (most_common_transition['count'] / total_transitions) * 100
    
    summary_text = (
        f"Total Transitions: {int(total_transitions)}\n"
        f"Stable Personas: {stable_percentage:.1f}%\n"
        f"Most Common: {most_common_transition['from_persona']} → {most_common_transition['to_persona']} "
        f"({most_common_transition['count']} times, {most_common_pct:.1f}%)"
    )
    
    # Add summary text box
    plt.annotate(
        summary_text,
        xy=(0.02, 0.02),
        xycoords='figure fraction',
        bbox=dict(boxstyle="round,pad=0.5", facecolor='white', alpha=0.8, edgecolor='#cccccc'),
        fontsize=12
    )
    
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
    Visualize clusters in PCA space with enhanced styling
    """
    if feature_df is None or 'pca_1' not in feature_df.columns:
        print("No clustering data available for visualization.")
        return None
    
    plt.figure(figsize=(14, 10))
    
    # Get cluster information
    clusters = feature_df['cluster'].unique()
    cluster_counts = feature_df['cluster'].value_counts().to_dict()
    
    # Create a custom color palette
    colors = sns.color_palette("viridis", len(clusters))
    
    # Create scatter plot of first two PCA components with enhanced styling
    for i, cluster in enumerate(sorted(clusters)):
        cluster_data = feature_df[feature_df['cluster'] == cluster]
        
        plt.scatter(
            x=cluster_data['pca_1'],
            y=cluster_data['pca_2'],
            s=100,
            alpha=0.8,
            c=[colors[i]],
            label=f"Cluster {cluster} (n={cluster_counts.get(cluster, 0)})"
        )
    
    # Add centroids
    centroids = feature_df.groupby('cluster')[['pca_1', 'pca_2']].mean().reset_index()
    for i, row in centroids.iterrows():
        plt.scatter(
            x=row['pca_1'],
            y=row['pca_2'],
            s=200,
            marker='X',
            c=[colors[i]],
            edgecolor='black',
            linewidth=1.5
        )
        
        # Label centroid
        plt.annotate(
            f"C{int(row['cluster'])}",
            (row['pca_1'], row['pca_2']),
            fontsize=12,
            fontweight='bold',
            ha='center',
            va='center',
            color='white',
            xytext=(0, 0),
            textcoords="offset points"
        )
    
    # Add labels and styling
    plt.title('Client Clusters in Principal Component Space', fontsize=18, fontweight='bold', pad=20)
    plt.xlabel('Principal Component 1', fontsize=14, labelpad=15)
    plt.ylabel('Principal Component 2', fontsize=14, labelpad=15)
    
    # Add gridlines
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Enhanced legend
    plt.legend(
        title="Cluster Information",
        title_fontsize=14,
        fontsize=12,
        loc='upper center',
        bbox_to_anchor=(0.5, -0.08),
        ncol=min(3, len(clusters)),
        frameon=True,
        facecolor='white',
        edgecolor='#dddddd'
    )
    
    # Add explanatory annotations
    plt.annotate(
        "Each point represents a company\nPositioned based on its financial behavior pattern",
        xy=(0.02, 0.02),
        xycoords='figure fraction',
        bbox=dict(boxstyle="round,pad=0.5", facecolor='white', alpha=0.8, edgecolor='#cccccc'),
        fontsize=12
    )
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15)  # Make room for legend
    
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
    
    # Plot enhanced heatmap
    plt.figure(figsize=(14, 10))
    
    # Create percentage heatmap (normalize by row)
    contingency_pct = contingency.iloc[:-1, :-1].div(contingency.iloc[:-1, -1], axis=0) * 100
    
    # Create custom annotations with both count and percentage
    annot_labels = np.empty_like(contingency.iloc[:-1, :-1], dtype=object)
    for i in range(contingency.iloc[:-1, :-1].shape[0]):
        for j in range(contingency.iloc[:-1, :-1].shape[1]):
            count = contingency.iloc[i, j]
            percentage = contingency_pct.iloc[i, j]
            if count > 0:
                annot_labels[i, j] = f"{count}\n({percentage:.1f}%)"
            else:
                annot_labels[i, j] = ""
    
    # Use a sequential color palette for the heatmap
    cmap = sns.cubehelix_palette(
        start=.5, rot=-.75, 
        as_cmap=True, 
        light=0.95, 
        dark=0.4
    )
    
    # Plot the heatmap with enhanced styling
    ax = sns.heatmap(
        contingency.iloc[:-1, :-1],  # Exclude margins
        annot=annot_labels,
        fmt="",
        cmap=cmap,
        linewidths=0.5,
        linecolor="#dddddd",
        cbar_kws={'label': 'Number of Companies'}
    )
    
    # Style the chart
    plt.title('Comparison of Rule-Based Personas vs. Unsupervised Clusters', 
              fontsize=18, fontweight='bold', pad=20)
    plt.xlabel('Cluster ID', fontsize=14, labelpad=15)
    plt.ylabel('Persona', fontsize=14, labelpad=15)
    
    # Enhance tick labels
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    
    # Add summary text
    n_companies = contingency.iloc[-1, -1]
    highest_overlap = contingency.iloc[:-1, :-1].max().max()
    highest_row, highest_col = np.unravel_index(contingency.iloc[:-1, :-1].values.argmax(), contingency.iloc[:-1, :-1].shape)
    highest_persona = contingency.index[highest_row]
    highest_cluster = contingency.columns[highest_col]
    
    summary_text = (
        f"Total companies: {n_companies}\n"
        f"Strongest alignment: {highest_persona} with Cluster {highest_cluster}\n"
        f"({highest_overlap} companies, {(highest_overlap/n_companies*100):.1f}% of total)"
    )
    
    # Add summary text box
    plt.annotate(
        summary_text,
        xy=(0.02, 0.02),
        xycoords='figure fraction',
        bbox=dict(boxstyle="round,pad=0.5", facecolor='white', alpha=0.8, edgecolor='#cccccc'),
        fontsize=12
    )
    
    plt.tight_layout()
    return plt.gcf()

def main(df):
    """
    Main function to execute the entire analysis workflow with improved risk detection
    and persona-based cohort analysis. Focus on recent risk patterns and enhance visualizations.
    """
    print("Starting enhanced bank client risk analysis...")
    
    # 1. Clean data
    print("\nCleaning data...")
    df_clean, df_calc = clean_data(df, min_nonzero_pct=CONFIG['data']['min_nonzero_pct'])
    
    # 2. Add derived metrics with enhanced features and improved imputation
    print("\nAdding derived metrics and calculating trends...")
    df_with_metrics = add_derived_metrics(df_clean)
    
    # 3. Detect risk patterns with efficient implementation and persona assignment
    print("\nDetecting risk patterns and assigning personas...")
    risk_df, persona_df = detect_risk_patterns_efficient(df_with_metrics)
    
    if not risk_df.empty:
        print(f"Found {len(risk_df)} risk events across {risk_df['company_id'].nunique()} companies")
        print(f"Assigned {persona_df['persona'].nunique() if not persona_df.empty else 0} different personas")
        
        # 4. Get recent risk clients (last 30 days)
        print(f"\nIdentifying clients with risks in the last {CONFIG['data']['recent_days']} days...")
        recent_risk_clients, latest_date = get_recent_risk_clients(risk_df, recent_days=CONFIG['data']['recent_days'])
        
        if not recent_risk_clients.empty:
            print(f"Found {len(recent_risk_clients)} companies with recent risk events")
            print(f"Most common risk flags: {recent_risk_clients['most_common_risk_flag'].value_counts().head(3)}")
        else:
            print(f"No companies with risk events in the last {CONFIG['data']['recent_days']} days")
    else:
        print("No risk events detected")
        recent_risk_clients = pd.DataFrame()
        latest_date = None
    
    # 5. Perform clustering analysis
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
    
    # 6. Create persona-based cohort analysis
    print("\nCreating persona-based cohort analysis...")
    cohort_results = create_personas_cohort_analysis(persona_df)
    if cohort_results[0] is not None:
        cohort_data, risk_cohort_data, quarterly_persona_df = cohort_results
        
        # 7. Plot persona cohort analysis with improved visualization
        print("\nPlotting persona cohort analysis...")
        cohort_fig = plot_persona_cohort_improved(cohort_data)
        if cohort_fig:
            plt.savefig('persona_cohort_analysis.png')
            print("Saved persona cohort analysis to persona_cohort_analysis.png")
        
        # 8. Plot persona distribution
        print("\nPlotting persona distribution...")
        persona_dist_fig = plot_persona_distribution(persona_df)
        if persona_dist_fig:
            plt.savefig('persona_distribution.png')
            print("Saved persona distribution to persona_distribution.png")
        
        # 9. Plot persona transitions
        print("\nPlotting persona transitions...")
        transition_fig = plot_persona_transitions(quarterly_persona_df)
        if transition_fig:
            plt.savefig('persona_transitions.png')
            print("Saved persona transitions to persona_transitions.png")
    else:
        quarterly_persona_df = None
    
    # 10. Plot top risky companies from recent period with enhanced visualization
    print(f"\nPlotting top recent risky companies (last {CONFIG['data']['recent_days']} days)...")
    if not recent_risk_clients.empty:
        # Sort by risk level and risk count
        top_risky_companies = recent_risk_clients.sort_values(
            by=['max_risk_level', 'risk_count'], 
            ascending=[False, False]
        ).head(5)['company_id'].tolist()
        
        for company_id in top_risky_companies:
            print(f"Plotting risk analysis for {company_id}...")
            company_fig = plot_risk_company_recent(
                company_id, 
                df_with_metrics, 
                risk_df, 
                recent_days=CONFIG['data']['recent_days']
            )
            if company_fig:
                company_fig.savefig(f'recent_risk_{company_id}.png')
                print(f"Saved company risk analysis to recent_risk_{company_id}.png")
    else:
        print("No recent risky companies to plot")
    
    print("\nAnalysis complete! All visualization files saved.")
    
    return {
        'data': df_with_metrics,
        'risk_df': risk_df,
        'persona_df': persona_df,
        'recent_risk_clients': recent_risk_clients,
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

def get_recent_risk_clients(risk_df, recent_days=30):
    """
    Extract clients with risk events in the most recent period.
    Focus only on the latest risk data as historical risk is less important.
    
    Returns:
    - DataFrame with unique clients and their most frequent risk flags
    - The latest date in the dataset
    """
    if risk_df.empty:
        return pd.DataFrame(), None
    
    # Get the most recent date in the dataset
    latest_date = risk_df['date'].max()
    
    # Define the recent period
    recent_start = latest_date - pd.Timedelta(days=recent_days)
    
    # Filter for recent risk events
    recent_risks = risk_df[risk_df['date'] >= recent_start].copy()
    
    if recent_risks.empty:
        return pd.DataFrame(), latest_date
    
    # Find the most frequent risk flag for each company
    company_summary = []
    for company_id, group in recent_risks.groupby('company_id'):
        # Get all risk flags
        all_flags = []
        for flags in group['risk_flags']:
            all_flags.extend(flags.split('|'))
        
        # Count occurrences of each flag
        flag_counts = pd.Series(all_flags).value_counts()
        
        # Get the most common flag
        if not flag_counts.empty:
            most_common_flag = flag_counts.index[0]
            flag_count = flag_counts.iloc[0]
        else:
            most_common_flag = "unknown"
            flag_count = 0
        
        # Get the most severe risk level
        risk_levels = group['risk_level'].values
        if 'high' in risk_levels:
            max_risk = 'high'
        elif 'medium' in risk_levels:
            max_risk = 'medium'
        else:
            max_risk = 'low'
        
        # Get the latest persona
        latest_persona = group.loc[group['date'].idxmax(), 'persona']
        
        # Calculate average metrics
        avg_util = group['current_util'].mean()
        avg_deposit = group['current_deposit'].mean()
        
        company_summary.append({
            'company_id': company_id,
            'most_common_risk_flag': most_common_flag,
            'flag_occurrences': flag_count,
            'max_risk_level': max_risk,
            'latest_persona': latest_persona,
            'avg_utilization': avg_util,
            'avg_deposit': avg_deposit,
            'risk_count': len(group)
        })
    
    return pd.DataFrame(company_summary), latest_date

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

def plot_persona_cohort_improved(cohort_data):
    """
    Enhanced visually appealing plot for persona-based cohort analysis 
    showing how personas evolve over time.
    """
    if cohort_data is None or cohort_data.empty:
        print("No cohort data available for plotting.")
        return None
    
    # Create figure with better aesthetics
    plt.figure(figsize=(16, 10))
    
    # Get custom colors for each persona
    persona_colors = []
    for persona in cohort_data.columns:
        color = CONFIG['visualization']['colors'].get(persona, '#7f8c8d')  # Default gray if persona not found
        persona_colors.append(color)
    
    # Plot area chart with enhanced styling
    ax = cohort_data.plot(
        kind='area', 
        stacked=True, 
        alpha=0.85, 
        color=persona_colors,
        linewidth=2
    )
    
    # Enhance grid
    ax.grid(True, linestyle='--', alpha=0.7, axis='y')
    
    # Custom background
    ax.set_facecolor('#f9f9f9')
    
    # Formatting
    plt.title('Evolution of Client Personas Over Time', fontsize=20, fontweight='bold', pad=20)
    plt.xlabel('Quarter', fontsize=16, labelpad=15)
    plt.ylabel('Number of Companies', fontsize=16, labelpad=15)
    
    # Enhance legend
    legend = plt.legend(
        title='Client Personas', 
        title_fontsize=14,
        fontsize=12,
        loc='upper center',
        bbox_to_anchor=(0.5, -0.15),
        ncol=min(4, len(cohort_data.columns)),
        frameon=True,
        facecolor='white',
        edgecolor='#dddddd'
    )
    
    # Add total companies count on top of each stack
    for i, quarter in enumerate(cohort_data.index):
        total = cohort_data.loc[quarter].sum()
        plt.text(
            i, 
            total + (total * 0.03),  # Slightly above the stack
            f"{int(total)}",
            ha='center',
            va='bottom',
            fontsize=12,
            fontweight='bold',
            color='#2c3e50'
        )
    
    # Add percentage breakdown inside stacks
    y_offset = np.zeros(len(cohort_data.index))
    
    for col in cohort_data.columns:
        for i, (idx, val) in enumerate(cohort_data[col].items()):
            total = cohort_data.loc[idx].sum()
            if total > 0:
                percentage = val / total * 100
                # Only label if percentage is significant
                if percentage > 8:
                    plt.text(
                        i, 
                        y_offset[i] + val/2,  # Middle of the segment
                        f"{percentage:.0f}%",
                        ha='center',
                        va='center',
                        fontsize=11,
                        fontweight='bold',
                        color='white'
                    )
            y_offset[i] += val
    
    # Add stylish borders
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color('#dddddd')
        spine.set_linewidth(1.5)
    
    # Customize x-axis ticks
    plt.xticks(rotation=45, ha='right', fontsize=12)
    plt.yticks(fontsize=12)
    
    # Add subtle gradient background with alpha transparency
    gradient = np.linspace(0, 1, 100).reshape(-1, 1)
    gradient = np.repeat(gradient, 100, axis=1)
    extent = [ax.get_xlim()[0], ax.get_xlim()[1], ax.get_ylim()[0], ax.get_ylim()[1]]
    
    # Add annotations for key trends
    trends = []
    start_vals = cohort_data.iloc[0] if not cohort_data.empty else pd.Series()
    end_vals = cohort_data.iloc[-1] if len(cohort_data) > 1 else pd.Series()
    
    for col in cohort_data.columns:
        if col in start_vals and col in end_vals and start_vals[col] > 0 and end_vals[col] > 0:
            pct_change = (end_vals[col] - start_vals[col]) / start_vals[col] * 100
            if abs(pct_change) > 25:  # Only highlight significant changes
                trends.append({
                    'persona': col,
                    'change': pct_change,
                    'text': f"{col}: {'↑' if pct_change > 0 else '↓'}{abs(pct_change):.0f}%"
                })
    
    # Add trend annotations if we have any
    if trends:
        annotation_text = "Key Trends:\n" + "\n".join([t['text'] for t in trends])
        
        # Add an annotation box
        plt.annotate(
            annotation_text,
            xy=(0.02, 0.97),
            xycoords='axes fraction',
            backgroundcolor='white',
            bbox=dict(boxstyle="round,pad=0.5", facecolor='white', alpha=0.8, edgecolor='#cccccc'),
            fontsize=11,
            verticalalignment='top'
        )
    
    plt.tight_layout()
    return ax

def plot_risk_company_recent(company_id, df, risk_df, recent_days=30):
    """
    Create an enhanced plot of a company's recent risk patterns.
    Shows loan utilization and deposit balance with risk annotations,
    focusing on the most recent period and highlighting the most common risk pattern.
    """
    # Filter data for company
    company_data = df[df['company_id'] == company_id].sort_values('date')
    
    if company_data.empty:
        print(f"No data found for company {company_id}")
        return None
    
    # Get the most recent date in the dataset
    latest_date = company_data['date'].max()
    
    # Define the recent period
    recent_start = latest_date - pd.Timedelta(days=recent_days)
    
    # Get recent company data
    recent_company_data = company_data[company_data['date'] >= recent_start].copy()
    
    # Get all risk events for company, including historical (for comparison)
    company_risks = risk_df[risk_df['company_id'] == company_id].sort_values('date')
    
    # Get recent risks
    recent_risks = company_risks[company_risks['date'] >= recent_start].copy()
    
    # Create figure with two y-axes
    fig, ax1 = plt.subplots(figsize=(16, 9))
    
    # Use a modern style
    plt.style.use('seaborn-v0_8-whitegrid')
    ax1.set_facecolor('#f9f9f9')
    
    # First axis - Loan Utilization
    color = '#1a5276'  # Deep blue
    ax1.set_xlabel('Date', fontsize=14, labelpad=15)
    ax1.set_ylabel('Loan Utilization (%)', fontsize=14, labelpad=15, color=color)
    
    # Plot loan utilization with enhanced styling
    line1 = ax1.plot(company_data['date'], company_data['loan_utilization'] * 100, 
                     color=color, linewidth=2.5, label='Loan Utilization %')
    
    # Add moving average for utilization
    if 'util_ma_30d' in company_data.columns:
        ax1.plot(company_data['date'], company_data['util_ma_30d'] * 100, 
                color='#2980b9', linewidth=1.5, linestyle='--', alpha=0.7,
                label='30-Day Utilization MA')
    
    ax1.tick_params(axis='y', labelcolor=color, labelsize=12)
    ax1.set_ylim(0, 100)
    
    # Second axis - Deposits
    color = '#c0392b'  # Deep red
    ax2 = ax1.twinx()
    ax2.set_ylabel('Deposit Balance', fontsize=14, labelpad=15, color=color)
    
    # Plot deposit balance with enhanced styling
    line2 = ax2.plot(company_data['date'], company_data['deposit_balance'], 
                     color=color, linewidth=2.5, label='Deposit Balance')
    
    # Add moving average for deposits
    if 'deposit_ma_30d' in company_data.columns:
        ax2.plot(company_data['date'], company_data['deposit_ma_30d'], 
                color='#e74c3c', linewidth=1.5, linestyle='--', alpha=0.7,
                label='30-Day Deposit MA')
    
    ax2.tick_params(axis='y', labelcolor=color, labelsize=12)
    
    # Format dates on x-axis
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    ax1.xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))
    plt.xticks(rotation=45, ha='right', fontsize=12)
    
    # Highlight the recent period with a shaded area
    ax1.axvspan(recent_start, latest_date, alpha=0.2, color='#f1c40f', label='Recent Period')
    
    # Add risk markers
    if not recent_risks.empty:
        risk_dates = recent_risks['date'].tolist()
        risk_descriptions = recent_risks['risk_description'].tolist()
        risk_levels = recent_risks['risk_level'].tolist()
        personas = recent_risks['persona'].tolist()
        
        # Analyze risk flags to find the most common one
        all_flags = []
        for flags in recent_risks['risk_flags']:
            all_flags.extend(flags.split('|'))
        
        most_common_flag = pd.Series(all_flags).value_counts().index[0] if all_flags else "No flags"
        
        # Create color map for risk levels
        risk_colors = {
            'high': CONFIG['visualization']['risk_colors']['high'],
            'medium': CONFIG['visualization']['risk_colors']['medium'],
            'low': CONFIG['visualization']['risk_colors']['low']
        }
        
        # Add vertical lines for each risk event
        for i, (date, desc, level, persona) in enumerate(zip(risk_dates, risk_descriptions, risk_levels, personas)):
            # Limit to plotting only a reasonable number of risk markers
            if i < 8:  # Only show first 8 risk events to avoid cluttering
                ax1.axvline(x=date, color=risk_colors[level], linestyle='--', alpha=0.8, linewidth=1.5)
                
                # Add descriptions (shortened if too long)
                short_desc = desc.split('|')[0] if len(desc) > 50 else desc
                y_pos = 90 - (i % 4) * 20  # Stagger text vertically
                
                # Enhanced annotation with persona
                ax1.annotate(
                    f"{short_desc}\nPersona: {persona}",
                    xy=(date, y_pos),
                    xytext=(10, 0),
                    textcoords="offset points",
                    color=risk_colors[level],
                    rotation=0,
                    ha='left',
                    va='center',
                    fontsize=9,
                    bbox=dict(
                        boxstyle="round,pad=0.5",
                        fc="white",
                        ec=risk_colors[level],
                        alpha=0.9
                    )
                )
    
    # Add title and company details
    persona_text = ""
    if not recent_risks.empty:
        # Get most recent persona
        recent_persona = recent_risks.iloc[-1]['persona']
        persona_desc = CONFIG['risk']['persona_patterns'].get(recent_persona, "Unknown")
        persona_text = f"\nCurrent Persona: {recent_persona} - {persona_desc}"
    
    plt.title(f"Risk Analysis for {company_id} - Recent {recent_days} Days{persona_text}", 
              fontsize=16, fontweight='bold', pad=20)
    
    # Add key metrics in an enhanced text box
    avg_util = recent_company_data['loan_utilization'].mean() * 100
    avg_deposit = recent_company_data['deposit_balance'].mean()
    deposit_trend = (recent_company_data['deposit_balance'].iloc[-1] / recent_company_data['deposit_balance'].iloc[0] - 1) * 100 if len(recent_company_data) > 1 else 0
    util_trend = (recent_company_data['loan_utilization'].iloc[-1] - recent_company_data['loan_utilization'].iloc[0]) * 100 if len(recent_company_data) > 1 else 0
    risk_count = len(recent_risks)
    
    # Determine color for trends
    deposit_trend_color = '#27ae60' if deposit_trend >= 0 else '#c0392b'
    util_trend_color = '#c0392b' if util_trend >= 0 else '#27ae60'
    
    metrics_text = (
        f"Recent {recent_days}-Day Metrics:\n"
        f"Avg Utilization: {avg_util:.1f}%  "
        f"({util_trend:+.1f}pp)\n"
        f"Avg Deposit: ${avg_deposit:,.2f}  "
        f"({deposit_trend:+.1f}%)\n"
        f"Risk Events: {risk_count}"
    )
    
    # Add prominent text box with metrics
    props = dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.9, edgecolor='#bbbbbb')
    metrics_box = ax1.text(
        0.02, 0.05,
        metrics_text,
        transform=ax1.transAxes,
        fontsize=12,
        verticalalignment='bottom',
        bbox=props
    )
    
    # Add most common risk pattern in a highlighted box if risks exist
    if not recent_risks.empty and 'most_common_flag' in locals():
        risk_box_text = f"Most Common Risk Pattern: {most_common_flag.replace('_', ' ').title()}"
        risk_box = ax1.text(
            0.98, 0.05,
            risk_box_text,
            transform=ax1.transAxes,
            fontsize=12,
            color='white',
            horizontalalignment='right',
            verticalalignment='bottom',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='#d35400', alpha=0.9)
        )
    
    # Combine legends from both axes
    lines = line1 + line2
    if 'util_ma_30d' in company_data.columns and 'deposit_ma_30d' in company_data.columns:
        # Add MA lines to legend if they exist
        lines += [plt.Line2D([0], [0], color='#2980b9', linestyle='--', linewidth=1.5, alpha=0.7)]
        lines += [plt.Line2D([0], [0], color='#e74c3c', linestyle='--', linewidth=1.5, alpha=0.7)]
    
    # Add legend for highlighted area
    lines += [plt.Rectangle((0, 0), 1, 1, fc='#f1c40f', alpha=0.2)]
    
    labels = [l.get_label() for l in lines[:2]]
    if 'util_ma_30d' in company_data.columns and 'deposit_ma_30d' in company_data.columns:
        labels += ['30-Day Util. MA', '30-Day Deposit MA']
    labels += ['Recent Period']
    
    # Enhanced legend
    ax1.legend(
        lines, labels,
        loc='upper center',
        bbox_to_anchor=(0.5, -0.15),
        ncol=min(5, len(lines)),
        fontsize=12,
        frameon=True,
        facecolor='white',
        edgecolor='#dddddd'
    )
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.2)  # Make room for legend
    return fig
