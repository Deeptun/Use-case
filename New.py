import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import re

def visualize_company_data(df):
    """
    Visualizes daily revenue, deposit balance, and risk grades over time for each company.
    Fills missing dates using interpolation and plots three y-axes for each metric.
    
    Parameters:
    df (DataFrame): Input dataframe with columns ['date', 'company_code_id', 'revenue_amount', 
                      'daily_deposit_amount', 'risk_grade']
    """
    
    # Convert risk grades to numerical values (e.g., '2+' -> 2.2, '3-' -> 2.8)
    def grade_to_num(grade):
        if pd.isna(grade):
            return np.nan
        if isinstance(grade, (int, float)):
            return float(grade)
        
        base_match = re.match(r'^(\d+)', str(grade))
        if not base_match:
            return np.nan
        
        base = float(base_match.group(1))
        modifier = str(grade)[len(base_match.group(1)):]
        
        if modifier == '+':
            return base + 0.2
        elif modifier == '-':
            return base - 0.2
        else:
            return base
    
    df['risk_num'] = df['risk_grade'].apply(grade_to_num)
    
    # Process each company individually
    for company in df['company_code_id'].unique():
        company_df = df[df['company_code_id'] == company].copy()
        
        # Convert dates and set as index
        company_df['date'] = pd.to_datetime(company_df['date'])
        company_df.set_index('date', inplace=True)
        
        # Create complete date range
        full_date_range = pd.date_range(
            start=company_df.index.min(),
            end=company_df.index.max(),
            freq='D'
        )
        company_df = company_df.reindex(full_date_range)
        
        # Forward fill company code
        company_df['company_code_id'] = company
        
        # Interpolate numerical columns
        num_cols = ['revenue_amount', 'daily_deposit_amount', 'risk_num']
        company_df[num_cols] = company_df[num_cols].interpolate(method='time')
        
        # Create plot with three y-axes
        fig, ax1 = plt.subplots(figsize=(15, 7))
        plt.title(f'Company {company} Metrics Over Time')
        
        # Revenue (left axis)
        ax1.plot(company_df.index, company_df['revenue_amount'], 'g-', label='Revenue')
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Revenue Amount', color='g')
        ax1.tick_params(axis='y', labelcolor='g')
        
        # Deposit (right axis)
        ax2 = ax1.twinx()
        ax2.plot(company_df.index, company_df['daily_deposit_amount'], 'b-', label='Deposit')
        ax2.set_ylabel('Daily Deposit Amount', color='b')
        ax2.tick_params(axis='y', labelcolor='b')
        
        # Risk grade (offset right axis)
        ax3 = ax2.twinx()
        ax3.spines['right'].set_position(('outward', 60))
        ax3.plot(company_df.index, company_df['risk_num'], 'r-', label='Risk Grade')
        ax3.set_ylabel('Risk Grade (Numeric)', color='r')
        ax3.tick_params(axis='y', labelcolor='r')
        
        # Combine legends
        lines = ax1.get_legend_handles_labels()[0] + \
                ax2.get_legend_handles_labels()[0] + \
                ax3.get_legend_handles_labels()[0]
        
        labels = [l.get_label() for l in lines]
        fig.legend(lines, labels, loc='upper left', bbox_to_anchor=(0.1, 0.9))
        
        plt.tight_layout()
        plt.show()

# Example usage:
# visualize_company_data(your_dataframe)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

def plot_company_trends(df, company_id):
    """
    Plots the daily revenue, daily deposit, and risk grade trends over time for a given company.

    Parameters:
    df (pd.DataFrame): Input dataframe with columns ['date', 'company_id', 'revenue', 'deposit', 'risk_grade'].
    company_id (str or int): Company ID to filter the data.

    Returns:
    None
    """
    # Ensure date column is in datetime format
    df['date'] = pd.to_datetime(df['date'])

    # Filter data for the selected company
    company_df = df[df['company_id'] == company_id].copy()

    # Create full date range
    full_date_range = pd.date_range(start=company_df['date'].min(), end=company_df['date'].max(), freq='D')

    # Set date as index and reindex to fill missing dates
    company_df = company_df.set_index('date').reindex(full_date_range).reset_index()
    company_df.rename(columns={'index': 'date'}, inplace=True)

    # Forward fill company_id and interpolate missing values
    company_df['company_id'] = company_id
    company_df[['revenue', 'deposit']] = company_df[['revenue', 'deposit']].interpolate()
    
    # Convert categorical risk grades into numerical values for interpolation
    risk_map = {val: idx for idx, val in enumerate(sorted(df['risk_grade'].dropna().unique()))}
    inv_risk_map = {v: k for k, v in risk_map.items()}
    company_df['risk_numeric'] = company_df['risk_grade'].map(risk_map)
    company_df['risk_numeric'] = company_df['risk_numeric'].interpolate()
    company_df['risk_grade'] = company_df['risk_numeric'].round().map(inv_risk_map)

    # Plot the data
    fig, ax1 = plt.subplots(figsize=(12, 6))

    # Revenue plot
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Revenue', color='tab:blue')
    ax1.plot(company_df['date'], company_df['revenue'], color='tab:blue', label='Revenue')
    ax1.tick_params(axis='y', labelcolor='tab:blue')

    # Deposit plot
    ax2 = ax1.twinx()
    ax2.set_ylabel('Deposit', color='tab:green')
    ax2.plot(company_df['date'], company_df['deposit'], color='tab:green', label='Deposit')
    ax2.tick_params(axis='y', labelcolor='tab:green')

    # Risk Grade plot on secondary x-axis
    ax3 = ax1.twinx()
    ax3.spines['right'].set_position(('outward', 60))
    ax3.set_ylabel('Risk Grade', color='tab:red')
    ax3.plot(company_df['date'], company_df['risk_numeric'], color='tab:red', linestyle='dotted', label='Risk Grade')
    ax3.tick_params(axis='y', labelcolor='tab:red')

    # Formatting x-axis for better readability
    ax1.xaxis.set_major_locator(mdates.MonthLocator())
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))

    # Adding title and legend
    plt.title(f"Daily Revenue, Deposit & Risk Grade Trends for Company {company_id}")
    fig.tight_layout()
    plt.show()

####**********

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from datetime import datetime, timedelta

def visualize_company_metrics(df):
    """
    Create 3D visualization of company metrics over time.
    
    Parameters:
    df (pandas.DataFrame): DataFrame with columns:
        - date: Date in daily format
        - company_code: Company identifier
        - revenue: Daily revenue amount
        - deposit: Daily deposit amount
        - risk_grade: Risk grade (1, 2, 2-, 2+, 3, etc.)
    
    Returns:
    None (displays the plot)
    """
    # Convert date to datetime if not already
    df['date'] = pd.to_datetime(df['date'])
    
    # Create a complete date range
    date_range = pd.date_range(start=df['date'].min(), end=df['date'].max(), freq='D')
    
    # Convert risk grades to numeric values
    def convert_risk_grade(grade):
        if isinstance(grade, (int, float)):
            return float(grade)
        
        # Handle special cases like '2+', '2-'
        base = float(grade[0])
        if len(grade) > 1:
            if grade[1] == '+':
                return base + 0.3
            elif grade[1] == '-':
                return base - 0.3
        return base

    # Process data for each company
    unique_companies = df['company_code'].unique()
    
    # Create a figure for all companies
    fig = plt.figure(figsize=(15, 5 * len(unique_companies)))
    
    for idx, company in enumerate(unique_companies, 1):
        company_data = df[df['company_code'] == company].copy()
        
        # Convert risk grades to numeric
        company_data['risk_grade_numeric'] = company_data['risk_grade'].apply(convert_risk_grade)
        
        # Create a complete time series for this company
        company_full = pd.DataFrame({'date': date_range})
        company_full = company_full.merge(company_data, on='date', how='left')
        
        # Interpolate missing values
        for col in ['revenue', 'deposit', 'risk_grade_numeric']:
            company_full[col] = company_full[col].interpolate(method='cubic')
        
        # Create 3D plot for this company
        ax = fig.add_subplot(len(unique_companies), 1, idx, projection='3d')
        
        # Normalize data for better visualization
        revenue_norm = (company_full['revenue'] - company_full['revenue'].min()) / \
                      (company_full['revenue'].max() - company_full['revenue'].min())
        deposit_norm = (company_full['deposit'] - company_full['deposit'].min()) / \
                      (company_full['deposit'].max() - company_full['deposit'].min())
        risk_norm = (company_full['risk_grade_numeric'] - company_full['risk_grade_numeric'].min()) / \
                   (company_full['risk_grade_numeric'].max() - company_full['risk_grade_numeric'].min())
        
        # Convert dates to numeric format for plotting
        dates_num = (company_full['date'] - company_full['date'].min()).dt.days
        
        # Create the 3D scatter plot
        scatter = ax.scatter(dates_num, 
                           deposit_norm,
                           revenue_norm,
                           c=risk_norm,
                           cmap='RdYlGn_r',  # Red for high risk, green for low risk
                           s=50)
        
        # Customize the plot
        ax.set_xlabel('Time (days)')
        ax.set_ylabel('Normalized Deposit')
        ax.set_zlabel('Normalized Revenue')
        ax.set_title(f'Company {company} - 3D Metrics Visualization')
        
        # Add colorbar for risk grades
        colorbar = plt.colorbar(scatter)
        colorbar.set_label('Risk Grade (Normalized)')
        
        # Add date labels on x-axis (every 90 days)
        date_ticks = np.arange(0, len(dates_num), 90)
        date_labels = company_full['date'].dt.strftime('%Y-%m-%d').iloc[::90]
        ax.set_xticks(date_ticks)
        ax.set_xticklabels(date_labels, rotation=45)
        
        # Rotate the plot for better visibility
        ax.view_init(elev=20, azim=45)
    
    plt.tight_layout()
    plt.show()

# Example usage:
# Assuming your DataFrame is called 'data':
# visualize_company_metrics(data)
