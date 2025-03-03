import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

def generate_credit_risk_data(num_clients=100, time_periods=4):
    """
    Generate a realistic pandas DataFrame for credit risk analysis.
    
    Parameters:
    -----------
    num_clients : int
        Number of client companies to generate data for
    time_periods : int
        Number of quarterly financial reports to generate per client
    
    Returns:
    --------
    pd.DataFrame
        DataFrame containing financial metrics and risk indicators
    """
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Company sectors
    sectors = ['Technology', 'Healthcare', 'Finance', 'Energy', 'Consumer Goods', 
               'Manufacturing', 'Real Estate', 'Transportation', 'Telecommunications', 'Utilities']
    
    # Company sizes
    sizes = ['Small', 'Medium', 'Large', 'Enterprise']
    
    # Risk ratings
    risk_ratings = ['AAA', 'AA', 'A', 'BBB', 'BB', 'B', 'CCC', 'CC', 'C', 'D']
    risk_weights = [0.05, 0.1, 0.15, 0.2, 0.15, 0.15, 0.1, 0.05, 0.03, 0.02]  # Probability distribution
    
    # Generate base client data
    client_ids = [f'CL{str(i).zfill(5)}' for i in range(1, num_clients + 1)]
    client_names = [f'Company {chr(65 + i % 26)}{chr(65 + (i // 26) % 26)}{i % 1000}' for i in range(num_clients)]
    client_sectors = np.random.choice(sectors, num_clients)
    client_sizes = np.random.choice(sizes, num_clients, p=[0.3, 0.4, 0.2, 0.1])
    
    # Generate time periods (quarters)
    current_date = datetime.now()
    quarters = []
    for i in range(time_periods):
        quarter_date = current_date - timedelta(days=90 * i)
        quarter = f"Q{((quarter_date.month - 1) // 3) + 1} {quarter_date.year}"
        quarters.append(quarter)
    
    # Lists to store data
    data = []
    
    # Financial metrics ranges by company size
    size_metrics = {
        'Small': {
            'revenue': (1e6, 1e7),
            'assets': (5e5, 5e6),
            'liabilities': (3e5, 3e6),
            'cash': (1e5, 1e6),
            'debt': (2e5, 2e6)
        },
        'Medium': {
            'revenue': (1e7, 1e8),
            'assets': (5e6, 5e7),
            'liabilities': (3e6, 3e7),
            'cash': (1e6, 1e7),
            'debt': (2e6, 2e7)
        },
        'Large': {
            'revenue': (1e8, 1e9),
            'assets': (5e7, 5e8),
            'liabilities': (3e7, 3e8),
            'cash': (1e7, 5e7),
            'debt': (2e7, 2e8)
        },
        'Enterprise': {
            'revenue': (1e9, 1e10),
            'assets': (5e8, 5e9),
            'liabilities': (3e8, 3e9),
            'cash': (5e7, 5e8),
            'debt': (2e8, 2e9)
        }
    }
    
    # Generate financial data for each client across time periods
    for i, client_id in enumerate(client_ids):
        client_name = client_names[i]
        sector = client_sectors[i]
        size = client_sizes[i]
        
        # Base financial metrics based on company size
        base_revenue = np.random.uniform(*size_metrics[size]['revenue'])
        base_assets = np.random.uniform(*size_metrics[size]['assets'])
        base_liabilities = np.random.uniform(*size_metrics[size]['liabilities'])
        base_cash = np.random.uniform(*size_metrics[size]['cash'])
        base_debt = np.random.uniform(*size_metrics[size]['debt'])
        
        # Sector-specific trends
        if sector == 'Technology':
            revenue_trend = 1.15  # Tech companies grow faster
            debt_trend = 0.9  # Tech companies tend to have less debt growth
        elif sector == 'Energy':
            revenue_trend = 0.95  # Energy sector facing challenges
            debt_trend = 1.05  # Energy companies taking on more debt
        else:
            revenue_trend = 1.05  # Average growth
            debt_trend = 1.02  # Average debt growth
        
        # Initial risk indicators
        base_debt_to_asset = base_debt / base_assets
        base_debt_to_revenue = base_debt / base_revenue
        base_cash_ratio = base_cash / base_liabilities
        
        # Generate data for each time period
        for q_idx, quarter in enumerate(quarters):
            # Progress over time (newest to oldest)
            time_factor = 1 - (0.05 * q_idx)  # Slight decline as we go back in time
            
            # Calculate metrics with realistic variations
            revenue = base_revenue * (revenue_trend ** q_idx) * (1 + np.random.normal(0, 0.05))
            assets = base_assets * (1.03 ** q_idx) * (1 + np.random.normal(0, 0.03))
            liabilities = base_liabilities * (1.02 ** q_idx) * (1 + np.random.normal(0, 0.04))
            cash = base_cash * (1.01 ** q_idx) * (1 + np.random.normal(0, 0.08))
            debt = base_debt * (debt_trend ** q_idx) * (1 + np.random.normal(0, 0.06))
            
            # Derived financial ratios
            debt_to_asset = debt / assets
            debt_to_revenue = debt / revenue
            cash_ratio = cash / liabilities
            net_profit = revenue * np.random.uniform(0.05, 0.25)  # Profit margin 5-25%
            profit_margin = net_profit / revenue * 100
            ebitda = revenue * np.random.uniform(0.1, 0.35)  # EBITDA margin 10-35%
            debt_service_coverage = ebitda / (debt * 0.1)  # Assuming 10% of debt is serviced annually
            
            # Market factors
            market_sentiment = np.random.uniform(-1, 1)  # -1 to 1 scale
            industry_outlook = np.random.uniform(-1, 1)  # -1 to 1 scale
            
            # Documents status
            financial_statement_status = np.random.choice(['Complete', 'Incomplete', 'Under Review', 'Delayed'], p=[0.7, 0.1, 0.1, 0.1])
            earnings_report_status = np.random.choice(['Filed On Time', 'Filed Late', 'Amended', 'Not Required'], p=[0.8, 0.1, 0.05, 0.05])
            audit_status = np.random.choice(['Clean Opinion', 'Qualified Opinion', 'Adverse Opinion', 'Disclaimer of Opinion', 'Not Audited'], p=[0.75, 0.1, 0.05, 0.05, 0.05])
            
            # Risk assessment
            # Calculate risk score (0-100 scale)
            risk_factors = [
                50 * (1 - debt_to_asset),  # Lower debt/asset is better (0-50 pts)
                20 * min(cash_ratio, 1),   # Higher cash ratio is better (0-20 pts)
                15 * min(debt_service_coverage / 2, 1),  # Higher coverage is better (0-15 pts)
                10 * (1 + market_sentiment) / 2,  # Better market sentiment is better (0-10 pts)
                5 * (1 + industry_outlook) / 2    # Better industry outlook is better (0-5 pts)
            ]
            risk_score = sum(risk_factors)
            
            # Determine risk rating based on score with some randomness
            base_risk_index = min(9, max(0, int(9 - (risk_score / 100) * 9)))
            # Add some randomness (+/- 1 category)
            risk_index = min(9, max(0, base_risk_index + np.random.choice([-1, 0, 1], p=[0.2, 0.6, 0.2])))
            risk_rating = risk_ratings[risk_index]
            
            # Add data point
            data.append({
                'client_id': client_id,
                'client_name': client_name,
                'quarter': quarter,
                'sector': sector,
                'company_size': size,
                'revenue': revenue,
                'assets': assets,
                'liabilities': liabilities,
                'cash': cash,
                'debt': debt,
                'net_profit': net_profit,
                'ebitda': ebitda,
                'profit_margin': profit_margin,
                'debt_to_asset_ratio': debt_to_asset,
                'debt_to_revenue_ratio': debt_to_revenue,
                'cash_ratio': cash_ratio,
                'debt_service_coverage_ratio': debt_service_coverage,
                'market_sentiment': market_sentiment,
                'industry_outlook': industry_outlook,
                'financial_statement_status': financial_statement_status,
                'earnings_report_status': earnings_report_status,
                'audit_status': audit_status,
                'risk_score': risk_score,
                'risk_rating': risk_rating,
                'days_to_process': np.random.randint(3, 21),  # Days it took to manually process
                'analyst_id': f'AN{random.randint(1001, 1020):04d}'  # ID of analyst who processed
            })
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Add some NaN values to simulate incomplete data
    for col in ['profit_margin', 'debt_service_coverage_ratio', 'market_sentiment', 'industry_outlook']:
        mask = np.random.random(len(df)) < 0.05  # 5% of values will be NaN
        df.loc[mask, col] = np.nan
    
    return df

# Example usage
if __name__ == "__main__":
    credit_data = generate_credit_risk_data(num_clients=100, time_periods=4)
    print(f"Generated dataset with {len(credit_data)} rows and {len(credit_data.columns)} columns")
    print("\nSample data:")
    print(credit_data.head())
    
    # Display statistics
    print("\nData summary:")
    print(credit_data.describe())
    
    # Show distribution of risk ratings
    print("\nRisk rating distribution:")
    print(credit_data['risk_rating'].value_counts())
    
    # Average processing time
    print(f"\nAverage days to process: {credit_data['days_to_process'].mean():.2f}")
