import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from scipy import stats
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.ensemble import IsolationForest
from sklearn.metrics import silhouette_score, adjusted_rand_score
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.model_selection import GridSearchCV
from tqdm import tqdm
import warnings
import json
from collections import defaultdict, Counter
from typing import Dict, List, Tuple, Optional, Union
import itertools

warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("viridis")

#######################################################
# ENHANCED CONFIGURATION WITH BUSINESS LIFECYCLE INTEGRATION
#######################################################

ENHANCED_CONFIG = {
    'data': {
        'min_nonzero_pct': 0.8,
        'min_continuous_days': 365,
        'recent_window': 90,
        'lifecycle_analysis_window': 180  # 6 months for lifecycle determination
    },
    'risk': {
        'trend_windows': [30, 45, 60, 90, 120],  # More granular analysis
        'change_thresholds': {
            'sharp': 0.2,
            'moderate': 0.1,
            'gradual': 0.05
        },
        'business_lifecycle_personas': {
            # Risk-Taking Personas
            'innovation_economy': {
                'stage': 'Startup',
                'cash_flow_variability': 'High',
                'loan_usage': 'High',
                'deposit_levels': 'Low',
                'description': 'Startup with high cash flow variability, high loan usage, low deposits, engages in high-risk ventures',
                'risk_profile': 'high',
                'key_characteristics': ['high_volatility', 'high_utilization', 'growth_focused', 'cash_intensive']
            },
            'deteriorating_health': {
                'stage': 'Declining',
                'cash_flow_variability': 'High',
                'loan_usage': 'Moderate',
                'deposit_levels': 'Low',
                'description': 'Declining business with high cash flow variability, experiencing distress and cash flow challenges',
                'risk_profile': 'high',
                'key_characteristics': ['declining_trends', 'cash_flow_stress', 'increasing_dependency']
            },
            
            # Growth-Oriented Personas
            'rapidly_growing': {
                'stage': 'Growth',
                'cash_flow_variability': 'Moderate',
                'loan_usage': 'High',
                'deposit_levels': 'Moderate',
                'description': 'Growth-stage business rapidly expanding, relies heavily on loans for expansion',
                'risk_profile': 'medium',
                'key_characteristics': ['rapid_expansion', 'loan_dependent', 'growth_investments']
            },
            'strategic_planner': {
                'stage': 'Growth',
                'cash_flow_variability': 'Moderate',
                'loan_usage': 'High',
                'deposit_levels': 'High',
                'description': 'Strategic growth company using loans for sustainable expansion while maintaining strong deposits',
                'risk_profile': 'medium',
                'key_characteristics': ['strategic_growth', 'balanced_financing', 'strong_reserves']
            },
            
            # Operationally Focused Personas
            'cash_flow_inventory_manager': {
                'stage': 'Maturity',
                'cash_flow_variability': 'Moderate',
                'loan_usage': 'Moderate',
                'deposit_levels': 'Moderate',
                'description': 'Mature business balancing loans and deposits effectively for operational efficiency',
                'risk_profile': 'low',
                'key_characteristics': ['operational_efficiency', 'balanced_approach', 'working_capital_mgmt']
            },
            'seasonal_borrower': {
                'stage': 'Maturity',
                'cash_flow_variability': 'High',
                'loan_usage': 'High',
                'deposit_levels': 'Low',
                'description': 'Mature business with seasonal patterns, relies on loans during peak seasons',
                'risk_profile': 'medium',
                'key_characteristics': ['seasonal_patterns', 'cyclical_borrowing', 'predictable_cycles']
            },
            
            # Conservative Personas
            'cash_flow_business': {
                'stage': 'Maturity',
                'cash_flow_variability': 'Low',
                'loan_usage': 'Moderate',
                'deposit_levels': 'High',
                'description': 'Mature business generating steady cash flow with strong deposit base',
                'risk_profile': 'low',
                'key_characteristics': ['steady_cash_flow', 'strong_deposits', 'stable_operations']
            },
            'financially_conservative': {
                'stage': 'Maturity',
                'cash_flow_variability': 'Low',
                'loan_usage': 'Low',
                'deposit_levels': 'High',
                'description': 'Conservative mature business prioritizing cash reserves over growth',
                'risk_profile': 'low',
                'key_characteristics': ['risk_averse', 'high_liquidity', 'conservative_growth']
            },
            'conservative_operator': {
                'stage': 'Maturity',
                'cash_flow_variability': 'Low',
                'loan_usage': 'Low',
                'deposit_levels': 'Low',
                'description': 'Efficient mature business focusing on stability and operational efficiency',
                'risk_profile': 'low',
                'key_characteristics': ['operational_focus', 'efficiency_oriented', 'minimal_external_financing']
            }
        },
        'risk_levels': {
            'high': 3,
            'medium': 2,
            'low': 1,
            'none': 0
        }
    },
    'lifecycle_thresholds': {
        'startup_indicators': {
            'business_age_months': 36,  # Less than 3 years
            'volatility_threshold': 0.20,
            'growth_rate_threshold': 0.05  # 5% monthly growth
        },
        'growth_indicators': {
            'growth_rate_threshold': 0.02,  # 2% monthly growth
            'utilization_increase_threshold': 0.1,
            'deposit_growth_threshold': 0.01
        },
        'mature_indicators': {
            'stability_threshold': 0.05,  # Low volatility
            'consistent_operations_months': 12
        },
        'declining_indicators': {
            'decline_rate_threshold': -0.02,  # 2% monthly decline
            'deterioration_months': 6
        }
    },
    'clustering': {
        'n_clusters': 9,
        'random_state': 42,
        'lifecycle_weight': 0.3,  # Weight for lifecycle features in clustering
        'financial_weight': 0.7   # Weight for financial features
    }
}

#######################################################
# SECTION 1: ENHANCED RULE GENERATION WITH BUSINESS LIFECYCLE
#######################################################

def generate_business_lifecycle_synthetic_data(num_companies: int = 200, days: int = 730, 
                                             random_seed: int = 42) -> pd.DataFrame:
    """
    Generate sophisticated synthetic data with realistic business lifecycle patterns.
    
    This function creates companies that follow realistic business evolution patterns,
    from startup phase through growth, maturity, and potential decline. Each company
    exhibits financial behaviors consistent with their lifecycle stage.
    
    Parameters:
    -----------
    num_companies : int
        Number of companies to generate (default: 200)
    days : int
        Number of days of historical data (default: 730, about 2 years)
    random_seed : int
        Random seed for reproducible results (default: 42)
        
    Returns:
    --------
    pd.DataFrame
        Enhanced DataFrame with lifecycle-aware financial patterns
    """
    
    print(f"Generating business lifecycle-aware synthetic data for {num_companies} companies...")
    np.random.seed(random_seed)
    
    # Create date range
    end_date = pd.Timestamp('2024-12-31')
    start_date = end_date - pd.Timedelta(days=days)
    date_range = pd.date_range(start=start_date, end=end_date, freq='D')
    
    # Define realistic lifecycle distribution based on business demographics
    lifecycle_distribution = {
        'innovation_economy': 0.08,      # 8% - High-risk startups
        'deteriorating_health': 0.05,    # 5% - Declining businesses
        'rapidly_growing': 0.12,         # 12% - Fast growth companies
        'strategic_planner': 0.10,       # 10% - Strategic growth companies
        'cash_flow_inventory_manager': 0.18,  # 18% - Operational managers
        'seasonal_borrower': 0.12,       # 12% - Seasonal businesses
        'cash_flow_business': 0.15,      # 15% - Steady cash flow generators
        'financially_conservative': 0.12, # 12% - Conservative companies
        'conservative_operator': 0.08    # 8% - Efficient operators
    }
    
    # Assign companies to personas
    company_personas = []
    for persona, percentage in lifecycle_distribution.items():
        count = int(num_companies * percentage)
        company_personas.extend([persona] * count)
    
    # Fill remaining with most common persona
    while len(company_personas) < num_companies:
        company_personas.append('cash_flow_inventory_manager')
    
    np.random.shuffle(company_personas)
    
    # Generate sophisticated data for each company
    data = []
    
    for i, persona in enumerate(tqdm(company_personas, desc="Generating lifecycle-aware companies")):
        company_id = f'COMP_{i:04d}'
        persona_config = ENHANCED_CONFIG['risk']['business_lifecycle_personas'][persona]
        
        # Initialize company parameters based on persona
        company_data = initialize_company_by_persona(persona, persona_config, date_range)
        
        # Generate time series data
        for day_idx, current_date in enumerate(date_range):
            daily_metrics = generate_daily_metrics(
                company_data, persona, persona_config, day_idx, len(date_range), current_date
            )
            
            data.append({
                'company_id': company_id,
                'date': current_date,
                'deposit_balance': max(0, daily_metrics['deposit']),
                'used_loan': max(0, daily_metrics['used_loan']),
                'unused_loan': max(0, daily_metrics['unused_loan']),
                'persona_type': persona,
                'business_stage': persona_config['stage'],
                'risk_profile': persona_config['risk_profile']
            })
    
    df = pd.DataFrame(data)
    print(f"Generated {len(df)} records with realistic business lifecycle patterns")
    
    # Display persona distribution
    persona_counts = df['persona_type'].value_counts()
    print("\nBusiness Persona Distribution:")
    for persona, count in persona_counts.items():
        companies = count // len(date_range)
        percentage = companies / num_companies * 100
        stage = ENHANCED_CONFIG['risk']['business_lifecycle_personas'][persona]['stage']
        print(f"  {persona} ({stage}): {companies} companies ({percentage:.1f}%)")
    
    return df

def initialize_company_by_persona(persona: str, persona_config: Dict, date_range) -> Dict:
    """
    Initialize company financial parameters based on their business persona.
    
    This function sets realistic starting conditions and behavioral parameters
    that reflect the financial characteristics of each business lifecycle stage.
    """
    
    stage = persona_config['stage']
    
    # Base financial parameters vary by lifecycle stage
    if stage == 'Startup':
        base_deposit = np.random.lognormal(10.5, 1.5)  # Smaller, more variable deposits
        base_loan = np.random.lognormal(10.0, 1.3)     # Moderate loans
        initial_utilization = np.random.uniform(0.6, 0.85)  # High utilization
        growth_volatility = np.random.uniform(0.15, 0.35)   # High volatility
        
    elif stage == 'Growth':
        base_deposit = np.random.lognormal(11.5, 1.2)  # Growing deposits
        base_loan = np.random.lognormal(11.0, 1.1)     # Significant loans for expansion
        initial_utilization = np.random.uniform(0.5, 0.75)  # Moderate-high utilization
        growth_volatility = np.random.uniform(0.08, 0.20)   # Moderate volatility
        
    elif stage == 'Maturity':
        base_deposit = np.random.lognormal(12.0, 1.0)  # Stable, larger deposits
        base_loan = np.random.lognormal(11.2, 1.0)     # Established loan capacity
        initial_utilization = np.random.uniform(0.3, 0.65)  # Variable by persona
        growth_volatility = np.random.uniform(0.03, 0.12)   # Lower volatility
        
    else:  # Declining stage
        base_deposit = np.random.lognormal(11.0, 1.4)  # Declining deposits
        base_loan = np.random.lognormal(10.8, 1.2)     # May have loan capacity
        initial_utilization = np.random.uniform(0.4, 0.70)  # Stressed utilization
        growth_volatility = np.random.uniform(0.12, 0.25)   # High volatility due to stress
    
    # Persona-specific adjustments
    persona_adjustments = get_persona_specific_adjustments(persona, persona_config)
    
    return {
        'base_deposit': base_deposit * persona_adjustments['deposit_multiplier'],
        'base_loan': base_loan * persona_adjustments['loan_multiplier'],
        'initial_utilization': min(0.95, initial_utilization * persona_adjustments['utilization_multiplier']),
        'growth_volatility': growth_volatility * persona_adjustments['volatility_multiplier'],
        'trend_direction': persona_adjustments['trend_direction'],
        'seasonal_strength': persona_adjustments['seasonal_strength']
    }

def get_persona_specific_adjustments(persona: str, persona_config: Dict) -> Dict:
    """
    Get persona-specific adjustments to financial parameters.
    
    This function fine-tunes the financial behavior based on the specific
    characteristics of each business persona within their lifecycle stage.
    """
    
    # Default adjustments
    adjustments = {
        'deposit_multiplier': 1.0,
        'loan_multiplier': 1.0,
        'utilization_multiplier': 1.0,
        'volatility_multiplier': 1.0,
        'trend_direction': 0.0,
        'seasonal_strength': 0.05
    }
    
    # Persona-specific customizations
    if persona == 'innovation_economy':
        adjustments.update({
            'deposit_multiplier': 0.7,    # Lower deposits
            'utilization_multiplier': 1.2, # Higher utilization
            'volatility_multiplier': 1.8,  # Much higher volatility
            'trend_direction': 0.001       # Slight growth trend
        })
        
    elif persona == 'deteriorating_health':
        adjustments.update({
            'deposit_multiplier': 0.8,     # Declining deposits
            'volatility_multiplier': 1.5,  # High volatility
            'trend_direction': -0.002      # Declining trend
        })
        
    elif persona == 'rapidly_growing':
        adjustments.update({
            'loan_multiplier': 1.3,        # Higher loan capacity
            'utilization_multiplier': 1.1, # Higher utilization
            'trend_direction': 0.003       # Strong growth trend
        })
        
    elif persona == 'strategic_planner':
        adjustments.update({
            'deposit_multiplier': 1.4,     # Higher deposits
            'loan_multiplier': 1.2,        # Higher loans
            'trend_direction': 0.002       # Steady growth
        })
        
    elif persona == 'seasonal_borrower':
        adjustments.update({
            'seasonal_strength': 0.25,     # Strong seasonality
            'volatility_multiplier': 1.3   # Higher volatility due to seasonality
        })
        
    elif persona == 'financially_conservative':
        adjustments.update({
            'deposit_multiplier': 1.5,     # Much higher deposits
            'utilization_multiplier': 0.6, # Lower utilization
            'volatility_multiplier': 0.5   # Much lower volatility
        })
        
    elif persona == 'conservative_operator':
        adjustments.update({
            'deposit_multiplier': 0.8,     # Lower deposits (efficient operations)
            'utilization_multiplier': 0.7, # Lower utilization
            'volatility_multiplier': 0.4   # Very low volatility
        })
    
    return adjustments

def generate_daily_metrics(company_data: Dict, persona: str, persona_config: Dict,
                          day_idx: int, total_days: int, current_date: datetime) -> Dict:
    """
    Generate daily financial metrics for a company based on their persona and lifecycle stage.
    
    This function creates realistic day-to-day financial behavior that reflects both
    the company's business persona and their position in the business lifecycle.
    """
    
    # Time-based factors
    time_progress = day_idx / total_days
    
    # Base values from company initialization
    base_deposit = company_data['base_deposit']
    base_loan = company_data['base_loan']
    base_utilization = company_data['initial_utilization']
    volatility = company_data['growth_volatility']
    trend_direction = company_data['trend_direction']
    seasonal_strength = company_data['seasonal_strength']
    
    # Calculate trend component based on lifecycle stage
    trend_factor = calculate_lifecycle_trend(persona_config['stage'], time_progress, trend_direction)
    
    # Add seasonality (stronger for certain personas)
    seasonal_factor = 1 + seasonal_strength * np.sin(2 * np.pi * current_date.dayofyear / 365)
    
    # Add realistic volatility
    volatility_factor_deposit = np.random.normal(1, volatility * 0.6)
    volatility_factor_util = np.random.normal(0, volatility * 0.4)
    
    # Special business events (more frequent for startups and declining businesses)
    event_probability = get_event_probability(persona_config['stage'])
    event_factor_deposit = 1
    event_factor_util = 0
    
    if np.random.random() < event_probability:
        event_type = determine_event_type(persona_config['stage'], persona)
        event_factor_deposit, event_factor_util = apply_business_event(event_type)
    
    # Calculate final values
    final_deposit = (base_deposit * trend_factor * seasonal_factor * 
                    volatility_factor_deposit * event_factor_deposit)
    
    final_utilization = min(0.98, max(0.05, 
        base_utilization + volatility_factor_util + event_factor_util +
        calculate_utilization_lifecycle_adjustment(persona_config['stage'], time_progress)
    ))
    
    used_loan = base_loan * final_utilization
    unused_loan = base_loan * (1 - final_utilization)
    
    return {
        'deposit': final_deposit,
        'used_loan': used_loan,
        'unused_loan': unused_loan
    }

def calculate_lifecycle_trend(stage: str, time_progress: float, base_trend: float) -> float:
    """Calculate trend factor based on business lifecycle stage."""
    
    if stage == 'Startup':
        # Startups often have exponential growth early, then level off
        growth_curve = np.exp(base_trend * time_progress * 10) if base_trend > 0 else 1 + base_trend * time_progress
        return max(0.5, growth_curve)
        
    elif stage == 'Growth':
        # Growth companies have sustained but moderating growth
        return 1 + base_trend * time_progress * 5
        
    elif stage == 'Maturity':
        # Mature companies have steady, predictable trends
        return 1 + base_trend * time_progress * 2
        
    else:  # Declining
        # Declining companies accelerate downward over time
        decline_acceleration = 1 + time_progress * 0.5  # Accelerating decline
        return max(0.3, 1 + base_trend * time_progress * decline_acceleration)

def get_event_probability(stage: str) -> float:
    """Get probability of special business events based on lifecycle stage."""
    
    event_probabilities = {
        'Startup': 0.008,      # Higher chance of significant events
        'Growth': 0.005,       # Moderate chance of events
        'Maturity': 0.003,     # Lower chance of major events
        'Declining': 0.007     # Higher chance due to instability
    }
    
    return event_probabilities.get(stage, 0.003)

def determine_event_type(stage: str, persona: str) -> str:
    """Determine the type of business event based on stage and persona."""
    
    if stage == 'Startup':
        return np.random.choice(['funding_round', 'market_breakthrough', 'cash_crunch'], 
                               p=[0.4, 0.3, 0.3])
    elif stage == 'Growth':
        return np.random.choice(['expansion', 'new_contract', 'investment'], 
                               p=[0.4, 0.4, 0.2])
    elif stage == 'Maturity':
        return np.random.choice(['seasonal_peak', 'operational_efficiency', 'market_adjustment'], 
                               p=[0.4, 0.3, 0.3])
    else:  # Declining
        return np.random.choice(['restructuring', 'asset_sale', 'emergency_funding'], 
                               p=[0.4, 0.3, 0.3])

def apply_business_event(event_type: str) -> Tuple[float, float]:
    """Apply the effects of a business event to deposits and utilization."""
    
    event_effects = {
        'funding_round': (np.random.uniform(1.5, 3.0), np.random.uniform(-0.1, -0.05)),
        'market_breakthrough': (np.random.uniform(1.2, 2.0), np.random.uniform(-0.05, 0.05)),
        'cash_crunch': (np.random.uniform(0.6, 0.85), np.random.uniform(0.05, 0.15)),
        'expansion': (np.random.uniform(0.9, 1.3), np.random.uniform(0.02, 0.08)),
        'new_contract': (np.random.uniform(1.1, 1.4), np.random.uniform(-0.02, 0.02)),
        'investment': (np.random.uniform(0.85, 1.1), np.random.uniform(0.03, 0.07)),
        'seasonal_peak': (np.random.uniform(1.1, 1.3), np.random.uniform(0.01, 0.05)),
        'operational_efficiency': (np.random.uniform(1.05, 1.15), np.random.uniform(-0.02, 0.01)),
        'market_adjustment': (np.random.uniform(0.9, 1.05), np.random.uniform(-0.01, 0.03)),
        'restructuring': (np.random.uniform(0.7, 0.9), np.random.uniform(0.02, 0.08)),
        'asset_sale': (np.random.uniform(1.2, 1.8), np.random.uniform(-0.05, 0.00)),
        'emergency_funding': (np.random.uniform(1.1, 1.5), np.random.uniform(0.05, 0.12))
    }
    
    return event_effects.get(event_type, (1.0, 0.0))

def calculate_comprehensive_business_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate comprehensive business and financial features with lifecycle awareness.
    
    This enhanced function creates features that capture not just financial metrics,
    but also business lifecycle indicators, operational patterns, and strategic behaviors
    that are relevant to different business stages.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input data with business lifecycle information
        
    Returns:
    --------
    pd.DataFrame
        Enhanced dataframe with comprehensive business-aware features
    """
    
    print("Calculating comprehensive business lifecycle features...")
    
    # Create basic derived metrics
    df_features = df.copy()
    df_features['total_loan'] = df_features['used_loan'] + df_features['unused_loan']
    df_features['loan_utilization'] = np.where(
        df_features['total_loan'] > 0,
        df_features['used_loan'] / df_features['total_loan'],
        0
    )
    
    # Ensure date is datetime
    df_features['date'] = pd.to_datetime(df_features['date'])
    
    feature_records = []
    
    # Enhanced trend windows from configuration
    windows = ENHANCED_CONFIG['risk']['trend_windows']  # [30, 45, 60, 90, 120]
    
    for company_id in tqdm(df_features['company_id'].unique(), desc="Computing business features"):
        company_data = df_features[df_features['company_id'] == company_id].sort_values('date').copy()
        
        if len(company_data) < 60:
            continue
        
        # Get company's persona and stage information
        persona_type = company_data['persona_type'].iloc[0]
        business_stage = company_data['business_stage'].iloc[0]
        
        # Calculate standard financial features for all windows
        for window in windows:
            if len(company_data) >= window:
                # Rolling means and volatilities
                company_data[f'util_mean_{window}d'] = company_data['loan_utilization'].rolling(
                    window, min_periods=max(3, window//4)).mean()
                company_data[f'deposit_mean_{window}d'] = company_data['deposit_balance'].rolling(
                    window, min_periods=max(3, window//4)).mean()
                
                company_data[f'util_vol_{window}d'] = company_data['loan_utilization'].rolling(
                    window, min_periods=max(3, window//4)).std()
                company_data[f'deposit_vol_{window}d'] = company_data['deposit_balance'].pct_change().rolling(
                    window, min_periods=max(3, window//4)).std()
                
                # Trend calculations
                if window >= 30:
                    util_trend = company_data['loan_utilization'].rolling(
                        window, min_periods=window//2).apply(
                        lambda x: calculate_trend_slope(x), raw=False)
                    company_data[f'util_trend_{window}d'] = util_trend
                    
                    deposit_trend = company_data['deposit_balance'].rolling(
                        window, min_periods=window//2).apply(
                        lambda x: calculate_trend_slope(x), raw=False)
                    company_data[f'deposit_trend_{window}d'] = deposit_trend
        
        # Calculate change rates for all windows
        for period in windows:
            if len(company_data) > period:
                company_data[f'util_change_{period}d'] = company_data['loan_utilization'].pct_change(periods=period)
                company_data[f'deposit_change_{period}d'] = company_data['deposit_balance'].pct_change(periods=period)
        
        # Enhanced business lifecycle features
        company_data = calculate_business_lifecycle_features(company_data, persona_type, business_stage)
        
        # Advanced pattern recognition features
        company_data = calculate_advanced_pattern_features(company_data)
        
        # Strategic behavior indicators
        company_data = calculate_strategic_behavior_features(company_data)
        
        # Add to results
        for _, row in company_data.iterrows():
            feature_records.append(row.to_dict())
    
    df_enhanced = pd.DataFrame(feature_records)
    
    total_features = len([col for col in df_enhanced.columns 
                         if any(suffix in col for suffix in ['_mean_', '_vol_', '_trend_', '_change_', '_score', '_ratio'])])
    
    print(f"Calculated {total_features} enhanced features for {df_enhanced['company_id'].nunique()} companies")
    
    return df_enhanced

def calculate_business_lifecycle_features(company_data: pd.DataFrame, 
                                        persona_type: str, 
                                        business_stage: str) -> pd.DataFrame:
    """
    Calculate features specific to business lifecycle analysis.
    
    These features help identify where a company is in their business lifecycle
    and how their financial behavior aligns with typical patterns for their stage.
    """
    
    # Lifecycle stage consistency score
    stage_consistency = calculate_stage_consistency_score(company_data, business_stage)
    company_data['lifecycle_consistency_score'] = stage_consistency
    
    # Growth trajectory analysis
    if len(company_data) >= 90:
        growth_trajectory = analyze_growth_trajectory(company_data)
        company_data['growth_trajectory_score'] = growth_trajectory['score']
        company_data['growth_acceleration'] = growth_trajectory['acceleration']
        company_data['growth_sustainability'] = growth_trajectory['sustainability']
    
    # Business maturity indicators
    maturity_indicators = calculate_maturity_indicators(company_data)
    company_data['operational_stability_score'] = maturity_indicators['stability']
    company_data['financial_sophistication_score'] = maturity_indicators['sophistication']
    
    # Risk-adjusted performance metrics
    if business_stage == 'Startup':
        company_data['startup_burn_rate'] = calculate_startup_burn_rate(company_data)
        company_data['runway_sustainability'] = calculate_runway_sustainability(company_data)
        
    elif business_stage == 'Growth':
        company_data['growth_efficiency_score'] = calculate_growth_efficiency(company_data)
        company_data['expansion_sustainability'] = calculate_expansion_sustainability(company_data)
        
    elif business_stage == 'Maturity':
        company_data['operational_efficiency_score'] = calculate_operational_efficiency(company_data)
        company_data['cash_conversion_efficiency'] = calculate_cash_conversion_efficiency(company_data)
        
    else:  # Declining
        company_data['distress_probability'] = calculate_distress_probability(company_data)
        company_data['turnaround_potential'] = calculate_turnaround_potential(company_data)
    
    return company_data

def calculate_stage_consistency_score(company_data: pd.DataFrame, expected_stage: str) -> pd.Series:
    """Calculate how consistently a company's behavior matches their expected lifecycle stage."""
    
    if len(company_data) < 30:
        return pd.Series([0.5] * len(company_data), index=company_data.index)
    
    # Define stage-specific characteristics
    stage_characteristics = {
        'Startup': {
            'high_volatility': lambda x: x > 0.15,
            'high_growth_potential': lambda x: x > 0.02,
            'resource_constrained': lambda x: x > 0.7
        },
        'Growth': {
            'moderate_volatility': lambda x: 0.05 < x < 0.20,
            'consistent_growth': lambda x: x > 0.01,
            'strategic_borrowing': lambda x: 0.4 < x < 0.8
        },
        'Maturity': {
            'low_volatility': lambda x: x < 0.10,
            'stable_operations': lambda x: abs(x) < 0.01,
            'balanced_financing': lambda x: 0.2 < x < 0.6
        },
        'Declining': {
            'increasing_volatility': lambda x: x > 0.12,
            'negative_trends': lambda x: x < -0.01,
            'stress_indicators': lambda x: x > 0.6
        }
    }
    
    consistency_scores = []
    
    for i in range(len(company_data)):
        if i < 30:
            consistency_scores.append(0.5)
            continue
        
        # Get recent data window
        recent_window = company_data.iloc[max(0, i-30):i+1]
        
        # Calculate relevant metrics
        volatility = recent_window['loan_utilization'].std() if len(recent_window) > 1 else 0
        growth_trend = calculate_trend_slope(recent_window['deposit_balance'])
        utilization = recent_window['loan_utilization'].mean()
        
        # Score against expected stage characteristics
        if expected_stage in stage_characteristics:
            characteristics = stage_characteristics[expected_stage]
            matches = 0
            total_tests = len(characteristics)
            
            for char_name, test_func in characteristics.items():
                if 'volatility' in char_name and test_func(volatility):
                    matches += 1
                elif 'growth' in char_name and test_func(growth_trend):
                    matches += 1
                elif any(term in char_name for term in ['borrowing', 'financing', 'constrained', 'indicators']) and test_func(utilization):
                    matches += 1
                elif 'trend' in char_name and test_func(growth_trend):
                    matches += 1
            
            consistency_score = matches / total_tests if total_tests > 0 else 0.5
        else:
            consistency_score = 0.5
        
        consistency_scores.append(consistency_score)
    
    return pd.Series(consistency_scores, index=company_data.index)

def analyze_growth_trajectory(company_data: pd.DataFrame) -> Dict:
    """Analyze the growth trajectory and sustainability of a company."""
    
    if len(company_data) < 90:
        return {'score': 0.5, 'acceleration': 0, 'sustainability': 0.5}
    
    # Calculate growth metrics over different periods
    deposit_values = company_data['deposit_balance'].values
    util_values = company_data['loan_utilization'].values
    
    # Short-term growth (30 days)
    short_term_growth = (deposit_values[-1] - deposit_values[-30]) / deposit_values[-30] if deposit_values[-30] > 0 else 0
    
    # Medium-term growth (90 days)
    medium_term_growth = (deposit_values[-1] - deposit_values[-90]) / deposit_values[-90] if deposit_values[-90] > 0 else 0
    
    # Growth acceleration
    acceleration = short_term_growth - (medium_term_growth / 3)  # Compare monthly rates
    
    # Growth sustainability (consistency of growth)
    monthly_growth_rates = []
    for i in range(30, len(deposit_values), 30):
        if i + 30 <= len(deposit_values) and deposit_values[i-30] > 0:
            monthly_rate = (deposit_values[i] - deposit_values[i-30]) / deposit_values[i-30]
            monthly_growth_rates.append(monthly_rate)
    
    sustainability = 1 - (np.std(monthly_growth_rates) / (abs(np.mean(monthly_growth_rates)) + 0.01)) if monthly_growth_rates else 0.5
    sustainability = max(0, min(1, sustainability))
    
    # Overall growth score
    growth_score = (
        0.4 * min(1, max(0, short_term_growth * 10)) +  # Normalize growth rate
        0.3 * min(1, max(0, medium_term_growth * 5)) +
        0.3 * sustainability
    )
    
    return {
        'score': growth_score,
        'acceleration': acceleration,
        'sustainability': sustainability
    }

def calculate_maturity_indicators(company_data: pd.DataFrame) -> Dict:
    """Calculate indicators of business maturity and operational sophistication."""
    
    if len(company_data) < 60:
        return {'stability': 0.5, 'sophistication': 0.5}
    
    # Operational stability (lower volatility = higher stability)
    util_volatility = company_data['loan_utilization'].rolling(60).std().mean()
    deposit_volatility = company_data['deposit_balance'].pct_change().rolling(60).std().mean()
    
    stability_score = max(0, 1 - (util_volatility * 5 + deposit_volatility * 2))
    
    # Financial sophistication (balanced use of financial instruments)
    avg_utilization = company_data['loan_utilization'].mean()
    utilization_consistency = 1 - company_data['loan_utilization'].std()
    
    # Sophisticated companies maintain moderate, consistent utilization
    optimal_utilization = 0.5
    utilization_score = 1 - abs(avg_utilization - optimal_utilization) * 2
    
    sophistication_score = (utilization_score * 0.6 + utilization_consistency * 0.4)
    sophistication_score = max(0, min(1, sophistication_score))
    
    return {
        'stability': max(0, min(1, stability_score)),
        'sophistication': sophistication_score
    }

def calculate_startup_burn_rate(company_data: pd.DataFrame) -> pd.Series:
    """Calculate burn rate for startup companies."""
    
    burn_rates = []
    
    for i in range(len(company_data)):
        if i < 30:
            burn_rates.append(0)
            continue
        
        # Calculate 30-day deposit change as proxy for burn rate
        current_deposit = company_data.iloc[i]['deposit_balance']
        past_deposit = company_data.iloc[i-30]['deposit_balance']
        
        if past_deposit > 0:
            burn_rate = max(0, (past_deposit - current_deposit) / past_deposit)
        else:
            burn_rate = 0
        
        burn_rates.append(burn_rate)
    
    return pd.Series(burn_rates, index=company_data.index)

def calculate_runway_sustainability(company_data: pd.DataFrame) -> pd.Series:
    """Calculate runway sustainability for startup companies."""
    
    sustainability_scores = []
    
    for i in range(len(company_data)):
        if i < 60:
            sustainability_scores.append(0.5)
            continue
        
        # Look at trend in deposits vs utilization
        recent_data = company_data.iloc[max(0, i-60):i+1]
        
        deposit_trend = calculate_trend_slope(recent_data['deposit_balance'])
        util_trend = calculate_trend_slope(recent_data['loan_utilization'])
        
        # Good sustainability: growing deposits, stable or decreasing utilization
        if deposit_trend > 0 and util_trend <= 0:
            score = 0.8
        elif deposit_trend > 0:
            score = 0.6
        elif deposit_trend >= -0.01:  # Stable deposits
            score = 0.5
        else:
            score = max(0.1, 0.5 + deposit_trend * 50)  # Declining deposits
        
        sustainability_scores.append(score)
    
    return pd.Series(sustainability_scores, index=company_data.index)

def calculate_growth_efficiency(company_data: pd.DataFrame) -> pd.Series:
    """Calculate growth efficiency for growth-stage companies."""
    
    efficiency_scores = []
    
    for i in range(len(company_data)):
        if i < 90:
            efficiency_scores.append(0.5)
            continue
        
        # Analyze relationship between loan usage and deposit growth
        recent_data = company_data.iloc[max(0, i-90):i+1]
        
        deposit_growth = (recent_data['deposit_balance'].iloc[-1] - recent_data['deposit_balance'].iloc[0]) / recent_data['deposit_balance'].iloc[0]
        avg_utilization = recent_data['loan_utilization'].mean()
        
        # Efficient growth: moderate utilization with good deposit growth
        if deposit_growth > 0.05 and 0.4 < avg_utilization < 0.7:
            score = 0.8
        elif deposit_growth > 0.02:
            score = 0.6
        elif deposit_growth > 0:
            score = 0.5
        else:
            score = 0.3
        
        efficiency_scores.append(score)
    
    return pd.Series(efficiency_scores, index=company_data.index)

def calculate_expansion_sustainability(company_data: pd.DataFrame) -> pd.Series:
    """Calculate expansion sustainability for growth companies."""
    
    sustainability_scores = []
    
    for i in range(len(company_data)):
        if i < 120:
            sustainability_scores.append(0.5)
            continue
        
        # Look at growth consistency over time
        recent_data = company_data.iloc[max(0, i-120):i+1]
        
        # Calculate growth rates over different periods
        growth_30d = (recent_data['deposit_balance'].iloc[-1] - recent_data['deposit_balance'].iloc[-30]) / recent_data['deposit_balance'].iloc[-30]
        growth_60d = (recent_data['deposit_balance'].iloc[-1] - recent_data['deposit_balance'].iloc[-60]) / recent_data['deposit_balance'].iloc[-60] / 2  # Monthly rate
        growth_90d = (recent_data['deposit_balance'].iloc[-1] - recent_data['deposit_balance'].iloc[-90]) / recent_data['deposit_balance'].iloc[-90] / 3  # Monthly rate
        
        # Sustainable growth shows consistency
        growth_rates = [growth_30d, growth_60d, growth_90d]
        growth_consistency = 1 - (np.std(growth_rates) / (abs(np.mean(growth_rates)) + 0.01))
        
        # Factor in utilization sustainability
        util_trend = calculate_trend_slope(recent_data['loan_utilization'])
        util_sustainability = 1 if util_trend <= 0 else max(0, 1 - util_trend * 50)
        
        overall_sustainability = (growth_consistency * 0.7 + util_sustainability * 0.3)
        sustainability_scores.append(max(0, min(1, overall_sustainability)))
    
    return pd.Series(sustainability_scores, index=company_data.index)

def calculate_operational_efficiency(company_data: pd.DataFrame) -> pd.Series:
    """Calculate operational efficiency for mature companies."""
    
    efficiency_scores = []
    
    for i in range(len(company_data)):
        if i < 90:
            efficiency_scores.append(0.5)
            continue
        
        recent_data = company_data.iloc[max(0, i-90):i+1]
        
        # Efficient operations: stable deposits, low volatility, optimal utilization
        deposit_stability = 1 - recent_data['deposit_balance'].pct_change().std()
        util_stability = 1 - recent_data['loan_utilization'].std()
        
        # Optimal utilization for mature companies (around 40-60%)
        avg_util = recent_data['loan_utilization'].mean()
        utilization_optimality = 1 - abs(avg_util - 0.5) * 2
        
        efficiency = (deposit_stability * 0.4 + util_stability * 0.4 + utilization_optimality * 0.2)
        efficiency_scores.append(max(0, min(1, efficiency)))
    
    return pd.Series(efficiency_scores, index=company_data.index)

def calculate_cash_conversion_efficiency(company_data: pd.DataFrame) -> pd.Series:
    """Calculate cash conversion efficiency for mature companies."""
    
    efficiency_scores = []
    
    for i in range(len(company_data)):
        if i < 60:
            efficiency_scores.append(0.5)
            continue
        
        recent_data = company_data.iloc[max(0, i-60):i+1]
        
        # Cash conversion: relationship between deposits and loan usage
        deposit_to_loan_ratio = recent_data['deposit_balance'].mean() / (recent_data['used_loan'].mean() + 1)
        
        # Optimal ratio varies, but mature companies should have reasonable coverage
        if 1.5 <= deposit_to_loan_ratio <= 4.0:
            ratio_score = 1.0
        elif 0.8 <= deposit_to_loan_ratio < 1.5:
            ratio_score = 0.7
        elif deposit_to_loan_ratio > 4.0:
            ratio_score = 0.8  # Too much cash might be inefficient
        else:
            ratio_score = 0.3
        
        efficiency_scores.append(ratio_score)
    
    return pd.Series(efficiency_scores, index=company_data.index)

def calculate_distress_probability(company_data: pd.DataFrame) -> pd.Series:
    """Calculate probability of financial distress for declining companies."""
    
    distress_scores = []
    
    for i in range(len(company_data)):
        if i < 90:
            distress_scores.append(0.5)
            continue
        
        recent_data = company_data.iloc[max(0, i-90):i+1]
        
        # Distress indicators
        deposit_decline = (recent_data['deposit_balance'].iloc[0] - recent_data['deposit_balance'].iloc[-1]) / recent_data['deposit_balance'].iloc[0]
        util_increase = recent_data['loan_utilization'].iloc[-1] - recent_data['loan_utilization'].iloc[0]
        volatility_increase = recent_data['loan_utilization'].std()
        
        # Higher scores indicate higher distress probability
        distress_score = (
            deposit_decline * 2 +  # Declining deposits
            util_increase * 1.5 +  # Increasing utilization
            volatility_increase * 1  # Higher volatility
        )
        
        distress_probability = min(1, max(0, distress_score))
        distress_scores.append(distress_probability)
    
    return pd.Series(distress_scores, index=company_data.index)

def calculate_turnaround_potential(company_data: pd.DataFrame) -> pd.Series:
    """Calculate turnaround potential for declining companies."""
    
    turnaround_scores = []
    
    for i in range(len(company_data)):
        if i < 120:
            turnaround_scores.append(0.5)
            continue
        
        recent_data = company_data.iloc[max(0, i-60):i+1]
        historical_data = company_data.iloc[max(0, i-120):max(0, i-60)]
        
        # Turnaround indicators: improvement in key metrics
        recent_deposit_trend = calculate_trend_slope(recent_data['deposit_balance'])
        historical_deposit_trend = calculate_trend_slope(historical_data['deposit_balance'])
        
        recent_util_trend = calculate_trend_slope(recent_data['loan_utilization'])
        historical_util_trend = calculate_trend_slope(historical_data['loan_utilization'])
        
        # Positive turnaround: improving deposits, stabilizing utilization
        deposit_improvement = recent_deposit_trend - historical_deposit_trend
        util_stabilization = abs(historical_util_trend) - abs(recent_util_trend)
        
        turnaround_score = (
            max(0, deposit_improvement * 1000) * 0.6 +  # Scale trend differences
            max(0, util_stabilization * 500) * 0.4
        )
        
        turnaround_potential = min(1, turnaround_score)
        turnaround_scores.append(turnaround_potential)
    
    return pd.Series(turnaround_scores, index=company_data.index)

def calculate_advanced_pattern_features(company_data: pd.DataFrame) -> pd.DataFrame:
    """Calculate advanced pattern recognition features for sophisticated analysis."""
    
    if len(company_data) < 120:
        return company_data
    
    # Cyclical pattern detection
    company_data['cyclical_strength'] = detect_cyclical_patterns(company_data)
    
    # Momentum indicators
    company_data['momentum_score'] = calculate_momentum_indicators(company_data)
    
    # Regime change detection
    company_data['regime_change_probability'] = detect_regime_changes(company_data)
    
    # Stress test indicators
    company_data['stress_resilience_score'] = calculate_stress_resilience(company_data)
    
    return company_data

def detect_cyclical_patterns(company_data: pd.DataFrame) -> pd.Series:
    """Detect cyclical patterns in financial behavior."""
    
    cyclical_scores = []
    
    for i in range(len(company_data)):
        if i < 120:
            cyclical_scores.append(0)
            continue
        
        # Analyze cyclical patterns in the last 120 days
        recent_data = company_data.iloc[max(0, i-120):i+1]
        
        # Use FFT to detect cyclical patterns
        util_values = recent_data['loan_utilization'].fillna(method='ffill').fillna(0)
        
        if len(util_values) > 30 and util_values.std() > 0:
            try:
                fft_values = np.fft.fft(util_values - util_values.mean())
                fft_magnitude = np.abs(fft_values[1:len(fft_values)//2])
                
                if len(fft_magnitude) > 0:
                    # Strength of cyclical patterns
                    cyclical_strength = np.max(fft_magnitude) / np.sum(fft_magnitude)
                    cyclical_scores.append(min(1, cyclical_strength * 5))
                else:
                    cyclical_scores.append(0)
            except:
                cyclical_scores.append(0)
        else:
            cyclical_scores.append(0)
    
    return pd.Series(cyclical_scores, index=company_data.index)

def calculate_momentum_indicators(company_data: pd.DataFrame) -> pd.Series:
    """Calculate momentum indicators for trend analysis."""
    
    momentum_scores = []
    
    for i in range(len(company_data)):
        if i < 60:
            momentum_scores.append(0)
            continue
        
        # Calculate momentum using multiple time frames
        short_data = company_data.iloc[max(0, i-30):i+1]
        medium_data = company_data.iloc[max(0, i-60):i+1]
        
        # Deposit momentum
        short_deposit_trend = calculate_trend_slope(short_data['deposit_balance'])
        medium_deposit_trend = calculate_trend_slope(medium_data['deposit_balance'])
        
        # Utilization momentum
        short_util_trend = calculate_trend_slope(short_data['loan_utilization'])
        medium_util_trend = calculate_trend_slope(medium_data['loan_utilization'])
        
        # Momentum acceleration
        deposit_momentum = short_deposit_trend - medium_deposit_trend
        util_momentum = short_util_trend - medium_util_trend
        
        # Combined momentum score
        momentum_score = (deposit_momentum * 500 - util_momentum * 200)  # Scale and weight
        momentum_scores.append(max(-1, min(1, momentum_score)))
    
    return pd.Series(momentum_scores, index=company_data.index)

def detect_regime_changes(company_data: pd.DataFrame) -> pd.Series:
    """Detect structural changes in financial behavior patterns."""
    
    regime_change_probs = []
    
    for i in range(len(company_data)):
        if i < 90:
            regime_change_probs.append(0)
            continue
        
        # Compare recent vs historical behavior
        recent_period = company_data.iloc[max(0, i-45):i+1]
        historical_period = company_data.iloc[max(0, i-90):max(0, i-45)]
        
        if len(recent_period) < 20 or len(historical_period) < 20:
            regime_change_probs.append(0)
            continue
        
        # Statistical tests for regime change
        recent_util_mean = recent_period['loan_utilization'].mean()
        historical_util_mean = historical_period['loan_utilization'].mean()
        
        recent_util_std = recent_period['loan_utilization'].std()
        historical_util_std = historical_period['loan_utilization'].std()
        
        # Detect significant changes in mean and variance
        mean_change = abs(recent_util_mean - historical_util_mean)
        variance_change = abs(recent_util_std - historical_util_std)
        
        # Regime change probability
        regime_prob = min(1, (mean_change * 5 + variance_change * 3))
        regime_change_probs.append(regime_prob)
    
    return pd.Series(regime_change_probs, index=company_data.index)

def calculate_stress_resilience(company_data: pd.DataFrame) -> pd.Series:
    """Calculate how resilient a company is to financial stress."""
    
    resilience_scores = []
    
    for i in range(len(company_data)):
        if i < 90:
            resilience_scores.append(0.5)
            continue
        
        recent_data = company_data.iloc[max(0, i-90):i+1]
        
        # Resilience factors
        deposit_buffer = recent_data['deposit_balance'].mean() / (recent_data['used_loan'].mean() + 1)
        utilization_cushion = 1 - recent_data['loan_utilization'].mean()
        stability = 1 - recent_data['loan_utilization'].std()
        
        # Normalize and combine
        buffer_score = min(1, deposit_buffer / 3)  # Normalize around 3:1 ratio
        cushion_score = max(0, utilization_cushion)
        stability_score = max(0, stability)
        
        resilience = (buffer_score * 0.4 + cushion_score * 0.3 + stability_score * 0.3)
        resilience_scores.append(resilience)
    
    return pd.Series(resilience_scores, index=company_data.index)

def calculate_strategic_behavior_features(company_data: pd.DataFrame) -> pd.DataFrame:
    """Calculate features that indicate strategic financial behavior."""
    
    if len(company_data) < 90:
        return company_data
    
    # Strategic planning indicators
    company_data['planning_horizon_score'] = calculate_planning_horizon_score(company_data)
    
    # Financial sophistication measures
    company_data['financial_sophistication'] = calculate_financial_sophistication(company_data)
    
    # Risk management indicators
    company_data['risk_management_score'] = calculate_risk_management_score(company_data)
    
    # Strategic flexibility
    company_data['strategic_flexibility'] = calculate_strategic_flexibility(company_data)
    
    return company_data

def calculate_planning_horizon_score(company_data: pd.DataFrame) -> pd.Series:
    """Calculate indicators of strategic planning sophistication."""
    
    planning_scores = []
    
    for i in range(len(company_data)):
        if i < 90:
            planning_scores.append(0.5)
            continue
        
        # Look for evidence of forward-looking behavior
        recent_data = company_data.iloc[max(0, i-90):i+1]
        
        # Companies with good planning show:
        # 1. Proactive deposit building before high utilization periods
        # 2. Smooth utilization patterns (less reactive behavior)
        # 3. Seasonal preparation
        
        # Smoothness of utilization (less reactive = more planned)
        util_smoothness = 1 - recent_data['loan_utilization'].diff().abs().mean()
        
        # Deposit accumulation before utilization spikes
        util_spikes = (recent_data['loan_utilization'].diff() > 0.1).sum()
        if util_spikes > 0:
            # Check if deposits were built up before spikes
            deposit_preparation = 0
            util_diff = recent_data['loan_utilization'].diff()
            for j in range(1, len(util_diff)):
                if util_diff.iloc[j] > 0.1:  # Utilization spike
                    # Check if deposits increased in prior 30 days
                    if j >= 30:
                        prior_deposit_trend = calculate_trend_slope(recent_data['deposit_balance'].iloc[j-30:j])
                        if prior_deposit_trend > 0:
                            deposit_preparation += 1
            
            preparation_score = deposit_preparation / util_spikes if util_spikes > 0 else 0
        else:
            preparation_score = 0.5  # No spikes to prepare for
        
        planning_score = (util_smoothness * 0.6 + preparation_score * 0.4)
        planning_scores.append(max(0, min(1, planning_score)))
    
    return pd.Series(planning_scores, index=company_data.index)

def calculate_financial_sophistication(company_data: pd.DataFrame) -> pd.Series:
    """Calculate financial sophistication based on optimization behaviors."""
    
    sophistication_scores = []
    
    for i in range(len(company_data)):
        if i < 120:
            sophistication_scores.append(0.5)
            continue
        
        recent_data = company_data.iloc[max(0, i-120):i+1]
        
        # Sophisticated companies optimize their capital structure
        avg_utilization = recent_data['loan_utilization'].mean()
        utilization_consistency = 1 - recent_data['loan_utilization'].std()
        
        # Optimal utilization range (not too high, not too low)
        utilization_optimality = 1 - abs(avg_utilization - 0.5) * 1.5
        
        # Cash management efficiency
        deposit_to_loan_ratio = recent_data['deposit_balance'].mean() / (recent_data['used_loan'].mean() + 1)
        cash_efficiency = 1 - abs(np.log(deposit_to_loan_ratio + 0.1))  # Log to handle wide ranges
        cash_efficiency = max(0, min(1, cash_efficiency))
        
        # Timing of financial decisions (correlations between deposits and utilization)
        if len(recent_data) > 30:
            correlation = recent_data['deposit_balance'].corr(recent_data['loan_utilization'])
            timing_score = 1 - abs(correlation)  # Sophisticated timing shows low correlation
        else:
            timing_score = 0.5
        
        sophistication = (
            utilization_optimality * 0.3 +
            utilization_consistency * 0.3 +
            cash_efficiency * 0.2 +
            timing_score * 0.2
        )
        
        sophistication_scores.append(max(0, min(1, sophistication)))
    
    return pd.Series(sophistication_scores, index=company_data.index)

def calculate_risk_management_score(company_data: pd.DataFrame) -> pd.Series:
    """Calculate evidence of active risk management."""
    
    risk_mgmt_scores = []
    
    for i in range(len(company_data)):
        if i < 90:
            risk_mgmt_scores.append(0.5)
            continue
        
        recent_data = company_data.iloc[max(0, i-90):i+1]
        
        # Good risk management shows:
        # 1. Maintaining cash buffers
        # 2. Avoiding extreme utilization
        # 3. Quick recovery from stress periods
        
        # Cash buffer maintenance
        min_deposit_ratio = recent_data['deposit_balance'].min() / (recent_data['used_loan'].mean() + 1)
        buffer_score = min(1, min_deposit_ratio / 0.5)  # Target 50% minimum coverage
        
        # Avoiding extreme utilization
        max_utilization = recent_data['loan_utilization'].max()
        utilization_control = 1 - max(0, (max_utilization - 0.9) * 10)  # Penalty for >90%
        
        # Recovery capability (how quickly they reduce utilization after spikes)
        recovery_scores = []
        util_values = recent_data['loan_utilization'].values
        for j in range(30, len(util_values)):
            if util_values[j-1] > 0.8:  # High utilization
                recovery_period = 0
                for k in range(j, min(j+30, len(util_values))):
                    if util_values[k] < 0.7:  # Recovered to moderate level
                        recovery_period = k - j
                        break
                
                if recovery_period > 0:
                    recovery_scores.append(max(0, 1 - recovery_period / 30))
        
        recovery_score = np.mean(recovery_scores) if recovery_scores else 0.5
        
        risk_mgmt = (buffer_score * 0.4 + utilization_control * 0.3 + recovery_score * 0.3)
        risk_mgmt_scores.append(max(0, min(1, risk_mgmt)))
    
    return pd.Series(risk_mgmt_scores, index=company_data.index)

def calculate_strategic_flexibility(company_data: pd.DataFrame) -> pd.Series:
    """Calculate strategic flexibility and adaptability."""
    
    flexibility_scores = []
    
    for i in range(len(company_data)):
        if i < 120:
            flexibility_scores.append(0.5)
            continue
        
        recent_data = company_data.iloc[max(0, i-60):i+1]
        historical_data = company_data.iloc[max(0, i-120):max(0, i-60)]
        
        # Flexibility shown through:
        # 1. Ability to adjust utilization patterns
        # 2. Responsive to changing conditions
        # 3. Maintaining options (unused credit capacity)
        
        # Utilization adaptability
        recent_util_range = recent_data['loan_utilization'].max() - recent_data['loan_utilization'].min()
        historical_util_range = historical_data['loan_utilization'].max() - historical_data['loan_utilization'].min()
        
        adaptability = min(1, (recent_util_range + historical_util_range) / 2 * 5)  # Scale to 0-1
        
        # Option preservation (maintaining unused credit)
        avg_unused_capacity = 1 - recent_data['loan_utilization'].mean()
        option_score = min(1, avg_unused_capacity * 2)  # Up to 50% unused = max score
        
        # Response speed (how quickly they adapt to changes)
        # Measure autocorrelation - lower autocorrelation = more responsive
        if len(recent_data) > 30:
            autocorr = recent_data['loan_utilization'].autocorr(lag=7)  # 7-day lag
            response_speed = 1 - abs(autocorr) if not np.isnan(autocorr) else 0.5
        else:
            response_speed = 0.5
        
        flexibility = (adaptability * 0.4 + option_score * 0.4 + response_speed * 0.2)
        flexibility_scores.append(max(0, min(1, flexibility)))
    
    return pd.Series(flexibility_scores, index=company_data.index)

def calculate_trend_slope(series: pd.Series) -> float:
    """Calculate the slope of a linear trend for a time series."""
    if len(series) < 3 or series.isna().sum() > len(series) * 0.5:
        return 0.0
    
    clean_series = series.dropna()
    if len(clean_series) < 3:
        return 0.0
    
    try:
        x = np.arange(len(clean_series))
        slope, _, _, _, _ = stats.linregress(x, clean_series.values)
        return slope
    except:
        return 0.0

def discover_business_lifecycle_patterns(df_features: pd.DataFrame) -> Dict:
    """
    Discover business lifecycle-aware risk patterns using sophisticated clustering.
    
    This enhanced function discovers patterns that consider both financial metrics
    and business lifecycle characteristics, creating more meaningful and actionable
    personas for credit risk management.
    
    Parameters:
    -----------
    df_features : pd.DataFrame
        DataFrame with comprehensive business lifecycle features
        
    Returns:
    --------
    Dict
        Dictionary containing discovered business lifecycle patterns and rules
    """
    
    print("Discovering business lifecycle-aware risk patterns...")
    
    # Prepare comprehensive feature set
    latest_data = df_features.groupby('company_id').last().reset_index()
    
    if len(latest_data) < 30:
        print("Insufficient data for pattern discovery")
        return create_default_business_lifecycle_rules()
    
    # Select features for lifecycle-aware clustering
    financial_features = [col for col in latest_data.columns if any(
        suffix in col for suffix in ['_mean_', '_vol_', '_trend_', '_change_']
    ) and col in latest_data.columns]
    
    lifecycle_features = [col for col in latest_data.columns if any(
        suffix in col for suffix in ['_score', '_ratio', '_sustainability', '_efficiency']
    ) and col in latest_data.columns]
    
    all_clustering_features = financial_features + lifecycle_features
    
    # Ensure we have sufficient features
    available_features = [f for f in all_clustering_features if f in latest_data.columns and 
                         latest_data[f].notna().sum() > len(latest_data) * 0.7]
    
    if len(available_features) < 8:
        print(f"Insufficient features for advanced clustering. Available: {len(available_features)}")
        return create_default_business_lifecycle_rules()
    
    print(f"Using {len(available_features)} features for lifecycle-aware clustering")
    
    # Prepare clustering data
    X = latest_data[available_features].fillna(0)
    
    # Remove zero-variance features
    feature_variance = X.var()
    X = X.loc[:, feature_variance > 0]
    available_features = list(X.columns)
    
    # Separate financial and lifecycle features for weighted clustering
    financial_cols = [col for col in available_features if any(
        suffix in col for suffix in ['_mean_', '_vol_', '_trend_', '_change_']
    )]
    lifecycle_cols = [col for col in available_features if col not in financial_cols]
    
    # Scale features separately to give appropriate weights
    scaler_financial = RobustScaler()
    scaler_lifecycle = RobustScaler()
    
    X_financial_scaled = scaler_financial.fit_transform(X[financial_cols]) if financial_cols else np.array([]).reshape(len(X), 0)
    X_lifecycle_scaled = scaler_lifecycle.fit_transform(X[lifecycle_cols]) if lifecycle_cols else np.array([]).reshape(len(X), 0)
    
    # Combine with configured weights
    financial_weight = ENHANCED_CONFIG['clustering']['financial_weight']
    lifecycle_weight = ENHANCED_CONFIG['clustering']['lifecycle_weight']
    
    if X_financial_scaled.shape[1] > 0 and X_lifecycle_scaled.shape[1] > 0:
        X_combined = np.hstack([
            X_financial_scaled * financial_weight,
            X_lifecycle_scaled * lifecycle_weight
        ])
    elif X_financial_scaled.shape[1] > 0:
        X_combined = X_financial_scaled
    else:
        X_combined = X_lifecycle_scaled
    
    # Perform enhanced clustering
    target_clusters = ENHANCED_CONFIG['clustering']['n_clusters']
    best_clustering_result = perform_enhanced_clustering(X_combined, target_clusters, latest_data)
    
    if best_clustering_result is None:
        print("Clustering failed, using default patterns")
        return create_default_business_lifecycle_rules()
    
    # Analyze discovered patterns with business context
    discovered_patterns = analyze_business_lifecycle_clusters(
        latest_data, best_clustering_result['labels'], available_features
    )
    
    # Generate sophisticated business rules
    business_rules = generate_business_lifecycle_rules(discovered_patterns, available_features)
    
    # Validate patterns against expected business lifecycle personas
    validation_results = validate_against_expected_personas(discovered_patterns, latest_data)
    
    return {
        'patterns': discovered_patterns,
        'rules': business_rules,
        'clustering_features': available_features,
        'validation_results': validation_results,
        'scaler_financial': scaler_financial,
        'scaler_lifecycle': scaler_lifecycle,
        'n_clusters': best_clustering_result['n_clusters'],
        'silhouette_score': best_clustering_result['silhouette_score'],
        'business_alignment_score': validation_results.get('alignment_score', 0)
    }

def perform_enhanced_clustering(X_scaled: np.ndarray, target_clusters: int, data: pd.DataFrame) -> Optional[Dict]:
    """Perform enhanced clustering with multiple algorithms and validation."""
    
    best_result = None
    best_score = -1
    
    # Try different clustering approaches
    clustering_methods = [
        ('kmeans', KMeans(n_clusters=target_clusters, random_state=42, n_init=10)),
        ('kmeans_plus', KMeans(n_clusters=target_clusters, random_state=42, n_init=20, init='k-means++')),
    ]
    
    # Also try different numbers of clusters around the target
    cluster_range = [max(3, target_clusters-2), max(3, target_clusters-1), target_clusters, 
                    min(len(data)//10, target_clusters+1), min(len(data)//8, target_clusters+2)]
    
    for method_name, base_method in clustering_methods:
        for n_clusters in cluster_range:
            try:
                if method_name.startswith('kmeans'):
                    method = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                
                labels = method.fit_predict(X_scaled)
                
                # Calculate multiple quality metrics
                if len(np.unique(labels)) > 1:
                    silhouette = silhouette_score(X_scaled, labels)
                    
                    # Business relevance score (cluster size distribution)
                    cluster_sizes = np.bincount(labels)
                    size_balance = 1 - np.std(cluster_sizes) / np.mean(cluster_sizes)
                    
                    # Combined score
                    combined_score = silhouette * 0.7 + size_balance * 0.3
                    
                    if combined_score > best_score:
                        best_score = combined_score
                        best_result = {
                            'labels': labels,
                            'n_clusters': n_clusters,
                            'method': method_name,
                            'silhouette_score': silhouette,
                            'size_balance_score': size_balance,
                            'combined_score': combined_score
                        }
                        
            except Exception as e:
                print(f"Error with {method_name} n_clusters={n_clusters}: {e}")
                continue
    
    return best_result

def analyze_business_lifecycle_clusters(data: pd.DataFrame, labels: np.ndarray, 
                                      feature_columns: List[str]) -> Dict:
    """Analyze clusters with business lifecycle context."""
    
    patterns = {}
    data_with_clusters = data.copy()
    data_with_clusters['cluster'] = labels
    
    for cluster_id in sorted(np.unique(labels)):
        cluster_data = data_with_clusters[data_with_clusters['cluster'] == cluster_id]
        
        # Calculate comprehensive cluster statistics
        cluster_stats = {}
        for feature in feature_columns:
            if feature in cluster_data.columns:
                values = cluster_data[feature].dropna()
                if len(values) > 0:
                    cluster_stats[feature] = {
                        'mean': values.mean(),
                        'std': values.std(),
                        'median': values.median(),
                        'q25': values.quantile(0.25),
                        'q75': values.quantile(0.75)
                    }
        
        # Identify distinctive characteristics
        distinctive_features = identify_business_cluster_characteristics(
            cluster_data, data_with_clusters, feature_columns
        )
        
        # Determine business lifecycle stage
        lifecycle_assessment = assess_cluster_lifecycle_stage(cluster_stats, distinctive_features)
        
        # Assess risk profile with business context
        risk_assessment = assess_business_cluster_risk(cluster_stats, lifecycle_assessment)
        
        # Generate business-oriented description
        description = generate_business_cluster_description(
            distinctive_features, lifecycle_assessment, risk_assessment
        )
        
        # Map to expected persona if possible
        expected_persona = map_to_expected_persona(lifecycle_assessment, risk_assessment, cluster_stats)
        
        patterns[f'business_pattern_{cluster_id}'] = {
            'cluster_id': cluster_id,
            'size': len(cluster_data),
            'companies': cluster_data['company_id'].tolist(),
            'statistics': cluster_stats,
            'distinctive_features': distinctive_features,
            'lifecycle_stage': lifecycle_assessment['stage'],
            'risk_profile': risk_assessment['risk_level'],
            'risk_score': risk_assessment['risk_score'],
            'description': description,
            'expected_persona': expected_persona,
            'business_characteristics': lifecycle_assessment['characteristics'],
            'key_indicators': risk_assessment['key_indicators']
        }
    
    return patterns

def identify_business_cluster_characteristics(cluster_data: pd.DataFrame,
                                           all_data: pd.DataFrame,
                                           feature_columns: List[str]) -> Dict:
    """Identify distinctive business characteristics of a cluster."""
    
    characteristics = {}
    
    for feature in feature_columns:
        if feature in cluster_data.columns and feature in all_data.columns:
            cluster_values = cluster_data[feature].dropna()
            all_values = all_data[feature].dropna()
            
            if len(cluster_values) >= 3 and len(all_values) >= 10:
                cluster_mean = cluster_values.mean()
                all_mean = all_values.mean()
                all_std = all_values.std()
                
                if all_std > 0:
                    z_score = abs(cluster_mean - all_mean) / all_std
                    
                    # More stringent threshold for business significance
                    if z_score > 1.0:  # Reduced from 1.5 to capture more patterns
                        direction = 'higher' if cluster_mean > all_mean else 'lower'
                        
                        characteristics[feature] = {
                            'cluster_mean': cluster_mean,
                            'population_mean': all_mean,
                            'z_score': z_score,
                            'direction': direction,
                            'relative_difference': (cluster_mean - all_mean) / all_mean if all_mean != 0 else 0,
                            'business_significance': classify_business_significance(feature, z_score, direction)
                        }
    
    return characteristics

def classify_business_significance(feature: str, z_score: float, direction: str) -> str:
    """Classify the business significance of a feature difference."""
    
    significance_level = "high" if z_score > 2.0 else "moderate" if z_score > 1.5 else "low"
    
    business_interpretations = {
        'util_mean': f"{significance_level} significance - {'Conservative' if direction == 'lower' else 'Aggressive'} credit usage pattern",
        'deposit_mean': f"{significance_level} significance - {'Strong' if direction == 'higher' else 'Constrained'} cash position",
        'vol': f"{significance_level} significance - {'Stable' if direction == 'lower' else 'Volatile'} operations",
        'trend': f"{significance_level} significance - {'Declining' if direction == 'lower' else 'Growing'} trajectory",
        'score': f"{significance_level} significance - {'Poor' if direction == 'lower' else 'Strong'} performance indicator"
    }
    
    for key, interpretation in business_interpretations.items():
        if key in feature:
            return interpretation
    
    return f"{significance_level} significance - Notable difference in {feature}"

def assess_cluster_lifecycle_stage(cluster_stats: Dict, distinctive_features: Dict) -> Dict:
    """Assess the business lifecycle stage of a cluster."""
    
    stage_indicators = {
        'startup_score': 0,
        'growth_score': 0,
        'maturity_score': 0,
        'declining_score': 0
    }
    
    # Analyze key indicators for each stage
    for feature, info in distinctive_features.items():
        feature_mean = info['cluster_mean']
        direction = info['direction']
        z_score = info['z_score']
        
        # Weight by significance
        weight = min(1.0, z_score / 2.0)
        
        # Startup indicators
        if 'vol' in feature and direction == 'higher':
            stage_indicators['startup_score'] += weight * 0.8
        if 'util_mean' in feature and direction == 'higher' and feature_mean > 0.6:
            stage_indicators['startup_score'] += weight * 0.6
        if 'runway' in feature or 'burn' in feature:
            stage_indicators['startup_score'] += weight * 1.0
        
        # Growth indicators
        if 'growth' in feature and direction == 'higher':
            stage_indicators['growth_score'] += weight * 1.0
        if 'trend' in feature and direction == 'higher':
            stage_indicators['growth_score'] += weight * 0.7
        if 'expansion' in feature or 'momentum' in feature:
            stage_indicators['growth_score'] += weight * 0.8
        
        # Maturity indicators
        if 'efficiency' in feature and direction == 'higher':
            stage_indicators['maturity_score'] += weight * 1.0
        if 'stability' in feature and direction == 'higher':
            stage_indicators['maturity_score'] += weight * 0.9
        if 'vol' in feature and direction == 'lower':
            stage_indicators['maturity_score'] += weight * 0.6
        if 'sophistication' in feature and direction == 'higher':
            stage_indicators['maturity_score'] += weight * 0.7
        
        # Declining indicators
        if 'distress' in feature and direction == 'higher':
            stage_indicators['declining_score'] += weight * 1.0
        if 'trend' in feature and direction == 'lower' and 'deposit' in feature:
            stage_indicators['declining_score'] += weight * 0.8
        if 'turnaround' in feature:
            stage_indicators['declining_score'] += weight * 0.6
    
    # Determine dominant stage
    max_score = max(stage_indicators.values())
    if max_score == 0:
        dominant_stage = 'Maturity'  # Default to mature
        confidence = 0.3
    else:
        dominant_stage = max(stage_indicators.items(), key=lambda x: x[1])[0].replace('_score', '').title()
        confidence = max_score / (sum(stage_indicators.values()) + 0.01)
    
    # Extract key characteristics for this stage
    characteristics = extract_stage_characteristics(dominant_stage, distinctive_features)
    
    return {
        'stage': dominant_stage,
        'confidence': confidence,
        'stage_scores': stage_indicators,
        'characteristics': characteristics
    }

def extract_stage_characteristics(stage: str, distinctive_features: Dict) -> List[str]:
    """Extract key characteristics for a business lifecycle stage."""
    
    characteristics = []
    
    stage_feature_mappings = {
        'Startup': ['high_volatility', 'high_utilization', 'growth_focus', 'cash_intensive'],
        'Growth': ['expansion_oriented', 'strategic_borrowing', 'scaling_operations', 'investment_focused'],
        'Maturity': ['operational_efficiency', 'stable_patterns', 'balanced_approach', 'established_operations'],
        'Declining': ['declining_trends', 'cash_flow_stress', 'operational_challenges', 'restructuring_needs']
    }
    
    relevant_patterns = stage_feature_mappings.get(stage, [])
    
    for feature, info in distinctive_features.items():
        direction = info['direction']
        
        # Map features to business characteristics
        if 'vol' in feature and direction == 'higher' and 'high_volatility' in relevant_patterns:
            characteristics.append('High financial volatility')
        elif 'util_mean' in feature and direction == 'higher' and 'high_utilization' in relevant_patterns:
            characteristics.append('High loan utilization')
        elif 'efficiency' in feature and direction == 'higher' and 'operational_efficiency' in relevant_patterns:
            characteristics.append('Strong operational efficiency')
        elif 'trend' in feature and direction == 'lower' and 'declining_trends' in relevant_patterns:
            characteristics.append('Declining financial trends')
        elif 'growth' in feature and direction == 'higher' and 'expansion_oriented' in relevant_patterns:
            characteristics.append('Active growth and expansion')
        elif 'stability' in feature and direction == 'higher' and 'stable_patterns' in relevant_patterns:
            characteristics.append('Stable operational patterns')
    
    # Add default characteristics if none found
    if not characteristics:
        default_characteristics = {
            'Startup': ['Emerging business patterns'],
            'Growth': ['Expansion-focused operations'],
            'Maturity': ['Established business operations'],
            'Declining': ['Challenging business conditions']
        }
        characteristics = default_characteristics.get(stage, ['Standard business profile'])
    
    return characteristics[:5]  # Limit to top 5 characteristics

def assess_business_cluster_risk(cluster_stats: Dict, lifecycle_assessment: Dict) -> Dict:
    """Assess risk level with business lifecycle context."""
    
    risk_indicators = []
    risk_score = 0
    base_stage = lifecycle_assessment['stage']
    
    # Stage-specific risk assessment
    if base_stage == 'Startup':
        # Startups have inherently higher risk
        base_risk = 2
        
        # Check startup-specific risks
        if 'runway_sustainability' in cluster_stats:
            runway = cluster_stats['runway_sustainability']['mean']
            if runway < 0.3:
                risk_score += 2
                risk_indicators.append("Low runway sustainability")
        
        if 'startup_burn_rate' in cluster_stats:
            burn_rate = cluster_stats['startup_burn_rate']['mean']
            if burn_rate > 0.1:
                risk_score += 1
                risk_indicators.append("High burn rate")
    
    elif base_stage == 'Growth':
        # Growth companies have moderate inherent risk
        base_risk = 1
        
        # Check growth sustainability
        if 'growth_efficiency_score' in cluster_stats:
            efficiency = cluster_stats['growth_efficiency_score']['mean']
            if efficiency < 0.4:
                risk_score += 1
                risk_indicators.append("Low growth efficiency")
        
        if 'expansion_sustainability' in cluster_stats:
            sustainability = cluster_stats['expansion_sustainability']['mean']
            if sustainability < 0.5:
                risk_score += 1
                risk_indicators.append("Unsustainable expansion pace")
    
    elif base_stage == 'Maturity':
        # Mature companies have lower inherent risk
        base_risk = 0
        
        # Check for mature company risks
        if 'operational_efficiency_score' in cluster_stats:
            efficiency = cluster_stats['operational_efficiency_score']['mean']
            if efficiency < 0.6:
                risk_score += 1
                risk_indicators.append("Below-average operational efficiency")
    
    else:  # Declining
        # Declining companies have high inherent risk
        base_risk = 3
        
        # Check decline severity
        if 'distress_probability' in cluster_stats:
            distress = cluster_stats['distress_probability']['mean']
            if distress > 0.7:
                risk_score += 2
                risk_indicators.append("High distress probability")
            elif distress > 0.5:
                risk_score += 1
                risk_indicators.append("Moderate distress indicators")
        
        if 'turnaround_potential' in cluster_stats:
            turnaround = cluster_stats['turnaround_potential']['mean']
            if turnaround < 0.3:
                risk_score += 1
                risk_indicators.append("Low turnaround potential")
    
    # Universal risk factors
    if 'util_mean_90d' in cluster_stats:
        util_mean = cluster_stats['util_mean_90d']['mean']
        if util_mean > 0.85:
            risk_score += 2
            risk_indicators.append(f"Very high utilization ({util_mean:.1%})")
        elif util_mean > 0.75:
            risk_score += 1
            risk_indicators.append(f"High utilization ({util_mean:.1%})")
    
    if 'util_vol_90d' in cluster_stats:
        util_vol = cluster_stats['util_vol_90d']['mean']
        if util_vol > 0.2:
            risk_score += 1
            risk_indicators.append("High utilization volatility")
    
    # Combine base risk with calculated risk
    total_risk_score = base_risk + risk_score
    
    # Map to risk levels
    if total_risk_score >= 5:
        risk_level = 'high'
    elif total_risk_score >= 3:
        risk_level = 'medium'
    elif total_risk_score >= 1:
        risk_level = 'low'
    else:
        risk_level = 'minimal'
    
    return {
        'risk_level': risk_level,
        'risk_score': total_risk_score,
        'base_stage_risk': base_risk,
        'calculated_risk': risk_score,
        'key_indicators': risk_indicators
    }

def generate_business_cluster_description(distinctive_features: Dict, 
                                        lifecycle_assessment: Dict,
                                        risk_assessment: Dict) -> str:
    """Generate comprehensive business-oriented cluster description."""
    
    stage = lifecycle_assessment['stage']
    risk_level = risk_assessment['risk_level']
    characteristics = lifecycle_assessment['characteristics']
    
    # Start with lifecycle stage and risk level
    description_parts = [
        f"{stage}-stage business with {risk_level} risk profile"
    ]
    
    # Add key business characteristics
    if characteristics:
        char_summary = ", ".join(characteristics[:3])
        description_parts.append(f"Characterized by: {char_summary}")
    
    # Add distinctive financial behaviors
    financial_behaviors = []
    for feature, info in list(distinctive_features.items())[:3]:  # Top 3 distinctive features
        direction = info['direction']
        
        if 'util_mean' in feature:
            level = "high" if direction == 'higher' else "low"
            financial_behaviors.append(f"{level} loan utilization")
        elif 'deposit' in feature and 'mean' in feature:
            level = "strong" if direction == 'higher' else "constrained"
            financial_behaviors.append(f"{level} deposit position")
        elif 'vol' in feature:
            level = "volatile" if direction == 'higher' else "stable"
            financial_behaviors.append(f"{level} financial patterns")
        elif 'trend' in feature:
            direction_desc = "growing" if direction == 'higher' else "declining"
            financial_behaviors.append(f"{direction_desc} trends")
    
    if financial_behaviors:
        description_parts.append(f"Financial profile: {', '.join(financial_behaviors)}")
    
    return ". ".join(description_parts)

def map_to_expected_persona(lifecycle_assessment: Dict, risk_assessment: Dict, 
                          cluster_stats: Dict) -> str:
    """Map discovered cluster to expected business persona."""
    
    stage = lifecycle_assessment['stage']
    risk_level = risk_assessment['risk_level']
    
    # Get utilization and deposit characteristics
    util_level = 'moderate'  # default
    deposit_level = 'moderate'  # default
    volatility_level = 'moderate'  # default
    
    if 'util_mean_90d' in cluster_stats:
        util_mean = cluster_stats['util_mean_90d']['mean']
        if util_mean > 0.7:
            util_level = 'high'
        elif util_mean < 0.4:
            util_level = 'low'
    
    if 'deposit_mean_90d' in cluster_stats:
        # This is relative, so we'll use a simple heuristic
        deposit_mean = cluster_stats['deposit_mean_90d']['mean']
        if deposit_mean > 100000:  # Above average
            deposit_level = 'high'
        elif deposit_mean < 50000:  # Below average
            deposit_level = 'low'
    
    if 'util_vol_90d' in cluster_stats:
        vol_mean = cluster_stats['util_vol_90d']['mean']
        if vol_mean > 0.12:
            volatility_level = 'high'
        elif vol_mean < 0.06:
            volatility_level = 'low'
    
    # Map to personas based on configuration
    persona_mapping = {
        ('Startup', 'high', 'high', 'low', 'high'): 'innovation_economy',
        ('Declining', 'high', 'moderate', 'low', 'high'): 'deteriorating_health',
        ('Growth', 'medium', 'high', 'moderate', 'moderate'): 'rapidly_growing',
        ('Growth', 'medium', 'high', 'high', 'moderate'): 'strategic_planner',
        ('Maturity', 'low', 'moderate', 'moderate', 'moderate'): 'cash_flow_inventory_manager',
        ('Maturity', 'medium', 'high', 'low', 'high'): 'seasonal_borrower',
        ('Maturity', 'low', 'moderate', 'high', 'low'): 'cash_flow_business',
        ('Maturity', 'low', 'low', 'high', 'low'): 'financially_conservative',
        ('Maturity', 'low', 'low', 'low', 'low'): 'conservative_operator'
    }
    
    # Try exact match first
    key = (stage, risk_level, util_level, deposit_level, volatility_level)
    if key in persona_mapping:
        return persona_mapping[key]
    
    # Fallback to stage and risk-based mapping
    stage_risk_mapping = {
        ('Startup', 'high'): 'innovation_economy',
        ('Startup', 'medium'): 'innovation_economy',
        ('Growth', 'high'): 'rapidly_growing',
        ('Growth', 'medium'): 'strategic_planner',
        ('Growth', 'low'): 'strategic_planner',
        ('Maturity', 'high'): 'seasonal_borrower',
        ('Maturity', 'medium'): 'cash_flow_inventory_manager',
        ('Maturity', 'low'): 'cash_flow_business',
        ('Maturity', 'minimal'): 'financially_conservative',
        ('Declining', 'high'): 'deteriorating_health',
        ('Declining', 'medium'): 'deteriorating_health'
    }
    
    stage_risk_key = (stage, risk_level)
    return stage_risk_mapping.get(stage_risk_key, 'cash_flow_inventory_manager')

def generate_business_lifecycle_rules(discovered_patterns: Dict, feature_columns: List[str]) -> Dict:
    """
    Generate sophisticated business lifecycle-aware rules from discovered patterns.
    
    This function creates rules that are not only statistically valid but also
    align with business lifecycle logic and credit risk best practices.
    """
    
    rules = {}
    
    for pattern_name, pattern_info in discovered_patterns.items():
        pattern_rules = []
        
        # Extract the most business-relevant distinctive features
        distinctive_features = pattern_info['distinctive_features']
        lifecycle_stage = pattern_info['lifecycle_stage']
        risk_profile = pattern_info['risk_profile']
        
        # Sort features by business relevance and statistical significance
        sorted_features = sorted(
            distinctive_features.items(),
            key=lambda x: (x[1]['z_score'] * get_business_relevance_weight(x[0], lifecycle_stage)),
            reverse=True
        )
        
        # Generate rules for top distinctive features
        top_features = sorted_features[:min(4, len(sorted_features))]  # Max 4 rules per pattern
        
        for feature_name, feature_info in top_features:
            rule = create_business_lifecycle_rule(
                feature_name, feature_info, lifecycle_stage, risk_profile
            )
            if rule:
                pattern_rules.append(rule)
        
        # Add lifecycle-specific rules
        lifecycle_rules = add_lifecycle_specific_rules(pattern_info, lifecycle_stage)
        pattern_rules.extend(lifecycle_rules)
        
        rules[pattern_name] = {
            'pattern_id': pattern_info['cluster_id'],
            'lifecycle_stage': lifecycle_stage,
            'risk_level': risk_profile,
            'expected_persona': pattern_info['expected_persona'],
            'description': pattern_info['description'],
            'business_characteristics': pattern_info['business_characteristics'],
            'rules': pattern_rules,
            'min_rules_to_match': max(1, len(pattern_rules) // 2)  # Need to match at least half
        }
    
    return rules

def get_business_relevance_weight(feature_name: str, lifecycle_stage: str) -> float:
    """Get business relevance weight for a feature based on lifecycle stage."""
    
    # Base weights for different feature types
    base_weights = {
        'util_mean': 1.0,
        'deposit_mean': 0.9,
        'util_trend': 0.8,
        'deposit_trend': 0.8,
        'util_vol': 0.7,
        'efficiency': 0.9,
        'sustainability': 0.8,
        'growth': 0.9,
        'distress': 1.0,
        'runway': 0.9
    }
    
    # Stage-specific adjustments
    stage_adjustments = {
        'Startup': {
            'runway': 1.5,
            'burn': 1.4,
            'growth': 1.3,
            'util_vol': 1.2
        },
        'Growth': {
            'growth': 1.4,
            'expansion': 1.3,
            'momentum': 1.2,
            'efficiency': 1.2
        },
        'Maturity': {
            'efficiency': 1.3,
            'stability': 1.3,
            'sophistication': 1.2,
            'util_vol': 0.8  # Lower volatility is more important
        },
        'Declining': {
            'distress': 1.5,
            'turnaround': 1.4,
            'trend': 1.3
        }
    }
    
    # Calculate base weight
    weight = 0.5  # Default weight
    for pattern, base_weight in base_weights.items():
        if pattern in feature_name:
            weight = base_weight
            break
    
    # Apply stage-specific adjustments
    adjustments = stage_adjustments.get(lifecycle_stage, {})
    for pattern, adjustment in adjustments.items():
        if pattern in feature_name:
            weight *= adjustment
            break
    
    return weight

def create_business_lifecycle_rule(feature_name: str, feature_info: Dict, 
                                 lifecycle_stage: str, risk_profile: str) -> Optional[Dict]:
    """Create a business-meaningful rule from feature analysis."""
    
    cluster_mean = feature_info['cluster_mean']
    direction = feature_info['direction']
    z_score = feature_info['z_score']
    
    # Generate threshold with business logic
    if direction == 'higher':
        # For higher values, set threshold slightly below cluster mean
        threshold = cluster_mean * 0.85
        operator = '>'
    else:
        # For lower values, set threshold slightly above cluster mean
        threshold = cluster_mean * 1.15
        operator = '<'
    
    # Adjust threshold based on business logic
    threshold = adjust_threshold_for_business_logic(feature_name, threshold, lifecycle_stage)
    
    # Generate business description
    description = generate_business_rule_description(
        feature_name, threshold, operator, lifecycle_stage, risk_profile
    )
    
    return {
        'feature': feature_name,
        'condition': f"{feature_name} {operator} {threshold:.6f}",
        'threshold': threshold,
        'operator': operator,
        'importance': min(3.0, z_score),  # Cap importance at 3.0
        'description': description,
        'lifecycle_stage': lifecycle_stage,
        'business_logic': get_business_logic_explanation(feature_name, operator, lifecycle_stage)
    }

def adjust_threshold_for_business_logic(feature_name: str, threshold: float, lifecycle_stage: str) -> float:
    """Adjust thresholds based on business logic and practical constraints."""
    
    # Utilization thresholds should be reasonable
    if 'util_mean' in feature_name:
        return max(0.1, min(0.95, threshold))
    
    # Volatility thresholds should be bounded
    elif 'vol' in feature_name:
        return max(0.01, min(0.5, threshold))
    
    # Score-based features should be 0-1 bounded
    elif 'score' in feature_name or 'ratio' in feature_name.lower():
        return max(0.0, min(1.0, threshold))
    
    # Trend features need sensible bounds
    elif 'trend' in feature_name:
        return max(-0.1, min(0.1, threshold))
    
    # Default: return as-is but handle extreme values
    else:
        if abs(threshold) > 1e6:  # Very large values
            return threshold / 1000
        elif abs(threshold) < 1e-6:  # Very small values
            return threshold * 1000
        else:
            return threshold

def generate_business_rule_description(feature_name: str, threshold: float, 
                                     operator: str, lifecycle_stage: str, 
                                     risk_profile: str) -> str:
    """Generate business-friendly rule descriptions."""
    
    # Feature to business description mapping
    feature_descriptions = {
        'util_mean_90d': 'average loan utilization over 90 days',
        'util_mean_60d': 'average loan utilization over 60 days',
        'util_mean_30d': 'average loan utilization over 30 days',
        'deposit_mean_90d': 'average deposit balance over 90 days',
        'util_trend_90d': 'loan utilization trend over 90 days',
        'deposit_trend_90d': 'deposit balance trend over 90 days',
        'util_vol_90d': 'loan utilization volatility over 90 days',
        'growth_efficiency_score': 'growth efficiency rating',
        'operational_efficiency_score': 'operational efficiency rating',
        'distress_probability': 'financial distress probability',
        'runway_sustainability': 'runway sustainability score',
        'lifecycle_consistency_score': 'business lifecycle consistency'
    }
    
    base_description = feature_descriptions.get(feature_name, feature_name.replace('_', ' '))
    direction_word = "exceeds" if operator == '>' else "falls below"
    
    # Add business context
    if lifecycle_stage == 'Startup':
        context = "for startup-stage businesses"
    elif lifecycle_stage == 'Growth':
        context = "for growth-stage businesses"
    elif lifecycle_stage == 'Maturity':
        context = "for mature businesses"
    else:
        context = "for businesses in transition"
    
    return f"Company's {base_description} {direction_word} {threshold:.3f} ({context})"

def get_business_logic_explanation(feature_name: str, operator: str, lifecycle_stage: str) -> str:
    """Get explanation of the business logic behind a rule."""
    
    explanations = {
        ('util_mean', '>', 'Startup'): "High utilization in startups indicates capital intensity and growth needs",
        ('util_mean', '>', 'Growth'): "Growth companies often require significant credit for expansion",
        ('util_mean', '>', 'Maturity'): "High utilization in mature companies may signal operational stress",
        ('util_mean', '<', 'Maturity'): "Conservative utilization indicates financial stability in mature companies",
        ('deposit_trend', '<', 'Declining'): "Declining deposits are a key indicator of business deterioration",
        ('distress_probability', '>', 'Declining'): "High distress probability indicates urgent attention needed",
        ('efficiency_score', '>', 'Maturity'): "High efficiency is expected from established businesses",
        ('growth', '>', 'Growth'): "Strong growth metrics validate expansion-stage classification"
    }
    
    # Create lookup key
    feature_type = None
    for key_part in ['util_mean', 'deposit_trend', 'distress', 'efficiency', 'growth']:
        if key_part in feature_name:
            feature_type = key_part
            break
    
    if feature_type:
        key = (feature_type, operator, lifecycle_stage)
        return explanations.get(key, f"Business logic: {feature_type} pattern for {lifecycle_stage} companies")
    
    return f"Distinctive pattern for {lifecycle_stage}-stage businesses"

def add_lifecycle_specific_rules(pattern_info: Dict, lifecycle_stage: str) -> List[Dict]:
    """Add rules specific to business lifecycle stages."""
    
    lifecycle_rules = []
    cluster_stats = pattern_info['statistics']
    
    if lifecycle_stage == 'Startup':
        # Startup-specific rules
        if 'runway_sustainability' in cluster_stats:
            runway_mean = cluster_stats['runway_sustainability']['mean']
            if runway_mean < 0.5:
                lifecycle_rules.append({
                    'feature': 'runway_sustainability',
                    'condition': f'runway_sustainability < {runway_mean * 1.2:.3f}',
                    'threshold': runway_mean * 1.2,
                    'operator': '<',
                    'importance': 2.5,
                    'description': f'Startup runway sustainability below {runway_mean * 1.2:.3f}',
                    'lifecycle_stage': lifecycle_stage,
                    'business_logic': 'Low runway sustainability indicates funding risk for startups'
                })
    
    elif lifecycle_stage == 'Growth':
        # Growth-specific rules
        if 'growth_efficiency_score' in cluster_stats:
            efficiency_mean = cluster_stats['growth_efficiency_score']['mean']
            if efficiency_mean > 0.6:
                lifecycle_rules.append({
                    'feature': 'growth_efficiency_score',
                    'condition': f'growth_efficiency_score > {efficiency_mean * 0.85:.3f}',
                    'threshold': efficiency_mean * 0.85,
                    'operator': '>',
                    'importance': 2.0,
                    'description': f'Growth efficiency exceeds {efficiency_mean * 0.85:.3f}',
                    'lifecycle_stage': lifecycle_stage,
                    'business_logic': 'High growth efficiency indicates sustainable expansion'
                })
    
    elif lifecycle_stage == 'Maturity':
        # Maturity-specific rules
        if 'operational_efficiency_score' in cluster_stats:
            efficiency_mean = cluster_stats['operational_efficiency_score']['mean']
            if efficiency_mean > 0.7:
                lifecycle_rules.append({
                    'feature': 'operational_efficiency_score',
                    'condition': f'operational_efficiency_score > {efficiency_mean * 0.9:.3f}',
                    'threshold': efficiency_mean * 0.9,
                    'operator': '>',
                    'importance': 2.0,
                    'description': f'Operational efficiency exceeds {efficiency_mean * 0.9:.3f}',
                    'lifecycle_stage': lifecycle_stage,
                    'business_logic': 'High operational efficiency expected from mature businesses'
                })
    
    else:  # Declining
        # Declining-specific rules
        if 'distress_probability' in cluster_stats:
            distress_mean = cluster_stats['distress_probability']['mean']
            if distress_mean > 0.4:
                lifecycle_rules.append({
                    'feature': 'distress_probability',
                    'condition': f'distress_probability > {distress_mean * 0.8:.3f}',
                    'threshold': distress_mean * 0.8,
                    'operator': '>',
                    'importance': 3.0,
                    'description': f'Distress probability exceeds {distress_mean * 0.8:.3f}',
                    'lifecycle_stage': lifecycle_stage,
                    'business_logic': 'High distress probability indicates severe financial challenges'
                })
    
    return lifecycle_rules

def validate_against_expected_personas(discovered_patterns: Dict, data: pd.DataFrame) -> Dict:
    """Validate discovered patterns against expected business personas."""
    
    validation_results = {
        'total_patterns': len(discovered_patterns),
        'matched_personas': 0,
        'alignment_scores': [],
        'persona_coverage': {},
        'recommendations': []
    }
    
    expected_personas = set(ENHANCED_CONFIG['risk']['business_lifecycle_personas'].keys())
    discovered_personas = set()
    
    for pattern_name, pattern_info in discovered_patterns.items():
        expected_persona = pattern_info.get('expected_persona', 'unknown')
        discovered_personas.add(expected_persona)
        
        if expected_persona in expected_personas:
            validation_results['matched_personas'] += 1
            
            # Calculate alignment score
            alignment_score = calculate_persona_alignment_score(pattern_info, expected_persona)
            validation_results['alignment_scores'].append(alignment_score)
    
    # Calculate coverage
    for persona in expected_personas:
        validation_results['persona_coverage'][persona] = persona in discovered_personas
    
    # Generate recommendations
    missing_personas = expected_personas - discovered_personas
    if missing_personas:
        validation_results['recommendations'].append(
            f"Consider collecting more data for personas: {', '.join(missing_personas)}"
        )
    
    if validation_results['alignment_scores']:
        avg_alignment = np.mean(validation_results['alignment_scores'])
        validation_results['alignment_score'] = avg_alignment
        
        if avg_alignment < 0.6:
            validation_results['recommendations'].append(
                "Low alignment with expected personas - consider reviewing business assumptions"
            )
    else:
        validation_results['alignment_score'] = 0
    
    return validation_results

def calculate_persona_alignment_score(pattern_info: Dict, expected_persona: str) -> float:
    """Calculate how well a discovered pattern aligns with the expected persona."""
    
    if expected_persona not in ENHANCED_CONFIG['risk']['business_lifecycle_personas']:
        return 0.0
    
    expected_config = ENHANCED_CONFIG['risk']['business_lifecycle_personas'][expected_persona]
    
    alignment_factors = []
    
    # Check lifecycle stage alignment
    expected_stage = expected_config['stage']
    discovered_stage = pattern_info['lifecycle_stage']
    
    stage_alignment = 1.0 if expected_stage == discovered_stage else 0.3
    alignment_factors.append(stage_alignment)
    
    # Check risk profile alignment
    expected_risk = expected_config['risk_profile']
    discovered_risk = pattern_info['risk_profile']
    
    risk_alignment = 1.0 if expected_risk == discovered_risk else 0.5
    alignment_factors.append(risk_alignment)
    
    # Check characteristic alignment
    expected_characteristics = expected_config.get('key_characteristics', [])
    discovered_characteristics = pattern_info.get('business_characteristics', [])
    
    if expected_characteristics and discovered_characteristics:
        # Simple text matching for characteristics
        matches = 0
        for expected_char in expected_characteristics:
            for discovered_char in discovered_characteristics:
                if any(word in discovered_char.lower() for word in expected_char.split('_')):
                    matches += 1
                    break
        
        char_alignment = matches / len(expected_characteristics)
        alignment_factors.append(char_alignment)
    
    return np.mean(alignment_factors)

def create_default_business_lifecycle_rules() -> Dict:
    """Create default business lifecycle rules when discovery fails."""
    
    default_rules = {}
    
    for persona_name, persona_config in ENHANCED_CONFIG['risk']['business_lifecycle_personas'].items():
        stage = persona_config['stage']
        risk_profile = persona_config['risk_profile']
        
        # Create basic rules based on persona configuration
        rules = []
        
        # Utilization rule based on loan usage level
        loan_usage = persona_config['loan_usage']
        if loan_usage == 'High':
            util_threshold = 0.7
            util_operator = '>'
        elif loan_usage == 'Low':
            util_threshold = 0.4
            util_operator = '<'
        else:  # Moderate
            util_threshold_low = 0.3
            util_threshold_high = 0.7
            # Use range rule (simplified to single condition)
            util_threshold = 0.3
            util_operator = '>'
        
        rules.append({
            'feature': 'util_mean_90d',
            'condition': f'util_mean_90d {util_operator} {util_threshold}',
            'threshold': util_threshold,
            'operator': util_operator,
            'importance': 2.0,
            'description': f'{loan_usage} loan usage pattern ({util_operator} {util_threshold:.1%})',
            'lifecycle_stage': stage,
            'business_logic': f'{stage}-stage businesses typically show {loan_usage.lower()} loan usage'
        })
        
        # Volatility rule based on cash flow variability
        cash_flow_var = persona_config['cash_flow_variability']
        if cash_flow_var == 'High':
            vol_threshold = 0.12
            vol_operator = '>'
        elif cash_flow_var == 'Low':
            vol_threshold = 0.08
            vol_operator = '<'
        else:  # Moderate
            vol_threshold = 0.06
            vol_operator = '>'
        
        rules.append({
            'feature': 'util_vol_90d',
            'condition': f'util_vol_90d {vol_operator} {vol_threshold}',
            'threshold': vol_threshold,
            'operator': vol_operator,
            'importance': 1.5,
            'description': f'{cash_flow_var} cash flow variability ({vol_operator} {vol_threshold:.3f})',
            'lifecycle_stage': stage,
            'business_logic': f'{cash_flow_var} variability typical for {persona_name.replace("_", " ")}'
        })
        
        default_rules[f'default_{persona_name}'] = {
            'pattern_id': f'default_{persona_name}',
            'lifecycle_stage': stage,
            'risk_level': risk_profile,
            'expected_persona': persona_name,
            'description': persona_config['description'],
            'business_characteristics': persona_config.get('key_characteristics', []),
            'rules': rules,
            'min_rules_to_match': 1
        }
    
    return default_rules

#######################################################
# SECTION 2: ENHANCED RULE APPLICATION WITH BUSINESS CONTEXT
#######################################################

def apply_business_lifecycle_rules_to_clients(df_prepared: pd.DataFrame, 
                                            validated_rules: Dict,
                                            confidence_threshold: float = 0.6) -> pd.DataFrame:
    """
    Apply business lifecycle-aware rules to client data for sophisticated persona tagging.
    
    This enhanced function considers business context, lifecycle stages, and provides
    more nuanced confidence scoring based on business logic alignment.
    
    Parameters:
    -----------
    df_prepared : pd.DataFrame
        Prepared client data with comprehensive business features
    validated_rules : Dict
        Validated business lifecycle rules
    confidence_threshold : float
        Minimum confidence threshold for assignments (default: 0.6)
        
    Returns:
    --------
    pd.DataFrame
        Enhanced client assignments with business context
    """
    
    print("Applying business lifecycle-aware rules for sophisticated persona tagging...")
    
    # Get the most recent data for each company
    latest_client_data = df_prepared.groupby('company_id').last().reset_index()
    
    results = []
    
    for _, client_row in tqdm(latest_client_data.iterrows(), total=len(latest_client_data), 
                             desc="Applying business lifecycle rules"):
        
        company_id = client_row['company_id']
        
        # Evaluate against all patterns
        pattern_evaluations = {}
        
        for pattern_name, pattern_rules in validated_rules.items():
            evaluation = evaluate_client_against_business_pattern(client_row, pattern_rules)
            pattern_evaluations[pattern_name] = evaluation
        
        # Determine best persona assignment with business logic
        assignment = determine_best_business_persona(pattern_evaluations, client_row, validated_rules)
        
        # Generate comprehensive business analysis
        business_analysis = generate_comprehensive_business_analysis(
            client_row, assignment, validated_rules, pattern_evaluations
        )
        
        # Apply confidence threshold
        if assignment['confidence'] >= confidence_threshold:
            assignment_status = 'confident'
        elif assignment['confidence'] >= confidence_threshold - 0.2:
            assignment_status = 'moderate'
        else:
            assignment_status = 'low_confidence'
        
        results.append({
            'company_id': company_id,
            'assigned_persona': assignment['persona'],
            'lifecycle_stage': assignment['lifecycle_stage'],
            'confidence_score': assignment['confidence'],
            'assignment_status': assignment_status,
            'risk_level': assignment['risk_level'],
            'business_characteristics': assignment['business_characteristics'],
            'matching_patterns': assignment['matching_patterns'],
            'risk_indicators': business_analysis['risk_indicators'],
            'business_strengths': business_analysis['strengths'],
            'risk_score': business_analysis['risk_score'],
            'business_explanation': business_analysis['explanation'],
            'lifecycle_consistency': business_analysis['lifecycle_consistency'],
            'strategic_recommendations': business_analysis['recommendations'],
            'assignment_date': datetime.now(),
            'pattern_scores': {k: v['confidence_score'] for k, v in pattern_evaluations.items()}
        })
    
    results_df = pd.DataFrame(results)
    
    # Generate enhanced summary
    print(f"\n🎯 BUSINESS LIFECYCLE PERSONA ASSIGNMENT SUMMARY")
    print(f"{'='*60}")
    print(f"Total companies analyzed: {len(results_df)}")
    
    print(f"\n📊 Lifecycle Stage Distribution:")
    for stage, count in results_df['lifecycle_stage'].value_counts().items():
        print(f"  {stage}: {count} companies ({count/len(results_df)*100:.1f}%)")
    
    print(f"\n🎭 Business Persona Distribution:")
    for persona, count in results_df['assigned_persona'].value_counts().head(8).items():
        clean_name = persona.replace('_', ' ').title()
        print(f"  {clean_name}: {count} companies ({count/len(results_df)*100:.1f}%)")
    
    print(f"\n⚠️ Risk Level Distribution:")
    for risk_level, count in results_df['risk_level'].value_counts().items():
        print(f"  {risk_level.upper()}: {count} companies ({count/len(results_df)*100:.1f}%)")
    
    print(f"\n🎯 Assignment Confidence:")
    for status, count in results_df['assignment_status'].value_counts().items():
        print(f"  {status.replace('_', ' ').title()}: {count} companies ({count/len(results_df)*100:.1f}%)")
    
    return results_df

def evaluate_client_against_business_pattern(client_data: pd.Series, pattern_rules: Dict) -> Dict:
    """
    Evaluate a client against a business pattern with enhanced business context.
    
    This function provides more sophisticated evaluation considering business logic,
    lifecycle stage appropriateness, and contextual factors.
    """
    
    rules = pattern_rules['rules']
    min_rules_to_match = pattern_rules.get('min_rules_to_match', 1)
    lifecycle_stage = pattern_rules['lifecycle_stage']
    
    matched_rules = []
    rule_scores = []
    business_logic_scores = []
    
    for rule in rules:
        feature_name = rule['feature']
        threshold = rule['threshold']
        operator = rule['operator']
        importance = rule.get('importance', 1.0)
        business_logic = rule.get('business_logic', '')
        
        if feature_name in client_data.index and pd.notna(client_data[feature_name]):
            client_value = client_data[feature_name]
            
            # Evaluate rule condition
            if operator == '>':
                rule_match = client_value > threshold
                if rule_match:
                    excess_ratio = (client_value - threshold) / threshold if threshold != 0 else 1
                    confidence_contribution = min(1.0, 0.6 + 0.4 * min(excess_ratio, 1))
                else:
                    confidence_contribution = max(0.0, client_value / threshold) if threshold > 0 else 0
            else:  # operator == '<'
                rule_match = client_value < threshold
                if rule_match:
                    deficit_ratio = (threshold - client_value) / threshold if threshold != 0 else 1
                    confidence_contribution = min(1.0, 0.6 + 0.4 * min(deficit_ratio, 1))
                else:
                    confidence_contribution = max(0.0, threshold / client_value) if client_value > 0 else 0
            
            if rule_match:
                matched_rules.append(rule)
            
            # Weight by importance and business logic alignment
            weighted_score = confidence_contribution * importance
            rule_scores.append(weighted_score)
            
            # Business logic alignment score
            logic_score = evaluate_business_logic_alignment(
                feature_name, client_value, lifecycle_stage, business_logic
            )
            business_logic_scores.append(logic_score)
        else:
            # Missing feature - use neutral scores
            rule_scores.append(0.5)
            business_logic_scores.append(0.5)
    
    # Calculate pattern match
    pattern_matches = len(matched_rules) >= min_rules_to_match
    
    # Enhanced confidence calculation
    if len(rule_scores) > 0:
        base_confidence = sum(rule_scores) / len(rule_scores)
        business_alignment = sum(business_logic_scores) / len(business_logic_scores)
        
        # Boost for matching more rules
        match_bonus = len(matched_rules) / len(rules) * 0.15
        
        # Business logic bonus
        logic_bonus = (business_alignment - 0.5) * 0.1
        
        final_confidence = min(1.0, base_confidence + match_bonus + logic_bonus)
    else:
        final_confidence = 0.0
    
    return {
        'matches': pattern_matches,
        'confidence_score': final_confidence,
        'matched_rules': len(matched_rules),
        'total_rules': len(rules),
        'business_alignment': sum(business_logic_scores) / len(business_logic_scores) if business_logic_scores else 0.5,
        'rule_details': matched_rules
    }

def evaluate_business_logic_alignment(feature_name: str, client_value: float, 
                                    lifecycle_stage: str, business_logic: str) -> float:
    """Evaluate how well a client's value aligns with business logic for their stage."""
    
    # Base alignment score
    alignment_score = 0.5
    
    # Stage-specific business logic
    if lifecycle_stage == 'Startup':
        if 'util' in feature_name and client_value > 0.6:
            alignment_score = 0.8  # High utilization expected for startups
        elif 'vol' in feature_name and client_value > 0.1:
            alignment_score = 0.8  # High volatility expected for startups
        elif 'runway' in feature_name and client_value < 0.5:
            alignment_score = 0.9  # Low runway sustainability is concerning for startups
            
    elif lifecycle_stage == 'Growth':
        if 'growth' in feature_name and client_value > 0.5:
            alignment_score = 0.9  # Growth metrics should be strong
        elif 'util' in feature_name and 0.4 < client_value < 0.8:
            alignment_score = 0.8  # Moderate-high utilization for growth
        elif 'efficiency' in feature_name and client_value > 0.6:
            alignment_score = 0.8  # Efficiency important for sustainable growth
            
    elif lifecycle_stage == 'Maturity':
        if 'efficiency' in feature_name and client_value > 0.7:
            alignment_score = 0.9  # High efficiency expected from mature companies
        elif 'vol' in feature_name and client_value < 0.08:
            alignment_score = 0.8  # Low volatility expected for mature companies
        elif 'stability' in feature_name and client_value > 0.7:
            alignment_score = 0.9  # High stability for mature companies
            
    else:  # Declining
        if 'distress' in feature_name and client_value > 0.4:
            alignment_score = 0.8  # Distress indicators align with declining stage
        elif 'trend' in feature_name and client_value < 0:
            alignment_score = 0.8  # Negative trends align with declining stage
    
    return alignment_score

def determine_best_business_persona(pattern_evaluations: Dict, client_data: pd.Series, 
                                  validated_rules: Dict) -> Dict:
    """
    Determine the best business persona with sophisticated business logic.
    
    This function considers not just pattern matching but also business sensibility,
    lifecycle stage consistency, and risk profile alignment.
    """
    
    # Find matching patterns
    matching_patterns = [pattern for pattern, eval_result in pattern_evaluations.items() 
                        if eval_result['matches']]
    
    if not matching_patterns:
        # No patterns match - assign conservative default
        return assign_default_business_persona(client_data)
    
    # Score all matching patterns
    pattern_scores = []
    
    for pattern in matching_patterns:
        evaluation = pattern_evaluations[pattern]
        pattern_config = validated_rules[pattern]
        
        confidence = evaluation['confidence_score']
        business_alignment = evaluation['business_alignment']
        risk_level = pattern_config['risk_level']
        lifecycle_stage = pattern_config['lifecycle_stage']
        
        # Risk priority (higher risk gets attention)
        risk_priority = {'high': 4, 'medium': 3, 'low': 2, 'minimal': 1}.get(risk_level, 1)
        
        # Business consistency bonus
        consistency_bonus = get_business_consistency_bonus(client_data, lifecycle_stage)
        
        # Combined priority score
        priority_score = (
            confidence * 0.4 +                    # Pattern match confidence
            business_alignment * 0.3 +             # Business logic alignment
            (risk_priority / 4) * 0.2 +           # Risk-based priority
            consistency_bonus * 0.1                # Business consistency
        )
        
        pattern_scores.append({
            'pattern': pattern,
            'confidence': confidence,
            'business_alignment': business_alignment,
            'risk_level': risk_level,
            'lifecycle_stage': lifecycle_stage,
            'priority_score': priority_score,
            'expected_persona': pattern_config['expected_persona']
        })
    
    # Sort by priority score
    pattern_scores.sort(key=lambda x: x['priority_score'], reverse=True)
    
    # Select best pattern
    best_pattern = pattern_scores[0]
    
    # Generate business characteristics
    business_characteristics = generate_business_characteristics(
        best_pattern, client_data
    )
    
    return {
        'persona': best_pattern['expected_persona'],
        'lifecycle_stage': best_pattern['lifecycle_stage'],
        'confidence': best_pattern['confidence'],
        'risk_level': best_pattern['risk_level'],
        'matching_patterns': matching_patterns,
        'business_characteristics': business_characteristics,
        'explanation': f"Best match: {best_pattern['pattern']} (confidence: {best_pattern['confidence']:.2f}, business alignment: {best_pattern['business_alignment']:.2f})"
    }

def get_business_consistency_bonus(client_data: pd.Series, lifecycle_stage: str) -> float:
    """Calculate bonus for business consistency based on financial metrics."""
    
    consistency_bonus = 0.0
    
    # Check utilization consistency with stage
    if 'util_mean_90d' in client_data.index:
        util_mean = client_data['util_mean_90d']
        
        if lifecycle_stage == 'Startup' and util_mean > 0.6:
            consistency_bonus += 0.3
        elif lifecycle_stage == 'Growth' and 0.4 < util_mean < 0.8:
            consistency_bonus += 0.3
        elif lifecycle_stage == 'Maturity' and 0.2 < util_mean < 0.7:
            consistency_bonus += 0.3
        elif lifecycle_stage == 'Declining' and util_mean > 0.5:
            consistency_bonus += 0.2
    
    # Check volatility consistency
    if 'util_vol_90d' in client_data.index:
        util_vol = client_data['util_vol_90d']
        
        if lifecycle_stage in ['Startup', 'Declining'] and util_vol > 0.1:
            consistency_bonus += 0.2
        elif lifecycle_stage == 'Maturity' and util_vol < 0.08:
            consistency_bonus += 0.2
    
    return min(1.0, consistency_bonus)

def assign_default_business_persona(client_data: pd.Series) -> Dict:
    """Assign default persona when no patterns match."""
    
    # Analyze basic financial characteristics
    util_mean = client_data.get('util_mean_90d', 0.5)
    util_vol = client_data.get('util_vol_90d', 0.1)
    
    # Simple heuristic assignment
    if util_mean > 0.8:
        persona = 'deteriorating_health'
        risk_level = 'high'
        stage = 'Declining'
    elif util_mean > 0.6 and util_vol > 0.15:
        persona = 'innovation_economy'
        risk_level = 'high'
        stage = 'Startup'
    elif util_mean < 0.3 and util_vol < 0.05:
        persona = 'financially_conservative'
        risk_level = 'low'
        stage = 'Maturity'
    else:
        persona = 'cash_flow_inventory_manager'
        risk_level = 'low'
        stage = 'Maturity'
    
    return {
        'persona': persona,
        'lifecycle_stage': stage,
        'confidence': 0.3,
        'risk_level': risk_level,
        'matching_patterns': [],
        'business_characteristics': ['Default assignment based on basic metrics'],
        'explanation': 'No specific patterns matched - assigned based on financial characteristics'
    }

def generate_business_characteristics(best_pattern: Dict, client_data: pd.Series) -> List[str]:
    """Generate business characteristics based on pattern match and client data."""
    
    characteristics = []
    
    # Stage-based characteristics
    stage = best_pattern['lifecycle_stage']
    if stage == 'Startup':
        characteristics.append('Emerging business with growth potential')
        if client_data.get('util_mean_90d', 0) > 0.7:
            characteristics.append('High capital requirements')
    elif stage == 'Growth':
        characteristics.append('Actively expanding operations')
        if client_data.get('growth_efficiency_score', 0) > 0.6:
            characteristics.append('Efficient growth management')
    elif stage == 'Maturity':
        characteristics.append('Established business operations')
        if client_data.get('operational_efficiency_score', 0) > 0.7:
            characteristics.append('Strong operational efficiency')
    else:  # Declining
        characteristics.append('Business facing challenges')
        if client_data.get('turnaround_potential', 0) > 0.5:
            characteristics.append('Potential for recovery')
    
    # Financial behavior characteristics
    util_mean = client_data.get('util_mean_90d', 0)
    if util_mean > 0.8:
        characteristics.append('Heavy reliance on credit facilities')
    elif util_mean < 0.3:
        characteristics.append('Conservative credit usage')
    
    util_vol = client_data.get('util_vol_90d', 0)
    if util_vol > 0.15:
        characteristics.append('Variable financial patterns')
    elif util_vol < 0.05:
        characteristics.append('Stable financial behavior')
    
    return characteristics[:4]  # Limit to 4 characteristics

def generate_comprehensive_business_analysis(client_data: pd.Series, assignment: Dict,
                                           validated_rules: Dict, 
                                           pattern_evaluations: Dict) -> Dict:
    """Generate comprehensive business analysis with actionable insights."""
    
    risk_indicators = []
    strengths = []
    recommendations = []
    
    persona = assignment['persona']
    lifecycle_stage = assignment['lifecycle_stage']
    risk_level = assignment['risk_level']
    
    # Get persona configuration for context
    if persona in ENHANCED_CONFIG['risk']['business_lifecycle_personas']:
        persona_config = ENHANCED_CONFIG['risk']['business_lifecycle_personas'][persona]
    else:
        persona_config = {'description': 'Business profile assessment'}
    
    # Analyze key financial metrics
    util_mean = client_data.get('util_mean_90d', 0)
    util_vol = client_data.get('util_vol_90d', 0)
    deposit_trend = client_data.get('deposit_trend_90d', 0)
    
    # Risk indicators analysis
    if util_mean > 0.85:
        risk_indicators.append(f"Very high loan utilization ({util_mean:.1%})")
    elif util_mean > 0.75:
        risk_indicators.append(f"High loan utilization ({util_mean:.1%})")
    
    if util_vol > 0.2:
        risk_indicators.append(f"High financial volatility ({util_vol:.1%})")
    elif util_vol > 0.12:
        risk_indicators.append(f"Moderate financial volatility ({util_vol:.1%})")
    
    if deposit_trend < -1000:
        risk_indicators.append("Declining deposit trend")
    
    # Lifecycle-specific risks
    if lifecycle_stage == 'Startup':
        runway_sustainability = client_data.get('runway_sustainability', 0.5)
        if runway_sustainability < 0.4:
            risk_indicators.append(f"Low runway sustainability ({runway_sustainability:.1%})")
    elif lifecycle_stage == 'Growth':
        growth_efficiency = client_data.get('growth_efficiency_score', 0.5)
        if growth_efficiency < 0.5:
            risk_indicators.append(f"Below-average growth efficiency ({growth_efficiency:.1%})")
    elif lifecycle_stage == 'Declining':
        distress_prob = client_data.get('distress_probability', 0)
        if distress_prob > 0.6:
            risk_indicators.append(f"High distress probability ({distress_prob:.1%})")
    
    # Strengths analysis
    if util_mean < 0.4:
        strengths.append("Conservative credit management")
    
    if util_vol < 0.08:
        strengths.append("Stable financial patterns")
    
    if deposit_trend > 500:
        strengths.append("Growing deposit base")
    
    # Lifecycle-specific strengths
    if lifecycle_stage == 'Maturity':
        efficiency = client_data.get('operational_efficiency_score', 0)
        if efficiency > 0.7:
            strengths.append(f"Strong operational efficiency ({efficiency:.1%})")
    
    # Calculate comprehensive risk score
    risk_score = calculate_comprehensive_risk_score(
        client_data, lifecycle_stage, risk_indicators
    )
    
    # Generate recommendations
    recommendations = generate_strategic_recommendations(
        persona, lifecycle_stage, risk_level, risk_indicators, strengths
    )
    
    # Lifecycle consistency check
    lifecycle_consistency = calculate_lifecycle_consistency(
        client_data, lifecycle_stage, assignment['confidence']
    )
    
    # Generate explanation
    explanation = generate_comprehensive_explanation(
        persona, lifecycle_stage, risk_level, assignment['confidence'],
        len(risk_indicators), len(strengths)
    )
    
    return {
        'risk_indicators': risk_indicators,
        'strengths': strengths,
        'risk_score': risk_score,
        'explanation': explanation,
        'lifecycle_consistency': lifecycle_consistency,
        'recommendations': recommendations
    }

def calculate_comprehensive_risk_score(client_data: pd.Series, lifecycle_stage: str, 
                                     risk_indicators: List[str]) -> int:
    """Calculate comprehensive risk score considering business context."""
    
    base_score = len(risk_indicators)  # Start with number of risk indicators
    
    # Stage-specific adjustments
    if lifecycle_stage == 'Startup':
        base_score += 1  # Inherently riskier
    elif lifecycle_stage == 'Declining':
        base_score += 2  # Significantly riskier
    elif lifecycle_stage == 'Maturity':
        base_score = max(0, base_score - 1)  # Generally more stable
    
    # Specific metric adjustments
    util_mean = client_data.get('util_mean_90d', 0)
    if util_mean > 0.9:
        base_score += 2
    elif util_mean > 0.8:
        base_score += 1
    
    # Cap at reasonable maximum
    return min(10, max(0, base_score))

def generate_strategic_recommendations(persona: str, lifecycle_stage: str, 
                                     risk_level: str, risk_indicators: List[str],
                                     strengths: List[str]) -> List[str]:
    """Generate strategic recommendations based on comprehensive analysis."""
    
    recommendations = []
    
    # Risk-based recommendations
    if risk_level == 'high':
        recommendations.append("URGENT: Schedule immediate comprehensive review")
        recommendations.append("Consider enhanced monitoring and reporting requirements")
        if 'utilization' in ' '.join(risk_indicators).lower():
            recommendations.append("Evaluate credit limit appropriateness and usage patterns")
    elif risk_level == 'medium':
        recommendations.append("Implement enhanced monitoring procedures")
        recommendations.append("Schedule quarterly business reviews")
    
    # Lifecycle-specific recommendations
    if lifecycle_stage == 'Startup':
        recommendations.append("Monitor cash runway and funding requirements closely")
        if persona == 'innovation_economy':
            recommendations.append("Assess scalability of business model and funding strategy")
    elif lifecycle_stage == 'Growth':
        recommendations.append("Evaluate growth sustainability and financing strategy")
        recommendations.append("Monitor expansion efficiency and market positioning")
    elif lifecycle_stage == 'Maturity':
        recommendations.append("Focus on operational efficiency and competitive positioning")
        if len(strengths) > 2:
            recommendations.append("Consider opportunities for portfolio expansion")
    else:  # Declining
        recommendations.append("Develop turnaround strategy and recovery timeline")
        recommendations.append("Evaluate restructuring options and support measures")
    
    # Persona-specific recommendations
    persona_recommendations = {
        'financially_conservative': ["Explore growth financing opportunities", "Consider strategic investments"],
        'seasonal_borrower': ["Optimize seasonal financing arrangements", "Develop off-season revenue strategies"],
        'strategic_planner': ["Leverage planning capabilities for competitive advantage", "Consider strategic partnerships"],
        'deteriorating_health': ["Immediate action required", "Develop comprehensive recovery plan"]
    }
    
    if persona in persona_recommendations:
        recommendations.extend(persona_recommendations[persona])
    
    return recommendations[:6]  # Limit to 6 recommendations

def calculate_lifecycle_consistency(client_data: pd.Series, lifecycle_stage: str, 
                                  confidence: float) -> float:
    """Calculate consistency between assigned lifecycle stage and financial patterns."""
    
    consistency_factors = []
    
    # Expected patterns for each stage
    util_mean = client_data.get('util_mean_90d', 0.5)
    util_vol = client_data.get('util_vol_90d', 0.1)
    
    if lifecycle_stage == 'Startup':
        # Startups should have higher utilization and volatility
        util_consistency = 1.0 if util_mean > 0.6 else util_mean / 0.6
        vol_consistency = 1.0 if util_vol > 0.1 else util_vol / 0.1
        consistency_factors.extend([util_consistency, vol_consistency])
        
    elif lifecycle_stage == 'Growth':
        # Growth companies should have moderate-high utilization
        util_consistency = 1.0 if 0.4 < util_mean < 0.8 else (1 - abs(util_mean - 0.6) / 0.4)
        consistency_factors.append(util_consistency)
        
    elif lifecycle_stage == 'Maturity':
        # Mature companies should be more stable
        vol_consistency = 1.0 if util_vol < 0.1 else max(0, 1 - (util_vol - 0.1) / 0.1)
        consistency_factors.append(vol_consistency)
        
    else:  # Declining
        # Declining companies may have varied patterns but often higher volatility
        vol_consistency = 1.0 if util_vol > 0.08 else util_vol / 0.08
        consistency_factors.append(vol_consistency)
    
    # Add confidence as a factor
    consistency_factors.append(confidence)
    
    return sum(consistency_factors) / len(consistency_factors) if consistency_factors else 0.5

def generate_comprehensive_explanation(persona: str, lifecycle_stage: str, risk_level: str,
                                     confidence: float, risk_count: int, strength_count: int) -> str:
    """Generate comprehensive business explanation."""
    
    persona_display = persona.replace('_', ' ').title()
    
    explanation_parts = [
        f"Client classified as '{persona_display}' ({lifecycle_stage}-stage, {risk_level} risk)"
    ]
    
    explanation_parts.append(f"Assignment confidence: {confidence:.1%}")
    
    if risk_count > 0:
        explanation_parts.append(f"Identified {risk_count} risk indicator{'s' if risk_count != 1 else ''}")
    
    if strength_count > 0:
        explanation_parts.append(f"Recognized {strength_count} business strength{'s' if strength_count != 1 else ''}")
    
    # Add business context
    if lifecycle_stage == 'Startup':
        explanation_parts.append("Startup-stage businesses require close monitoring due to inherent volatility")
    elif lifecycle_stage == 'Growth':
        explanation_parts.append("Growth-stage businesses need balanced oversight of expansion and sustainability")
    elif lifecycle_stage == 'Maturity':
        explanation_parts.append("Mature businesses benefit from operational efficiency focus and strategic guidance")
    else:
        explanation_parts.append("Businesses in transition require intensive support and recovery planning")
    
    return ". ".join(explanation_parts)

def run_enhanced_business_lifecycle_pipeline(df_raw: pd.DataFrame,
                                           save_results: bool = True,
                                           output_dir: str = "business_lifecycle_output") -> Dict:
    """
    Run the complete enhanced business lifecycle-aware credit risk pipeline.
    
    This is the main orchestrator function that brings together all components
    of the sophisticated business lifecycle credit risk system.
    
    Parameters:
    -----------
    df_raw : pd.DataFrame
        Raw banking data
    save_results : bool
        Whether to save results (default: True)
    output_dir : str
        Output directory for results
        
    Returns:
    --------
    Dict
        Complete pipeline results with business insights
    """
    
    print("🚀 ENHANCED BUSINESS LIFECYCLE CREDIT RISK PIPELINE")
    print("=" * 80)
    
    import os
    if save_results:
        os.makedirs(output_dir, exist_ok=True)
    
    # Stage 1: Enhanced Feature Engineering
    print("\n📊 STAGE 1: BUSINESS LIFECYCLE FEATURE ENGINEERING")
    print("-" * 60)
    df_features = calculate_comprehensive_business_features(df_raw)
    
    if save_results:
        df_features.to_csv(f"{output_dir}/business_lifecycle_features.csv", index=False)
        print(f"💾 Saved business lifecycle features to {output_dir}/")
    
    # Stage 2: Business Pattern Discovery
    print("\n🔍 STAGE 2: BUSINESS LIFECYCLE PATTERN DISCOVERY")
    print("-" * 60)
    discovery_results = discover_business_lifecycle_patterns(df_features)
    
    if save_results:
        with open(f"{output_dir}/business_patterns.json", 'w') as f:
            serializable_results = convert_to_serializable(discovery_results)
            json.dump(serializable_results, f, indent=2)
        print(f"💾 Saved business patterns to {output_dir}/")
    
    # Stage 3: Rule Optimization and Validation
    print("\n⚙️ STAGE 3: BUSINESS RULE OPTIMIZATION")
    print("-" * 60)
    
    business_rules = discovery_results.get('rules', {})
    if not business_rules:
        print("⚠️ Using default business lifecycle rules")
        business_rules = create_default_business_lifecycle_rules()
    
    # Enhanced validation with business context
    validation_results = validate_business_rules_comprehensive(df_features, business_rules)
    
    if save_results:
        with open(f"{output_dir}/business_rules.json", 'w') as f:
            json.dump(convert_to_serializable(business_rules), f, indent=2)
        with open(f"{output_dir}/business_validation.json", 'w') as f:
            json.dump(convert_to_serializable(validation_results), f, indent=2)
    
    # Stage 4: Business Persona Assignment
    print("\n🎭 STAGE 4: BUSINESS LIFECYCLE PERSONA ASSIGNMENT")
    print("-" * 60)
    persona_results = apply_business_lifecycle_rules_to_clients(df_features, business_rules)
    
    if save_results:
        persona_results.to_csv(f"{output_dir}/business_persona_assignments.csv", index=False)
        print(f"💾 Saved persona assignments to {output_dir}/")
    
    # Stage 5: Business Intelligence Reports
    print("\n📈 STAGE 5: BUSINESS INTELLIGENCE AND REPORTING")
    print("-" * 60)
    
    # Generate executive summary
    executive_summary = generate_executive_business_summary(persona_results, business_rules)
    
    # Generate detailed business insights
    business_insights = generate_business_insights_report(persona_results, df_features)
    
    # Create portfolio risk dashboard data
    portfolio_dashboard = create_business_portfolio_dashboard(persona_results, business_rules)
    
    if save_results:
        with open(f"{output_dir}/executive_summary.txt", 'w') as f:
            f.write(executive_summary)
        with open(f"{output_dir}/business_insights.txt", 'w') as f:
            f.write(business_insights)
        with open(f"{output_dir}/portfolio_dashboard.json", 'w') as f:
            json.dump(convert_to_serializable(portfolio_dashboard), f, indent=2)
        print(f"💾 Saved business reports to {output_dir}/")
    
    # Compile comprehensive results
    final_results = {
        'business_features': df_features,
        'discovered_patterns': discovery_results,
        'business_rules': business_rules,
        'validation_results': validation_results,
        'persona_assignments': persona_results,
        'executive_summary': executive_summary,
        'business_insights': business_insights,
        'portfolio_dashboard': portfolio_dashboard,
        'pipeline_metadata': {
            'execution_date': datetime.now(),
            'total_companies': df_features['company_id'].nunique(),
            'lifecycle_stages_identified': len(persona_results['lifecycle_stage'].unique()),
            'business_personas_assigned': len(persona_results['assigned_persona'].unique()),
            'high_risk_companies': len(persona_results[persona_results['risk_level'] == 'high']),
            'average_confidence': persona_results['confidence_score'].mean(),
            'business_pipeline_version': '2.0'
        }
    }
    
    # Final summary
    metadata = final_results['pipeline_metadata']
    print(f"\n✅ ENHANCED BUSINESS LIFECYCLE PIPELINE COMPLETE!")
    print(f"📊 Analyzed {metadata['total_companies']} companies")
    print(f"🏢 Identified {metadata['lifecycle_stages_identified']} lifecycle stages")
    print(f"🎭 Assigned {metadata['business_personas_assigned']} distinct business personas")
    print(f"⚠️ Flagged {metadata['high_risk_companies']} high-risk companies")
    print(f"🎯 Average assignment confidence: {metadata['average_confidence']:.1%}")
    
    if save_results:
        print(f"\n📁 All business results saved to: {output_dir}/")
        print("📄 Key business files generated:")
        print("  • business_lifecycle_features.csv - Enhanced business features")
        print("  • business_patterns.json - Discovered business patterns")
        print("  • business_rules.json - Business lifecycle rules")
        print("  • business_persona_assignments.csv - Client assignments")
        print("  • executive_summary.txt - Executive business summary")
        print("  • business_insights.txt - Detailed business insights")
        print("  • portfolio_dashboard.json - Portfolio dashboard data")
    
    return final_results

def validate_business_rules_comprehensive(df_features: pd.DataFrame, business_rules: Dict,
                                        validation_split: float = 0.25) -> Dict:
    """Validate business rules with comprehensive business context."""
    
    print("Validating business lifecycle rules...")
    
    latest_data = df_features.groupby('company_id').last().reset_index()
    n_validation = int(len(latest_data) * validation_split)
    
    validation_companies = latest_data.sample(n=n_validation, random_state=42)['company_id']
    train_data = latest_data[~latest_data['company_id'].isin(validation_companies)]
    val_data = latest_data[latest_data['company_id'].isin(validation_companies)]
    
    validation_results = {}
    
    for pattern_name, pattern_rules in business_rules.items():
        # Apply rules to validation data
        val_predictions = apply_business_pattern_rules(val_data, pattern_rules)
        train_predictions = apply_business_pattern_rules(train_data, pattern_rules)
        
        # Calculate validation metrics
        val_match_rate = val_predictions.mean()
        train_match_rate = train_predictions.mean()
        
        # Business-specific validation
        business_validation = validate_business_logic_comprehensive(
            val_data, val_predictions, pattern_rules
        )
        
        validation_results[pattern_name] = {
            'validation_match_rate': val_match_rate,
            'training_match_rate': train_match_rate,
            'stability_score': 1 - abs(val_match_rate - train_match_rate),
            'business_validation': business_validation,
            'lifecycle_consistency': business_validation.get('lifecycle_consistency', 0.5),
            'overall_score': (
                (1 - abs(val_match_rate - train_match_rate)) * 0.4 +
                business_validation.get('score', 0.5) * 0.6
            )
        }
    
    return validation_results

def apply_business_pattern_rules(data: pd.DataFrame, pattern_rules: Dict) -> pd.Series:
    """Apply business pattern rules to data."""
    
    rules = pattern_rules['rules']
    min_rules_to_match = pattern_rules.get('min_rules_to_match', 1)
    
    rule_matches = pd.DataFrame(index=data.index)
    
    for i, rule in enumerate(rules):
        feature_name = rule['feature']
        threshold = rule['threshold']
        operator = rule['operator']
        
        if feature_name in data.columns:
            if operator == '>':
                matches = data[feature_name] > threshold
            else:
                matches = data[feature_name] < threshold
            
            rule_matches[f'rule_{i}'] = matches.fillna(False)
        else:
            rule_matches[f'rule_{i}'] = False
    
    total_matches = rule_matches.sum(axis=1)
    return total_matches >= min_rules_to_match

def validate_business_logic_comprehensive(data: pd.DataFrame, predictions: pd.Series,
                                        pattern_rules: Dict) -> Dict:
    """Comprehensive business logic validation."""
    
    matched_companies = data[predictions]
    
    if len(matched_companies) == 0:
        return {'score': 0, 'issues': ['No companies matched'], 'lifecycle_consistency': 0}
    
    issues = []
    business_score = 1.0
    
    lifecycle_stage = pattern_rules['lifecycle_stage']
    risk_level = pattern_rules['risk_level']
    
    # Lifecycle consistency validation
    if 'util_mean_90d' in matched_companies.columns:
        avg_util = matched_companies['util_mean_90d'].mean()
        
        expected_util_ranges = {
            'Startup': (0.5, 0.9),
            'Growth': (0.4, 0.8),
            'Maturity': (0.2, 0.7),
            'Declining': (0.3, 0.8)
        }
        
        expected_range = expected_util_ranges.get(lifecycle_stage, (0.2, 0.8))
        if not (expected_range[0] <= avg_util <= expected_range[1]):
            issues.append(f"Utilization ({avg_util:.1%}) inconsistent with {lifecycle_stage} stage")
            business_score -= 0.2
    
    # Risk level validation
    if risk_level == 'high':
        high_risk_indicators = 0
        if 'util_mean_90d' in matched_companies.columns and matched_companies['util_mean_90d'].mean() > 0.8:
            high_risk_indicators += 1
        if 'distress_probability' in matched_companies.columns and matched_companies['distress_probability'].mean() > 0.5:
            high_risk_indicators += 1
        
        if high_risk_indicators == 0:
            issues.append("High risk classification but no supporting indicators")
            business_score -= 0.3
    
    # Portfolio concentration check
    match_rate = len(matched_companies) / len(data)
    if match_rate > 0.4:
        issues.append(f"Pattern matches too many companies ({match_rate:.1%})")
        business_score -= 0.1
    elif match_rate < 0.02:
        issues.append(f"Pattern matches too few companies ({match_rate:.1%})")
        business_score -= 0.1
    
    lifecycle_consistency = max(0, 1 - len([i for i in issues if 'inconsistent' in i]) * 0.5)
    
    return {
        'score': max(0, business_score),
        'issues': issues,
        'lifecycle_consistency': lifecycle_consistency,
        'match_rate': match_rate
    }

def generate_executive_business_summary(persona_results: pd.DataFrame, 
                                      business_rules: Dict) -> str:
    """Generate executive summary with business insights."""
    
    summary = []
    summary.append("🏢 EXECUTIVE BUSINESS SUMMARY - CREDIT PORTFOLIO LIFECYCLE ANALYSIS")
    summary.append("=" * 80)
    summary.append(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d')}")
    summary.append(f"Portfolio Size: {len(persona_results)} companies")
    summary.append("")
    
    # Business lifecycle distribution
    summary.append("📊 BUSINESS LIFECYCLE DISTRIBUTION")
    summary.append("-" * 40)
    for stage, count in persona_results['lifecycle_stage'].value_counts().items():
        percentage = count / len(persona_results) * 100
        summary.append(f"{stage:<12}: {count:>4} companies ({percentage:>5.1f}%)")
    summary.append("")
    
    # Risk concentration by lifecycle stage
    summary.append("⚠️ RISK CONCENTRATION BY LIFECYCLE STAGE")
    summary.append("-" * 45)
    risk_by_stage = persona_results.groupby(['lifecycle_stage', 'risk_level']).size().unstack(fill_value=0)
    
    for stage in risk_by_stage.index:
        high_risk = risk_by_stage.loc[stage, 'high'] if 'high' in risk_by_stage.columns else 0
        total_stage = risk_by_stage.loc[stage].sum()
        risk_pct = (high_risk / total_stage * 100) if total_stage > 0 else 0
        summary.append(f"{stage:<12}: {high_risk:>2} high-risk / {total_stage:>3} total ({risk_pct:>4.1f}%)")
    summary.append("")
    
    # Top business personas
    summary.append("🎭 DOMINANT BUSINESS PERSONAS")
    summary.append("-" * 30)
    top_personas = persona_results['assigned_persona'].value_counts().head(6)
    for persona, count in top_personas.items():
        clean_name = persona.replace('_', ' ').title()
        percentage = count / len(persona_results) * 100
        summary.append(f"{clean_name:<25}: {count:>3} ({percentage:>4.1f}%)")
    summary.append("")
    
    # Key insights and recommendations
    summary.append("💡 KEY BUSINESS INSIGHTS")
    summary.append("-" * 25)
    
    # Startup concentration
    startup_count = len(persona_results[persona_results['lifecycle_stage'] == 'Startup'])
    startup_pct = startup_count / len(persona_results) * 100
    if startup_pct > 15:
        summary.append(f"• High startup concentration ({startup_pct:.1f}%) requires enhanced monitoring")
    
    # High-risk mature companies
    mature_high_risk = len(persona_results[
        (persona_results['lifecycle_stage'] == 'Maturity') & 
        (persona_results['risk_level'] == 'high')
    ])
    if mature_high_risk > len(persona_results) * 0.05:
        summary.append(f"• {mature_high_risk} mature companies showing high risk - investigate operational issues")
    
    # Average confidence
    avg_confidence = persona_results['confidence_score'].mean()
    if avg_confidence < 0.7:
        summary.append(f"• Lower average confidence ({avg_confidence:.1%}) suggests need for additional data")
    
    # Growth stage analysis
    growth_companies = persona_results[persona_results['lifecycle_stage'] == 'Growth']
    if len(growth_companies) > 0:
        growth_high_risk = len(growth_companies[growth_companies['risk_level'] == 'high'])
        growth_risk_pct = growth_high_risk / len(growth_companies) * 100
        if growth_risk_pct > 20:
            summary.append(f"• Growth companies show elevated risk ({growth_risk_pct:.1f}%) - monitor expansion sustainability")
    
    return "\n".join(summary)

def generate_business_insights_report(persona_results: pd.DataFrame, 
                                    df_features: pd.DataFrame) -> str:
    """Generate detailed business insights report."""
    
    report = []
    report.append("📈 DETAILED BUSINESS INSIGHTS REPORT")
    report.append("=" * 50)
    report.append("")
    
    # Lifecycle stage deep dive
    report.append("🔍 LIFECYCLE STAGE ANALYSIS")
    report.append("-" * 30)
    
    for stage in persona_results['lifecycle_stage'].unique():
        stage_data = persona_results[persona_results['lifecycle_stage'] == stage]
        report.append(f"\n{stage.upper()} STAGE ({len(stage_data)} companies):")
        
        # Risk distribution
        risk_dist = stage_data['risk_level'].value_counts()
        for risk, count in risk_dist.items():
            report.append(f"  {risk.title()} Risk: {count} companies")
        
        # Average confidence
        avg_conf = stage_data['confidence_score'].mean()
        report.append(f"  Average Confidence: {avg_conf:.1%}")
        
        # Common characteristics
        char_counts = {}
        for chars in stage_data['business_characteristics']:
            for char in chars:
                char_counts[char] = char_counts.get(char, 0) + 1
        
        if char_counts:
            top_chars = sorted(char_counts.items(), key=lambda x: x[1], reverse=True)[:3]
            report.append("  Common Characteristics:")
            for char, count in top_chars:
                report.append(f"    • {char} ({count} companies)")
    
    report.append("")
    
    # Persona insights
    report.append("🎭 BUSINESS PERSONA INSIGHTS")
    report.append("-" * 30)
    
    top_personas = persona_results['assigned_persona'].value_counts().head(5)
    for persona, count in top_personas.items():
        persona_data = persona_results[persona_results['assigned_persona'] == persona]
        report.append(f"\n{persona.replace('_', ' ').upper()} ({count} companies):")
        
        # Risk breakdown
        risk_dist = persona_data['risk_level'].value_counts()
        report.append(f"  Risk Profile: {dict(risk_dist)}")
        
        # Average risk score
        avg_risk_score = persona_data['risk_score'].mean()
        report.append(f"  Average Risk Score: {avg_risk_score:.1f}/10")
        
        # Confidence distribution
        high_conf = (persona_data['confidence_score'] >= 0.7).sum()
        report.append(f"  High Confidence Assignments: {high_conf}/{count} ({high_conf/count*100:.1f}%)")
    
    return "\n".join(report)

def create_business_portfolio_dashboard(persona_results: pd.DataFrame, 
                                      business_rules: Dict) -> Dict:
    """Create comprehensive business portfolio dashboard data."""
    
    dashboard = {
        'portfolio_overview': {},
        'lifecycle_analysis': {},
        'risk_analytics': {},
        'business_personas': {},
        'alerts_and_recommendations': [],
        'performance_metrics': {}
    }
    
    total_companies = len(persona_results)
    
    # Portfolio overview
    dashboard['portfolio_overview'] = {
        'total_companies': total_companies,
        'analysis_date': datetime.now().isoformat(),
        'average_confidence': float(persona_results['confidence_score'].mean()),
        'high_confidence_rate': float((persona_results['confidence_score'] >= 0.7).mean()),
        'total_business_personas': len(persona_results['assigned_persona'].unique())
    }
    
    # Lifecycle analysis
    lifecycle_stats = {}
    for stage in persona_results['lifecycle_stage'].unique():
        stage_data = persona_results[persona_results['lifecycle_stage'] == stage]
        lifecycle_stats[stage] = {
            'count': len(stage_data),
            'percentage': float(len(stage_data) / total_companies * 100),
            'avg_risk_score': float(stage_data['risk_score'].mean()),
            'high_risk_count': len(stage_data[stage_data['risk_level'] == 'high'])
        }
    
    dashboard['lifecycle_analysis'] = lifecycle_stats
    
    # Risk analytics
    risk_stats = {}
    for risk_level in persona_results['risk_level'].unique():
        risk_data = persona_results[persona_results['risk_level'] == risk_level]
        risk_stats[risk_level] = {
            'count': len(risk_data),
            'percentage': float(len(risk_data) / total_companies * 100),
            'avg_confidence': float(risk_data['confidence_score'].mean())
        }
    
    dashboard['risk_analytics'] = risk_stats
    
    # Business personas
    persona_stats = {}
    for persona in persona_results['assigned_persona'].value_counts().head(10).index:
        persona_data = persona_results[persona_results['assigned_persona'] == persona]
        persona_stats[persona] = {
            'count': len(persona_data),
            'percentage': float(len(persona_data) / total_companies * 100),
            'primary_lifecycle_stage': persona_data['lifecycle_stage'].mode().iloc[0],
            'avg_risk_score': float(persona_data['risk_score'].mean())
        }
    
    dashboard['business_personas'] = persona_stats
    
    # Alerts and recommendations
    alerts = []
    
    # High-risk concentration alerts
    high_risk_count = len(persona_results[persona_results['risk_level'] == 'high'])
    if high_risk_count > total_companies * 0.15:
        alerts.append({
            'level': 'HIGH',
            'type': 'risk_concentration',
            'message': f'High-risk companies ({high_risk_count}) exceed 15% of portfolio',
            'action': 'Review portfolio risk management strategy'
        })
    
    # Low confidence alerts
    low_conf_count = len(persona_results[persona_results['confidence_score'] < 0.5])
    if low_conf_count > total_companies * 0.2:
        alerts.append({
            'level': 'MEDIUM',
            'type': 'low_confidence',
            'message': f'Low confidence assignments ({low_conf_count}) exceed 20%',
            'action': 'Collect additional data for better classification'
        })
    
    # Startup concentration
    startup_count = len(persona_results[persona_results['lifecycle_stage'] == 'Startup'])
    if startup_count > total_companies * 0.2:
        alerts.append({
            'level': 'MEDIUM',
            'type': 'startup_concentration',
            'message': f'High startup concentration ({startup_count}) requires enhanced monitoring',
            'action': 'Implement startup-specific monitoring procedures'
        })
    
    dashboard['alerts_and_recommendations'] = alerts
    
    # Performance metrics
    dashboard['performance_metrics'] = {
        'rule_coverage': len([r for r in business_rules.keys()]) / 9,  # 9 expected personas
        'assignment_distribution_balance': float(1 - persona_results['assigned_persona'].value_counts().std() / 
                                                persona_results['assigned_persona'].value_counts().mean()),
        'risk_stratification_effectiveness': float(
            persona_results.groupby('risk_level')['risk_score'].mean().std()
        )
    }
    
    return dashboard

def convert_to_serializable(obj) -> Union[Dict, List, str, int, float, bool, None]:
    """Convert complex objects to JSON-serializable format."""
    if isinstance(obj, dict):
        return {key: convert_to_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(item) for item in obj]
    elif isinstance(obj, (np.ndarray, pd.Series)):
        return obj.tolist()
    elif isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    elif isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    elif hasattr(obj, '__dict__'):
        return str(obj)
    else:
        return obj

def demonstrate_enhanced_business_system():
    """Demonstrate the enhanced business lifecycle credit risk system."""
    
    print("🚀 ENHANCED BUSINESS LIFECYCLE CREDIT RISK SYSTEM DEMONSTRATION")
    print("=" * 80)
    
    # Generate sophisticated business lifecycle data
    print("\n📊 Generating sophisticated business lifecycle data...")
    synthetic_data = generate_business_lifecycle_synthetic_data(
        num_companies=100, days=730, random_seed=42
    )
    
    # Run the complete enhanced pipeline
    print("\n🔄 Running enhanced business lifecycle pipeline...")
    results = run_enhanced_business_lifecycle_pipeline(
        synthetic_data, 
        save_results=True, 
        output_dir="enhanced_business_output"
    )
    
    # Display key insights
    print("\n📈 KEY BUSINESS INSIGHTS:")
    print("-" * 40)
    
    metadata = results['pipeline_metadata']
    persona_results = results['persona_assignments']
    
    print(f"🏢 Business Portfolio Analysis:")
    print(f"  • Total Companies: {metadata['total_companies']}")
    print(f"  • Lifecycle Stages: {metadata['lifecycle_stages_identified']}")
    print(f"  • Business Personas: {metadata['business_personas_assigned']}")
    print(f"  • Average Confidence: {metadata['average_confidence']:.1%}")
    
    print(f"\n⚠️ Risk Distribution:")
    risk_dist = persona_results['risk_level'].value_counts()
    for risk, count in risk_dist.items():
        pct = count / len(persona_results) * 100
        print(f"  • {risk.title()} Risk: {count} companies ({pct:.1f}%)")
    
    print(f"\n🎭 Top Business Personas:")
    top_personas = persona_results['assigned_persona'].value_counts().head(5)
    for persona, count in top_personas.items():
        clean_name = persona.replace('_', ' ').title()
        pct = count / len(persona_results) * 100
        print(f"  • {clean_name}: {count} companies ({pct:.1f}%)")
    
    print(f"\n🏗️ Lifecycle Stage Distribution:")
    stage_dist = persona_results['lifecycle_stage'].value_counts()
    for stage, count in stage_dist.items():
        pct = count / len(persona_results) * 100
        print(f"  • {stage}: {count} companies ({pct:.1f}%)")
    
    print(f"\n✅ ENHANCED DEMONSTRATION COMPLETE!")
    print(f"📁 Check 'enhanced_business_output' directory for comprehensive results")
    
    return results

# Main execution
if __name__ == "__main__":
    print("🏢 Enhanced Business Lifecycle Credit Risk System Ready!")
    
    # Run demonstration
    demo_results = demonstrate_enhanced_business_system()
    
    print(f"\n{'='*80}")
    print("🎯 PRODUCTION-READY BUSINESS LIFECYCLE CREDIT RISK SYSTEM")
    print(f"{'='*80}")
    
    print("""
📋 TO USE WITH YOUR REAL DATA:

1. PREPARE YOUR DATA:
   • Required: company_id, date, deposit_balance, used_loan, unused_loan
   • Recommended: At least 12 months of data per company
   • Format: Clean CSV with proper date formatting

2. RUN THE PIPELINE:
   results = run_enhanced_business_lifecycle_pipeline(your_dataframe)

3. ANALYZE RESULTS:
   • Business persona assignments with lifecycle context
   • Comprehensive risk analysis with business logic
   • Executive summaries and detailed insights
   • Portfolio dashboard for ongoing monitoring

4. CUSTOMIZE FOR YOUR BUSINESS:
   • Adjust lifecycle thresholds in ENHANCED_CONFIG
   • Modify persona definitions for your industry
   • Tune confidence thresholds for your risk appetite
   • Add custom business characteristics

🏢 BUSINESS ADVANTAGES:
   ✓ Lifecycle-aware risk assessment
   ✓ Business context in every decision
   ✓ Strategic recommendations by stage
   ✓ Sophisticated persona framework
   ✓ Executive-ready reporting
   ✓ Regulatory compliance support

🔧 ADVANCED FEATURES:
   ✓ Automatic pattern discovery
   ✓ Business logic validation
   ✓ Confidence-based assignments
   ✓ Portfolio-level insights
   ✓ Strategic recommendations
   ✓ Lifecycle consistency checks
    """)

    
    