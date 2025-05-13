def backtest_personas_vs_downgrades(persona_df, risk_df, downgrade_df, lookback_months=6, early_warning_days=45):
    """
    Back-test personas against credit score downgrades to evaluate prediction accuracy.
    
    This function analyzes how well the persona assignments predict actual credit downgrades.
    It identifies cases where a risky persona was assigned before a downgrade occurred,
    prioritizing early detection over recent assignments.
    
    Parameters:
    -----------
    persona_df : pandas.DataFrame
        DataFrame containing persona assignments
    risk_df : pandas.DataFrame
        DataFrame containing risk assessments
    downgrade_df : pandas.DataFrame
        DataFrame containing downgrade events
    lookback_months : int
        Number of months to look back (default: 6)
    early_warning_days : int
        Number of days before downgrade to search for earliest risky persona (default: 45)
        
    Returns:
    --------
    tuple
        (performance_df, timing_df)
        - performance_df: DataFrame with performance metrics by persona
        - timing_df: DataFrame with timing analysis
    """
    # Calculate cutoff date
    end_date = max(
        persona_df['date'].max() if not persona_df.empty else datetime.now(),
        downgrade_df['downgrade_date'].max() if not downgrade_df.empty else datetime.now()
    )
    start_date = end_date - pd.Timedelta(days=30*lookback_months)
    
    # Filter for recent data
    recent_persona_df = persona_df[persona_df['date'] >= start_date].copy()
    recent_risk_df = risk_df[risk_df['date'] >= start_date].copy()
    recent_downgrade_df = downgrade_df[downgrade_df['downgrade_date'] >= start_date].copy()
    
    # Get common companies
    common_companies = set(recent_persona_df['company_id'].unique()).intersection(
        set(recent_downgrade_df['company_id'].unique())
    )
    
    print(f"Found {len(common_companies)} companies with both persona data and downgrades")
    
    # If no common companies, return empty results
    if not common_companies:
        return pd.DataFrame(), pd.DataFrame()
    
    # Prepare results containers
    performance_data = []
    timing_data = []
    
    # Define risky personas (these are considered predictive of downgrades)
    risky_personas = [
        'deteriorating_health', 'distressed_client', 'credit_dependent',
        'cash_constrained', 'stagnant_growth', 'utilization_spikes',
        'approaching_limit', 'withdrawal_intensive', 'historical_low_deposits'
    ]
    
    # For each company with downgrades, analyze persona assignments
    for company_id in tqdm(common_companies, desc="Back-testing personas"):
        # Get company downgrades
        company_downgrades = recent_downgrade_df[recent_downgrade_df['company_id'] == company_id].sort_values('downgrade_date')
        
        # Get company persona history
        company_personas = recent_persona_df[recent_persona_df['company_id'] == company_id].sort_values('date')
        
        # Get company risk events
        company_risks = recent_risk_df[recent_risk_df['company_id'] == company_id].sort_values('date')
        
        # For each downgrade, find the earliest risky persona before it
        for _, downgrade in company_downgrades.iterrows():
            downgrade_date = downgrade['downgrade_date']
            
            # Define the early warning window (e.g., 45 days before downgrade)
            early_warning_cutoff = downgrade_date - pd.Timedelta(days=early_warning_days)
            
            # First, look for risky personas in the early warning window
            early_warning_personas = company_personas[
                (company_personas['date'] >= early_warning_cutoff) & 
                (company_personas['date'] < downgrade_date) &
                (company_personas['persona'].isin(risky_personas))
            ].sort_values('date')
            
            # If we found risky personas in the early warning window, use the earliest one
            if not early_warning_personas.empty:
                earliest_risky = early_warning_personas.iloc[0]
                before_persona = earliest_risky['persona']
                before_date = earliest_risky['date']
                before_confidence = earliest_risky['confidence']
                correctly_flagged = True  # We found a risky persona in the early window
                
                print(f"Early warning for {company_id}: Found {before_persona} "
                      f"{(downgrade_date - before_date).days} days before downgrade")
            
            # If no risky personas in early window, look for ANY persona in the broader window
            else:
                # Look for all personas before the downgrade (up to some reasonable limit)
                broader_window_cutoff = downgrade_date - pd.Timedelta(days=180)  # 6 months max
                before_personas = company_personas[
                    (company_personas['date'] >= broader_window_cutoff) & 
                    (company_personas['date'] < downgrade_date)
                ].sort_values('date')
                
                if not before_personas.empty:
                    # Use the most recent persona if no early risky personas were found
                    latest_persona = before_personas.iloc[-1]
                    before_persona = latest_persona['persona']
                    before_date = latest_persona['date']
                    before_confidence = latest_persona['confidence']
                    correctly_flagged = before_persona in risky_personas
                else:
                    # No personas found at all
                    before_persona = None
                    before_date = None
                    before_confidence = None
                    correctly_flagged = False
            
            # Find the first persona after the downgrade (for comparison)
            after_personas = company_personas[company_personas['date'] >= downgrade_date]
            after_persona = after_personas['persona'].iloc[0] if not after_personas.empty else None
            after_date = after_personas['date'].iloc[0] if not after_personas.empty else None
            
            # Calculate days before/after
            days_before = (downgrade_date - before_date).days if before_date is not None else None
            days_after = (after_date - downgrade_date).days if after_date is not None else None
            
            # Find risk level before downgrade
            risk_before = None
            if not company_risks.empty:
                risk_before_data = company_risks[company_risks['date'] < downgrade_date]
                if not risk_before_data.empty:
                    risk_before = risk_before_data['risk_level'].iloc[-1]
            
            # Record timing data with additional early warning information
            timing_data.append({
                'company_id': company_id,
                'downgrade_date': downgrade_date,
                'from_score': downgrade['from_score'],
                'to_score': downgrade['to_score'],
                'downgrade_severity': downgrade['downgrade_severity'],
                'before_persona': before_persona,
                'before_date': before_date,
                'before_confidence': before_confidence,
                'days_before': days_before,
                'after_persona': after_persona,
                'after_date': after_date,
                'days_after': days_after,
                'correctly_flagged': correctly_flagged,
                'early_warning_detected': not early_warning_personas.empty,  # New field!
                'early_warning_days': early_warning_days,  # Track the window used
                'risk_level_before': risk_before,
                'industry': downgrade['industry']
            })
    
    # Rest of the function remains the same...
    timing_df = pd.DataFrame(timing_data)
    
    # Calculate performance metrics by persona
    if not timing_df.empty:
        # Calculate statistics by persona
        persona_performance = []
        
        for persona in recent_persona_df['persona'].unique():
            # Get all downgrades where this was the persona before
            persona_downgrades = timing_df[timing_df['before_persona'] == persona]
            
            # Calculate metrics
            total_instances = len(persona_downgrades)
            
            if total_instances == 0:
                continue
                
            true_positives = persona_downgrades['correctly_flagged'].sum()
            false_negatives = total_instances - true_positives
            
            # Calculate early warning metrics
            early_warning_detected = persona_downgrades['early_warning_detected'].sum()
            early_warning_rate = early_warning_detected / total_instances if total_instances > 0 else 0
            
            # Is this a risky persona?
            is_risky_persona = persona in risky_personas
            
            # Calculate average days before downgrade
            avg_days_before = persona_downgrades['days_before'].mean()
            
            # Calculate average severity for this persona
            avg_severity = persona_downgrades['downgrade_severity'].mean()
            
            # Calculate reliability by confidence level
            high_conf_downgrades = persona_downgrades[persona_downgrades['before_confidence'] >= 0.7]
            high_conf_correctly = high_conf_downgrades['correctly_flagged'].sum() if not high_conf_downgrades.empty else 0
            high_conf_total = len(high_conf_downgrades)
            high_conf_precision = high_conf_correctly / high_conf_total if high_conf_total > 0 else 0
            
            persona_performance.append({
                'persona': persona,
                'total_downgrades': total_instances,
                'correctly_flagged': true_positives,
                'false_negatives': false_negatives,
                'precision': true_positives / total_instances if total_instances > 0 else 0,
                'early_warning_detected': early_warning_detected,  # New metric!
                'early_warning_rate': early_warning_rate,  # New metric!
                'is_risky_persona': is_risky_persona,
                'avg_days_before_downgrade': avg_days_before,
                'min_days_before': persona_downgrades['days_before'].min(),
                'max_days_before': persona_downgrades['days_before'].max(),
                'avg_severity': avg_severity,
                'high_conf_precision': high_conf_precision
            })
        
        performance_df = pd.DataFrame(persona_performance)
        
        # Add global statistics with early warning metrics
        all_downgrades = len(timing_df)
        correctly_flagged = timing_df['correctly_flagged'].sum()
        early_warnings = timing_df['early_warning_detected'].sum()
        
        global_stats = {
            'persona': 'OVERALL',
            'total_downgrades': all_downgrades,
            'correctly_flagged': correctly_flagged,
            'false_negatives': all_downgrades - correctly_flagged,
            'precision': correctly_flagged / all_downgrades if all_downgrades > 0 else 0,
            'early_warning_detected': early_warnings,
            'early_warning_rate': early_warnings / all_downgrades if all_downgrades > 0 else 0,
            'is_risky_persona': None,
            'avg_days_before_downgrade': timing_df['days_before'].mean(),
            'min_days_before': timing_df['days_before'].min(),
            'max_days_before': timing_df['days_before'].max(),
            'avg_severity': timing_df['downgrade_severity'].mean(),
            'high_conf_precision': timing_df[timing_df['before_confidence'] >= 0.7]['correctly_flagged'].mean() 
                                    if not timing_df[timing_df['before_confidence'] >= 0.7].empty else 0
        }
        
        performance_df = pd.concat([performance_df, pd.DataFrame([global_stats])], ignore_index=True)
        
        # Sort by early warning rate first, then by precision
        performance_df = performance_df.sort_values(['early_warning_rate', 'precision'], ascending=False)
    else:
        performance_df = pd.DataFrame()
    
    return performance_df, timing_df
