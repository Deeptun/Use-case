def find_optimal_k(data, max_k=10, random_state=42, plot=True):
    """
    Find the optimal number of clusters using silhouette score.
    
    Parameters:
    -----------
    data : numpy.ndarray
        Feature matrix to be clustered
    max_k : int, optional
        Maximum number of clusters to consider (default: 10)
    random_state : int, optional
        Random seed for KMeans (default: 42)
    plot : bool, optional
        Whether to create a plot of silhouette scores (default: True)
    
    Returns:
    --------
    int
        Optimal number of clusters
    """
    from sklearn.metrics import silhouette_score
    
    silhouette_scores = []
    k_values = range(2, min(max_k + 1, len(data)))
    
    if len(k_values) < 1:
        return 2  # Default to 2 clusters if not enough data points
    
    for k in tqdm(k_values, desc="Finding optimal K"):
        kmeans = KMeans(n_clusters=k, random_state=random_state)
        cluster_labels = kmeans.fit_predict(data)
        
        # Check if we have at least 2 clusters with data points
        if len(np.unique(cluster_labels)) < 2:
            silhouette_scores.append(0)
            continue
        
        # Calculate silhouette score
        silhouette_avg = silhouette_score(data, cluster_labels)
        silhouette_scores.append(silhouette_avg)
    
    # Find the best K
    if silhouette_scores:
        best_k = k_values[np.argmax(silhouette_scores)]
        
        # Create a plot of silhouette scores if requested
        if plot:
            plt.figure(figsize=(10, 6))
            plt.plot(k_values, silhouette_scores, 'o-', linewidth=2, markersize=8)
            plt.axvline(x=best_k, color='red', linestyle='--', label=f'Optimal k: {best_k}')
            plt.xlabel('Number of Clusters (k)', fontsize=12)
            plt.ylabel('Silhouette Score', fontsize=12)
            plt.title('Silhouette Score for Different k Values', fontsize=14)
            plt.grid(True, alpha=0.3)
            plt.legend()
            plt.tight_layout()
            plt.savefig('optimal_k_silhouette.png')
            plt.close()
            print(f"Saved silhouette score plot to optimal_k_silhouette.png")
        
        return best_k
    else:
        return 2  # Default if we couldn't calculate scores

def perform_clustering(df):
    """
    Use unsupervised learning to cluster companies based on behavioral patterns.
    Define risk personas based on cluster characteristics.
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
            
            # Enhanced metrics
            # Withdrawal frequency if available
            withdrawal_freq = company_data.get('withdrawal_count_30d', pd.Series([0] * len(company_data))).mean()
            withdrawal_size = company_data.get('withdrawal_avg_30d', pd.Series([0] * len(company_data))).mean()
            
            # Deposit concentration if available
            deposit_concentration = company_data.get('deposit_concentration_gini', pd.Series([0] * len(company_data))).mean()
            
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
                'deposit_amplitude': deposit_amplitude,
                'withdrawal_freq': withdrawal_freq,
                'withdrawal_size': withdrawal_size,
                'deposit_concentration': deposit_concentration
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
        'util_amplitude', 'deposit_amplitude', 'withdrawal_freq',
        'deposit_concentration'
    ]
    
    # Use only columns that exist in the dataframe
    feature_cols = [col for col in feature_cols if col in feature_df.columns]
    
    # Standardize features
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(feature_df[feature_cols].fillna(0))
    
    # Apply PCA for dimensionality reduction
    pca = PCA(n_components=min(5, len(feature_cols)))
    pca_result = pca.fit_transform(scaled_features)
    
    # Find optimal number of clusters using silhouette score
    optimal_k = find_optimal_k(pca_result, max_k=10, random_state=CONFIG['clustering']['random_state'])
    print(f"Optimal number of clusters determined by silhouette score: {optimal_k}")
    
    # Apply KMeans clustering with optimal K
    kmeans = KMeans(n_clusters=optimal_k, random_state=CONFIG['clustering']['random_state'])
    clusters = kmeans.fit_predict(pca_result)
    
    # Add results to feature dataframe
    feature_df['cluster'] = clusters
    
    # Add PCA components
    for i in range(pca_result.shape[1]):
        feature_df[f'pca_{i+1}'] = pca_result[:, i]
    
    # Analyze cluster characteristics and define personas
    cluster_profiles = []
    
    for cluster_id in range(optimal_k):
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
        
        # Enhanced metrics if available
        withdrawal_freq = cluster_data.get('withdrawal_freq', pd.Series([0] * len(cluster_data))).mean()
        deposit_concentration = cluster_data.get('deposit_concentration', pd.Series([0] * len(cluster_data))).mean()
        
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
        
        # Enhanced descriptions for new metrics
        withdrawal_description = ""
        if withdrawal_freq > 2:
            withdrawal_description = f" - Higher withdrawal frequency ({withdrawal_freq:.1f} per month)"
        
        concentration_description = ""
        if deposit_concentration > 0.5:
            concentration_description = f" - Higher deposit concentration (index: {deposit_concentration:.2f})"
        
        # Create cluster summary
        description = (
            f"Loan utilization {util_description}, deposits {deposit_description}, "
            f"deposit-to-loan ratio {ratio_description}{seasonal_description}"
            f"{withdrawal_description}{concentration_description}"
        )
        
        # Define risk level based on cluster characteristics
        risk_level = "low"  # Default risk level
        
        # High utilization with declining deposits indicates high risk
        if util_mean > 0.7 and deposit_trend_pct < -5:
            risk_level = "high"
        # High utilization or rapidly increasing utilization
        elif util_mean > 0.8 or (util_mean > 0.6 and util_trend_pct > 10):
            risk_level = "high"
        # Moderate utilization with negative deposit trend
        elif util_mean > 0.5 and deposit_trend_pct < -2:
            risk_level = "medium"
        # Increasing utilization with low deposit ratio
        elif util_trend_pct > 5 and deposit_loan_ratio < 1:
            risk_level = "medium"
        # High withdrawal frequency or concentration
        elif withdrawal_freq > 3 or deposit_concentration > 0.7:
            risk_level = "medium"
        
        # Define persona name based on key characteristics
        # Automatically determine persona based on cluster characteristics
        persona_name = ""
        
        if util_mean < 0.4 and deposit_trend_pct >= 0:
            persona_name = "stable_low_utilizer"
        elif util_mean > 0.8 and deposit_trend_pct < -5:
            persona_name = "high_risk_borrower"
        elif util_mean > 0.7 and deposit_trend_pct >= 0:
            persona_name = "heavy_borrower_stable_deposits"
        elif util_trend_pct > 8 and deposit_trend_pct < 2:
            persona_name = "increasing_credit_dependence"
        elif util_seasonal_pct > 40:
            persona_name = "seasonal_borrower"
        elif deposit_seasonal_pct > 40:
            persona_name = "seasonal_depositor"
        elif withdrawal_freq > 3:
            persona_name = "frequent_withdrawer"
        elif deposit_concentration > 0.7:
            persona_name = "lumpy_deposit_pattern"
        elif util_trend_pct < -5 and deposit_trend_pct > 5:
            persona_name = "improving_financial_health"
        elif volatility > 0.02:
            persona_name = "volatile_utilization_pattern"
        elif deposit_loan_ratio > 2 and util_mean < 0.5:
            persona_name = "deposit_rich_conservative"
        else:
            persona_name = f"cluster_{cluster_id}_persona"
        
        # Add cluster profile
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
            'persona': persona_name,  # Direct persona assignment from cluster
            'risk_level': risk_level  # Risk level based on cluster characteristics
        })
    
    cluster_profiles_df = pd.DataFrame(cluster_profiles)
    
    # Add persona to the feature dataframe based on cluster
    persona_map = {row['cluster']: row['persona'] for _, row in cluster_profiles_df.iterrows()}
    feature_df['persona'] = feature_df['cluster'].map(persona_map)
    
    # Add risk level to the feature dataframe based on cluster
    risk_map = {row['cluster']: row['risk_level'] for _, row in cluster_profiles_df.iterrows()}
    feature_df['risk_level'] = feature_df['cluster'].map(risk_map)
    
    # Update CONFIG with the new cluster-based personas
    # This ensures that the rest of the code knows about these new personas
    if 'CONFIG' in globals() and 'risk' in CONFIG and 'persona_patterns' in CONFIG['risk']:
        # Create descriptions for the new personas
        for _, profile in cluster_profiles_df.iterrows():
            persona = profile['persona']
            desc = profile['description']
            # Only add if it doesn't already exist
            if persona not in CONFIG['risk']['persona_patterns']:
                CONFIG['risk']['persona_patterns'][persona] = desc
    
    return feature_df, cluster_profiles_df


def plot_clusters(feature_df, cluster_profiles_df):
    """
    Visualize clusters in PCA space with detailed descriptions of the data-driven personas
    """
    if feature_df is None or 'pca_1' not in feature_df.columns:
        print("No clustering data available for visualization.")
        return None
    
    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
    
    # Get number of clusters for color map
    n_clusters = len(cluster_profiles_df)
    colors = plt.cm.viridis(np.linspace(0, 1, n_clusters))
    
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
    
    # Add labels for each cluster center and persona name
    for cluster_id, profile in cluster_profiles_df.iterrows():
        cluster_points = feature_df[feature_df['cluster'] == profile['cluster']]
        center_x = cluster_points['pca_1'].mean()
        center_y = cluster_points['pca_2'].mean()
        
        # Add a star marker at the cluster center
        ax1.scatter(center_x, center_y, marker='*', s=300, color=colors[cluster_id], edgecolors='black')
        
        # Format persona name for display (replace underscores with spaces)
        persona_display = profile['persona'].replace('_', ' ').title()
        
        # Add cluster label with persona name
        ax1.text(center_x, center_y + 0.2, 
                f"Cluster {profile['cluster']}: {persona_display}", 
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
    
    # Create a table for cluster descriptions with persona and risk level
    cluster_desc = cluster_profiles_df[['cluster', 'description', 'persona', 'risk_level']].copy()
    cluster_desc['cluster'] = 'Cluster ' + cluster_desc['cluster'].astype(str)
    cluster_desc['persona'] = cluster_desc['persona'].str.replace('_', ' ').str.title()
    cluster_desc['risk_level'] = cluster_desc['risk_level'].str.upper()
    cluster_desc.columns = ['Cluster', 'Description', 'Persona', 'Risk Level']
    
    # Create a table at the bottom
    table = plt.table(cellText=cluster_desc.values,
                     colLabels=cluster_desc.columns,
                     loc='bottom',
                     cellLoc='left',
                     colWidths=[0.1, 0.6, 0.15, 0.15],
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


# Complete Code for the Integrated Risk 
def detect_risk_patterns_with_clusters(df, feature_df, cluster_profiles_df):
    """
    Enhanced risk pattern detection that integrates K-means clustering results with temporal risk analysis.
    This function combines cluster-based personas with dynamic risk pattern detection.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        The full time-series data with metrics
    feature_df : pandas.DataFrame
        The clustering feature data with cluster assignments and personas
    cluster_profiles_df : pandas.DataFrame
        The cluster profiles with persona definitions
    
    Returns:
    --------
    tuple
        Tuple containing (risk_df, persona_df, recent_risk_summary)
    """
    risk_records = []
    persona_assignments = []
    
    # Create mapping from company_id to cluster-based persona
    company_to_cluster_persona = {}
    if feature_df is not None and 'company_id' in feature_df.columns and 'persona' in feature_df.columns:
        company_to_cluster_persona = dict(zip(feature_df['company_id'], feature_df['persona']))
    
    # Create mapping from company_id to cluster-based risk level
    company_to_cluster_risk = {}
    if feature_df is not None and 'company_id' in feature_df.columns and 'risk_level' in feature_df.columns:
        company_to_cluster_risk = dict(zip(feature_df['company_id'], feature_df['risk_level']))
    
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
        
        # Get cluster-based persona and risk level if available
        cluster_persona = company_to_cluster_persona.get(company, None)
        cluster_risk_level = company_to_cluster_risk.get(company, 'low')
        
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
            
            # Start with cluster-based persona and risk with high confidence
            persona = cluster_persona
            persona_confidence = 0.9 if persona is not None else 0.0  # High confidence for cluster-based persona
            risk_level = cluster_risk_level
            
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
                    
                    # Only override cluster persona if we're very confident in this pattern
                    # or we don't have a cluster persona
                    if persona_confidence < 0.8 or persona is None:
                        persona = "deteriorating_health"
                        persona_confidence = 0.8
                        
                    # Elevate risk level if detected pattern is more severe than cluster risk
                    if severity == "high" and risk_level != "high":
                        risk_level = severity
            
            # ---- RISK PATTERN 2: High utilization with low deposit ratio ----
            if current_util > 0.75 and current_row.get('deposit_loan_ratio', float('inf')) < 0.8:
                severity = "high" if current_util > 0.9 else "medium"
                risk_flags.append('credit_dependent')
                risk_descriptions.append(
                    f"[{severity.upper()}] Current: High loan utilization ({current_util:.1%}) "
                    f"with low deposit coverage (ratio: {current_row.get('deposit_loan_ratio', 0):.2f})"
                )
                risk_levels.append(severity)
                
                if persona_confidence < 0.7 or persona is None:
                    persona = "credit_dependent"
                    persona_confidence = 0.7
                
                if severity == "high" and risk_level != "high":
                    risk_level = severity
            
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
                    
                    if persona_confidence < 0.75 or persona is None:
                        persona = "cash_constrained"
                        persona_confidence = 0.75
                    
                    if severity == "high" and risk_level != "high":
                        risk_level = severity
            
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
                            
                            if persona_confidence < 0.6 or persona is None:
                                persona = "aggressive_expansion"
                                persona_confidence = 0.6
                            
                            if "medium" not in risk_level and risk_level != "high":
                                risk_level = "medium"
            
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
                    
                    if persona_confidence < 0.65 or persona is None:
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
                    
                    if persona_confidence < 0.65 and persona != "seasonal_loan_user" or persona is None:
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
                    
                    if "medium" not in risk_level and risk_level != "high":
                        risk_level = "medium"
            
            # ---- NEW RISK PATTERN 8: Loan utilization increasing but deposits stagnant ----
            if not pd.isna(current_row.get('util_change_90d')) and not pd.isna(current_row.get('deposit_change_90d')):
                util_change_90d = current_row['util_change_90d']
                deposit_change_90d = current_row['deposit_change_90d']
                
                if util_change_90d > 0.08 and abs(deposit_change_90d) < 0.02:
                    severity = "medium" if util_change_90d > 0.15 else "low"
                    risk_flags.append('stagnant_growth')
                    risk_descriptions.append(
                        f"[{severity.upper()}] 90d: Increasing utilization (+{util_change_90d:.1%}) "
                        f"with stagnant deposits (change: {deposit_change_90d:.1%})"
                    )
                    risk_levels.append(severity)
                    
                    if persona_confidence < 0.7 or persona is None:
                        persona = "stagnant_growth"
                        persona_confidence = 0.7
                    
                    if severity == "medium" and risk_level == "low":
                        risk_level = severity
            
            # ---- NEW RISK PATTERN 9: Sudden spikes in loan utilization ----
            if i >= 7:  # Need at least 7 days of history for short-term spike detection
                # Get short-term utilization history (7 days)
                recent_utils = company_data.iloc[i-7:i+1]['loan_utilization'].values
                if len(recent_utils) >= 2:
                    # Calculate maximum day-to-day change
                    day_changes = np.diff(recent_utils)
                    max_day_change = np.max(day_changes) if len(day_changes) > 0 else 0
                    
                    if max_day_change > 0.15:  # 15% spike in a single day
                        severity = "high" if max_day_change > 0.25 else "medium"
                        risk_flags.append('utilization_spike')
                        risk_descriptions.append(
                            f"[{severity.upper()}] Recent: Sudden utilization spike detected "
                            f"(+{max_day_change:.1%} in a single day)"
                        )
                        risk_levels.append(severity)
                        
                        if persona_confidence < 0.75 or persona is None:
                            persona = "utilization_spikes"
                            persona_confidence = 0.75
                        
                        if severity == "high" and risk_level != "high":
                            risk_level = severity
            
            # ---- NEW RISK PATTERN 10: Seasonal pattern breaking ----
            if 'deposit_seasonal_deviation' in current_row and not pd.isna(current_row['deposit_seasonal_deviation']):
                seasonal_deviation = current_row['deposit_seasonal_deviation']
                
                if seasonal_deviation > 0.3:  # 30% deviation from expected seasonal pattern
                    severity = "medium" if seasonal_deviation > 0.5 else "low"
                    risk_flags.append('seasonal_break')
                    risk_descriptions.append(
                        f"[{severity.upper()}] Seasonal pattern break: "
                        f"Deposit deviation {seasonal_deviation:.1%} from expected seasonal pattern"
                    )
                    risk_levels.append(severity)
                    
                    if persona_confidence < 0.75 or persona is None:
                        persona = "seasonal_pattern_breaking"
                        persona_confidence = 0.75
                    
                    if severity == "medium" and risk_level == "low":
                        risk_level = severity
            
            # ---- NEW RISK PATTERN 11: Approaching credit limit ----
            if current_util > 0.9:
                # Look at rate of approach to limit
                if i >= 30 and 'loan_utilization' in company_data.columns:
                    past_util = company_data.iloc[i-30]['loan_utilization']
                    util_velocity = (current_util - past_util) / 30  # Daily increase
                    
                    if util_velocity > 0.002:  # More than 0.2% per day increase
                        severity = "high" if current_util > 0.95 else "medium"
                        risk_flags.append('approaching_limit')
                        risk_descriptions.append(
                            f"[{severity.upper()}] Current utilization near limit ({current_util:.1%}) "
                            f"with velocity of +{util_velocity*100:.2f}% per day"
                        )
                        risk_levels.append(severity)
                        
                        if persona_confidence < 0.85 or persona is None:
                            persona = "approaching_limit"
                            persona_confidence = 0.85
                        
                        if severity == "high" and risk_level != "high":
                            risk_level = severity
            
            # ---- NEW RISK PATTERN 12: Withdrawal intensity ----
            if 'withdrawal_count_change' in current_row and not pd.isna(current_row['withdrawal_count_change']):
                withdrawal_count_change = current_row['withdrawal_count_change']
                withdrawal_avg_change = current_row.get('withdrawal_avg_change', 0)
                
                if withdrawal_count_change > 0.5 or withdrawal_avg_change > 0.3:
                    severity = "medium" if (withdrawal_count_change > 1 or withdrawal_avg_change > 0.5) else "low"
                    risk_flags.append('withdrawal_intensive')
                    risk_descriptions.append(
                        f"[{severity.upper()}] Increased withdrawal activity: "
                        f"Count change +{withdrawal_count_change:.1%}, "
                        f"Average size change +{withdrawal_avg_change:.1%}"
                    )
                    risk_levels.append(severity)
                    
                    if persona_confidence < 0.7 or persona is None:
                        persona = "withdrawal_intensive"
                        persona_confidence = 0.7
                    
                    if severity == "medium" and risk_level == "low":
                        risk_level = severity
            
            # ---- NEW RISK PATTERN 13: Deposit concentration risk ----
            if 'deposit_concentration_gini' in current_row and not pd.isna(current_row['deposit_concentration_gini']):
                gini = current_row['deposit_concentration_gini']
                
                if gini > 0.6:
                    severity = "medium" if gini > 0.75 else "low"
                    risk_flags.append('deposit_concentration')
                    risk_descriptions.append(
                        f"[{severity.upper()}] Deposit concentration detected: "
                        f"Concentration index {gini:.2f}"
                    )
                    risk_levels.append(severity)
                    
                    if persona_confidence < 0.65 or persona is None:
                        persona = "deposit_concentration"
                        persona_confidence = 0.65
                    
                    if severity == "medium" and risk_level == "low":
                        risk_level = severity
            
            # ---- NEW RISK PATTERN 14: Deposit balance below historical low with high utilization ----
            if 'deposit_to_min_ratio' in current_row and not pd.isna(current_row['deposit_to_min_ratio']):
                min_ratio = current_row['deposit_to_min_ratio']
                
                if min_ratio < 1.1 and current_util > 0.7:
                    severity = "high" if min_ratio <= 1.0 else "medium"
                    risk_flags.append('historical_low_deposits')
                    risk_descriptions.append(
                        f"[{severity.upper()}] Deposits near historical low "
                        f"({min_ratio:.2f}x minimum) with high utilization ({current_util:.1%})"
                    )
                    risk_levels.append(severity)
                    
                    if persona_confidence < 0.8 or persona is None:
                        persona = "historical_low_deposits"
                        persona_confidence = 0.8
                    
                    if severity == "high" and risk_level != "high":
                        risk_level = severity
            
            # If still no persona assigned (no cluster persona and no rule-based detection matched),
            # fall back to default assignment based on utilization
            if persona is None:
                if current_util < 0.4:
                    persona = "cautious_borrower"
                    persona_confidence = 0.6
                elif current_util > 0.8:
                    persona = "distressed_client"
                    persona_confidence = 0.7
                else:
                    persona = "credit_dependent"
                    persona_confidence = 0.6
                
                # Set risk level based on utilization if no other risk level determined
                if not risk_levels:
                    if current_util > 0.8:
                        risk_level = "high"
                    elif current_util > 0.6:
                        risk_level = "medium"
                    else:
                        risk_level = "low"
            
            # Record risk assessment and persona assignment
            is_cluster_based = persona == cluster_persona and cluster_persona is not None
            
            risk_records.append({
                'company_id': company,
                'date': current_date,
                'risk_flags': '|'.join(risk_flags) if risk_flags else 'cluster_based',
                'risk_description': ' | '.join(risk_descriptions) if risk_descriptions else f'Cluster-based risk assessment: {risk_level}',
                'risk_level': risk_level,
                'persona': persona,
                'confidence': persona_confidence,
                'current_util': current_util,
                'current_deposit': current_deposit,
                'is_recent': current_date >= recent_cutoff,
                'cluster_based': is_cluster_based
            })
            
            # Record persona assignment for cohort analysis
            persona_assignments.append({
                'company_id': company,
                'date': current_date,
                'persona': persona,
                'confidence': persona_confidence,
                'risk_level': risk_level,
                'is_recent': current_date >= recent_cutoff,
                'cluster_based': is_cluster_based
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
                if flags != 'cluster_based':  # Skip cluster_based non-flags
                    all_flags.extend(flags.split('|'))
            
            most_common_flag = 'cluster_based'
            if all_flags:
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
                'current_deposit': latest_entry['current_deposit'],
                'cluster_based': latest_entry['cluster_based']
            })
        
        recent_risk_summary = pd.DataFrame(recent_company_risks)
        
        return risk_df, persona_df, recent_risk_summary
    else:
        # Return empty dataframes with correct columns
        risk_df = pd.DataFrame(columns=['company_id', 'date', 'risk_flags', 'risk_description', 
                                        'risk_level', 'persona', 'confidence', 'current_util', 
                                        'current_deposit', 'is_recent', 'cluster_based'])
        persona_df = pd.DataFrame(columns=['company_id', 'date', 'persona', 'confidence', 
                                          'risk_level', 'is_recent', 'cluster_based'])
        recent_risk_summary = pd.DataFrame(columns=['company_id', 'latest_date', 'most_common_flag',
                                                   'risk_level', 'persona', 'current_util', 
                                                   'current_deposit', 'cluster_based'])
        return risk_df, persona_df, recent_risk_summary


def main(df):
    """
    Main function to execute the entire analysis workflow with K-means clustering
    for data-driven persona definition that's integrated with risk detection.
    """
    print("Starting enhanced bank client risk analysis with integrated K-means clustering...")
    
    # 1. Clean data with enhanced imputation techniques
    print("\nCleaning data and applying advanced imputation...")
    df_clean, df_calc = clean_data(df, min_nonzero_pct=CONFIG['data']['min_nonzero_pct'])
    
    # 2. Add derived metrics with enhanced features, including seasonality detection
    print("\nAdding derived metrics and detecting seasonality...")
    df_with_metrics = add_derived_metrics(df_clean)
    
    # 3. Perform clustering analysis with optimal k determined by silhouette score
    print("\nPerforming K-means clustering with optimal k...")
    clustering_results = perform_clustering(df_with_metrics)
    if clustering_results is not None:
        feature_df, cluster_profiles_df = clustering_results
        print(f"Created {len(cluster_profiles_df)} clusters with data-driven personas")
        
        # Plot clusters
        print("\nVisualizing clusters...")
        cluster_fig = plot_clusters(feature_df, cluster_profiles_df)
        if cluster_fig:
            plt.savefig('client_clusters.png')
            print("Saved cluster visualization to client_clusters.png")
    else:
        feature_df = None
        cluster_profiles_df = None
    
    # 4. Detect risk patterns and assign personas, now integrated with clustering results
    print("\nDetecting risk patterns with cluster-based personas...")
    risk_df, persona_df, recent_risk_df = detect_risk_patterns_with_clusters(
        df_with_metrics, feature_df, cluster_profiles_df
    )
    print(f"Found {len(risk_df)} risk events across {risk_df['company_id'].nunique()} companies")
    print(f"Identified {len(recent_risk_df)} companies with recent risk events")
    print(f"Assigned {persona_df['persona'].nunique() if not persona_df.empty else 0} different personas")
    print(f"Cluster-based personas: {persona_df['cluster_based'].sum() / len(persona_df) * 100:.1f}% of assignments")


