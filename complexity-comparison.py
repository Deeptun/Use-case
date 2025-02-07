import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, QuantileTransformer
from scipy import stats

def generate_time_series_data(n_regions=5, n_days=90):
    """Generate time series data for complexity across regions."""
    np.random.seed(42)
    
    # Generate dates
    start_date = datetime(2024, 1, 1)
    dates = [start_date + timedelta(days=x) for x in range(n_days)]
    
    data = []
    
    # Generate complexity patterns for each region
    for region in range(1, n_regions + 1):
        base_complexity = np.random.normal(loc=100 if region > 2 else 30, 
                                         scale=20 if region > 2 else 10, 
                                         size=n_days)
        
        # Add some seasonality
        seasonality = 20 * np.sin(np.linspace(0, 4*np.pi, n_days))
        trend = np.linspace(0, 30 if region > 2 else 10, n_days)
        
        complexity = base_complexity + seasonality + trend
        complexity = np.clip(complexity, 1, 200)  # Ensure values are between 1 and 200
        
        for i, date in enumerate(dates):
            data.append({
                'Date': date,
                'Region': f'Region{region}',
                'Complexity': complexity[i]
            })
    
    return pd.DataFrame(data)

def scale_to_range(data, target_min=1, target_max=100):
    """Scale data to a specific range."""
    min_val = np.min(data)
    max_val = np.max(data)
    scaled = (data - min_val) / (max_val - min_val) * (target_max - target_min) + target_min
    return scaled

def apply_transformations(df):
    """Apply different normalization techniques to the complexity scores."""
    # Get complexity values
    X = df['Complexity'].values.reshape(-1, 1)
    
    transformed_data = {}
    
    # 1. Min-Max Scaling
    minmax_scaler = MinMaxScaler()
    transformed_data['MinMax'] = scale_to_range(minmax_scaler.fit_transform(X))
    
    # 2. Z-score standardization
    standard_scaler = StandardScaler()
    transformed_data['ZScore'] = scale_to_range(standard_scaler.fit_transform(X))
    
    # 3. Robust Scaling
    robust_scaler = RobustScaler()
    transformed_data['Robust'] = scale_to_range(robust_scaler.fit_transform(X))
    
    # 4. Log transformation
    transformed_data['Log'] = scale_to_range(np.log1p(X))
    
    # 5. Square root transformation
    transformed_data['Sqrt'] = scale_to_range(np.sqrt(X))
    
    # 6. Quantile transformation
    n_samples = len(X)
    quantile_transformer = QuantileTransformer(n_quantiles=min(n_samples, 1000),
                                             output_distribution='uniform')
    transformed_data['Quantile'] = scale_to_range(quantile_transformer.fit_transform(X))
    
    # 7. Box-Cox transformation
    boxcox_scaled, _ = stats.boxcox(X.flatten() - X.min() + 1)
    transformed_data['BoxCox'] = scale_to_range(boxcox_scaled)
    
    # 8. Inverse transformation
    transformed_data['Inverse'] = scale_to_range(1 / (X + 1))
    
    # 9. Exponential transformation
    transformed_data['Exponential'] = scale_to_range(np.exp(X / X.max()))
    
    # 10. Cubic transformation
    transformed_data['Cubic'] = scale_to_range(np.power(X, 3))
    
    # 11. Power (0.5) transformation
    transformed_data['Power_0.5'] = scale_to_range(np.power(X, 0.5))
    
    # Create transformed dataframe
    transformed_df = pd.DataFrame({
        'Region': df['Region'],
        'Date': df['Date'],
        'Original': scale_to_range(X.flatten()),
        **{k: v.flatten() for k, v in transformed_data.items()}
    })
    
    return transformed_df

def calculate_region_metrics(df, value_column):
    """Calculate metrics for each region."""
    if value_column == 'Date':
        return None
        
    region_stats = df.groupby('Region')[value_column].agg(['mean', 'std']).round(2)
    region_stats['percentage'] = (region_stats['mean'] / region_stats['mean'].sum() * 100).round(2)
    
    return region_stats

def analyze_transformations(df, transformed_df):
    """Analyze the impact of transformations on regional complexity distribution."""
    # Calculate original metrics
    original_stats = calculate_region_metrics(df, 'Complexity')
    
    results = {}
    # Calculate metrics for each transformation
    for col in transformed_df.columns:
        if col not in ['Region', 'Date']:
            trans_stats = calculate_region_metrics(transformed_df, col)
            if trans_stats is not None:
                # Calculate metrics
                results[col] = {
                    'std_deviation': trans_stats['percentage'].std(),
                    'max_range': trans_stats['percentage'].max() - trans_stats['percentage'].min(),
                    'low_complexity_improvement': (
                        trans_stats.loc[['Region1', 'Region2'], 'percentage'].mean() -
                        original_stats.loc[['Region1', 'Region2'], 'percentage'].mean()
                    )
                }
    
    results_df = pd.DataFrame(results).T
    return results_df

def plot_time_series_comparison(original_df, transformed_df, best_transformation):
    """Plot time series comparison of original vs best transformation."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12))
    
    # Plot original complexity
    for region in sorted(original_df['Region'].unique()):
        region_data = original_df[original_df['Region'] == region]
        ax1.plot(region_data['Date'], region_data['Complexity'], label=region)
    
    ax1.set_title('Original Complexity Over Time')
    ax1.set_ylabel('Complexity Score')
    ax1.legend()
    
    # Plot transformed complexity
    for region in sorted(transformed_df['Region'].unique()):
        region_data = transformed_df[transformed_df['Region'] == region]
        ax2.plot(region_data['Date'], region_data[best_transformation], label=region)
    
    ax2.set_title(f'Transformed Complexity Over Time ({best_transformation})')
    ax2.set_ylabel('Transformed Complexity Score')
    ax2.legend()
    
    plt.tight_layout()
    plt.show()

def plot_region_comparisons(original_df, transformed_df, best_transformation):
    """Plot region-wise comparison of original vs transformed complexity."""
    n_regions = len(original_df['Region'].unique())
    fig, axes = plt.subplots(n_regions, 2, figsize=(15, 4*n_regions))
    
    for i, region in enumerate(sorted(original_df['Region'].unique())):
        # Original distribution
        region_data_orig = original_df[original_df['Region'] == region]
        sns.histplot(data=region_data_orig, x='Complexity', ax=axes[i, 0], bins=30)
        axes[i, 0].set_title(f'{region} - Original Distribution')
        
        # Transformed distribution
        region_data_trans = transformed_df[transformed_df['Region'] == region]
        sns.histplot(data=region_data_trans, x=best_transformation, ax=axes[i, 1], bins=30)
        axes[i, 1].set_title(f'{region} - Transformed Distribution')
    
    plt.tight_layout()
    plt.show()

def find_best_transformation(analysis_df):
    """Find the best transformation based on improvement metrics."""
    # Create a score based on our criteria
    scores = pd.DataFrame({
        'trans': analysis_df.index,
        'score': (
            -analysis_df['std_deviation'] +  # Lower is better
            -analysis_df['max_range'] +    # Lower is better
            analysis_df['low_complexity_improvement']  # Higher is better
        )
    })
    
    return scores.sort_values('score', ascending=False).iloc[0]['trans']

def print_transformation_summary(df, transformed_df, best_trans, analysis_df):
    """Print summary of transformations and their effects."""
    print("\nOriginal Complexity Stats:")
    print(calculate_region_metrics(df, 'Complexity'))
    
    print(f"\nBest Transformation ({best_trans}) Stats:")
    print(calculate_region_metrics(transformed_df, best_trans))
    
    print("\nTransformation Analysis:")
    print(analysis_df)

def main():
    # Generate time series data
    df = generate_time_series_data()
    
    # Apply transformations
    transformed_df = apply_transformations(df)
    
    # Analyze transformations
    analysis_df = analyze_transformations(df, transformed_df)
    
    # Find best transformation
    best_trans = find_best_transformation(analysis_df)
    
    # Print summary
    print_transformation_summary(df, transformed_df, best_trans, analysis_df)
    
    # Plot comparisons
    plot_time_series_comparison(df, transformed_df, best_trans)
    plot_region_comparisons(df, transformed_df, best_trans)
    
    return df, transformed_df, analysis_df, best_trans

if __name__ == "__main__":
    # df, transformed_df, analysis_df, best_trans = main()
    
    # Generate time series data
    df = generate_time_series_data()
    
    # Apply transformations
    transformed_df = apply_transformations(df)
    
    # Analyze transformations
    analysis_df = analyze_transformations(df, transformed_df)
    
    # Find best transformation
    best_trans = find_best_transformation(analysis_df)
    
    # Print summary
    print_transformation_summary(df, transformed_df, best_trans, analysis_df)
    
    # Plot comparisons
    plot_time_series_comparison(df, transformed_df, best_trans)
    plot_region_comparisons(df, transformed_df, best_trans)
    
    # Plot all transformation
    for trans in analysis_df.index:
        plot_time_series_comparison(df, transformed_df, trans)
        # plot_region_comparisons(df, transformed_df, trans)