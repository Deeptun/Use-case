import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, QuantileTransformer
from scipy import stats
import seaborn as sns

def generate_sample_data(n_regions=5, n_tasks=1000):
    """Generate sample complexity data for different regions."""
    np.random.seed(42)
    
    # Create regions with different complexity distributions
    regions = []
    complexities = []
    
    # Region 1: Mostly low complexity (1-50)
    region1_data = np.concatenate([
        np.random.uniform(1, 50, size=int(n_tasks * 0.8)),
        np.random.uniform(51, 200, size=int(n_tasks * 0.2))
    ])
    regions.extend(['Region1'] * len(region1_data))
    complexities.extend(region1_data)
    
    # Region 2: Mostly high complexity (150-200)
    region2_data = np.concatenate([
        np.random.uniform(150, 200, size=int(n_tasks * 0.8)),
        np.random.uniform(1, 149, size=int(n_tasks * 0.2))
    ])
    regions.extend(['Region2'] * len(region2_data))
    complexities.extend(region2_data)
    
    # Region 3: Medium complexity (50-150)
    region3_data = np.random.uniform(50, 150, size=n_tasks)
    regions.extend(['Region3'] * len(region3_data))
    complexities.extend(region3_data)
    
    # Region 4: Bimodal distribution
    region4_data = np.concatenate([
        np.random.uniform(1, 50, size=int(n_tasks * 0.5)),
        np.random.uniform(150, 200, size=int(n_tasks * 0.5))
    ])
    regions.extend(['Region4'] * len(region4_data))
    complexities.extend(region4_data)
    
    # Region 5: Uniform distribution
    region5_data = np.random.uniform(1, 200, size=n_tasks)
    regions.extend(['Region5'] * len(region5_data))
    complexities.extend(region5_data)
    
    # Create DataFrame
    df = pd.DataFrame({
        'Region': regions,
        'Complexity': complexities
    })
    return df

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
    quantile_transformer = QuantileTransformer(output_distribution='uniform')
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
        'Original': scale_to_range(X.flatten()),
        **{k: v.flatten() for k, v in transformed_data.items()}
    })
    
    return transformed_df

def calculate_statistics(transformed_df):
    """Calculate statistics including skewness for each transformation."""
    stats_dict = {}
    
    for column in transformed_df.columns:
        if column != 'Region':
            region_means = transformed_df.groupby('Region')[column].mean()
            stats_dict[column] = {
                'std_dev': region_means.std(),
                'range': region_means.max() - region_means.min(),
                'iqr': np.percentile(region_means, 75) - np.percentile(region_means, 25),
                'skewness': stats.skew(transformed_df[column]),
                'kurtosis': stats.kurtosis(transformed_df[column])
            }
    
    return pd.DataFrame(stats_dict).round(4)

def plot_transformations(transformed_df):
    """Plot boxplots and distributions for each transformation method."""
    transforms = [col for col in transformed_df.columns if col != 'Region']
    n_transforms = len(transforms)
    
    # Create a figure with two subplots for each transformation
    fig, axes = plt.subplots(n_transforms, 2, figsize=(20, 4*n_transforms))
    
    for i, transform in enumerate(transforms):
        # Boxplot
        transformed_df.boxplot(column=transform, by='Region', ax=axes[i, 0])
        axes[i, 0].set_title(f'{transform} Transformation - Boxplot')
        axes[i, 0].set_ylabel('Transformed Complexity')
        
        # Distribution plot
        sns.kdeplot(data=transformed_df, x=transform, hue='Region', ax=axes[i, 1])
        axes[i, 1].set_title(f'{transform} Transformation - Distribution')
        axes[i, 1].set_xlabel('Transformed Complexity')
    
    plt.tight_layout()
    plt.show()

def main():
    # Generate sample data
    df = generate_sample_data()
    
    # Apply transformations
    transformed_df = apply_transformations(df)
    
    # Calculate statistics including skewness
    print("\nStatistics for Different Transformations:")
    stats_df = calculate_statistics(transformed_df)
    print(stats_df)
    
    # Plot transformations and distributions
    plot_transformations(transformed_df)
    
    return transformed_df, stats_df

if __name__ == "__main__":
    # transformed_df, stats_df, fig = main()
    # Generate sample data
    df = generate_sample_data()
    
    # Apply transformations
    transformed_df = apply_transformations(df)
    
    # Calculate statistics including skewness
    print("\nStatistics for Different Transformations:")
    stats_df = calculate_statistics(transformed_df)
    print(stats_df)
    
    # Plot transformations and distributions
    plot_transformations(transformed_df)
    
