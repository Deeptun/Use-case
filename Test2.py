import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

def generate_sample_left_skewed_data(size=1000, min_val=1, max_val=200):
    """
    Generate heavily left-skewed data between 1 and 200
    """
    np.random.seed(42)
    # Use beta distribution with parameters for heavy left skew
    base = np.random.beta(8, 2, size)
    # Scale to our range
    data = min_val + (max_val - min_val) * base
    # Add some realistic noise while maintaining minimum at 1
    noise = np.random.normal(0, 2, size)
    data = np.clip(data + noise, min_val, max_val)
    return data

def exponential_transform(data, min_val=1, max_val=200):
    """
    Exponential transformation - very effective for heavy left skew
    """
    # Normalize data to 0-1 scale first
    normalized = (data - data.min()) / (data.max() - data.min())
    # Apply exponential transformation
    transformed = np.exp(normalized * 2)  # Multiplier 2 controls transformation strength
    # Scale back to original range
    scaled = min_val + (max_val - min_val) * (transformed - transformed.min()) / (transformed.max() - transformed.min())
    return scaled

def inverse_transform(data, min_val=1, max_val=200):
    """
    Inverse transformation (1/x) - good for heavy left skew
    """
    # Inverse transform while handling the scaling
    transformed = 1 / data
    # Scale back to original range
    scaled = min_val + (max_val - min_val) * (transformed - transformed.min()) / (transformed.max() - transformed.min())
    return scaled

def power_transform(data, power=3, min_val=1, max_val=200):
    """
    Power transformation - adjustable via power parameter
    """
    # Normalize data to 0-1 scale first
    normalized = (data - data.min()) / (data.max() - data.min())
    # Apply power transformation
    transformed = np.power(normalized, power)
    # Scale back to original range
    scaled = min_val + (max_val - min_val) * (transformed - transformed.min()) / (transformed.max() - transformed.min())
    return scaled

def calculate_skewness(data):
    """
    Calculate skewness of the data
    """
    return stats.skew(data)

def plot_transformations(original_data):
    """
    Plot original data and all transformations with statistics
    """
    # Apply transformations
    exp_data = exponential_transform(original_data)
    inv_data = inverse_transform(original_data)
    pow_data = power_transform(original_data)
    
    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(15, 15))
    fig.suptitle('Transformations for Heavily Left-Skewed Data', fontsize=16)
    
    # Helper function for plotting
    def plot_distribution(data, ax, title):
        sns.histplot(data, ax=ax, kde=True, bins=30)
        ax.axvline(np.mean(data), color='r', linestyle='--', label=f'Mean: {np.mean(data):.1f}')
        ax.axvline(np.median(data), color='g', linestyle='--', label=f'Median: {np.median(data):.1f}')
        skewness = calculate_skewness(data)
        ax.set_title(f'{title}\nSkewness: {skewness:.3f}')
        ax.legend()
    
    # Plot all distributions
    plot_distribution(original_data, axes[0,0], 'Original Data')
    plot_distribution(exp_data, axes[0,1], 'Exponential Transform')
    plot_distribution(inv_data, axes[1,0], 'Inverse Transform')
    plot_distribution(pow_data, axes[1,1], 'Power Transform (Cubic)')
    
    plt.tight_layout()
    plt.show()
    
    return {
        'Original': original_data,
        'Exponential': exp_data,
        'Inverse': inv_data,
        'Power': pow_data
    }

def get_transformation_statistics(data_dict):
    """
    Calculate and return detailed statistics for all transformations
    """
    stats_df = pd.DataFrame()
    
    for name, data in data_dict.items():
        stats_dict = {
            'Mean': np.mean(data),
            'Median': np.median(data),
            'Std': np.std(data),
            'Skewness': calculate_skewness(data),
            'IQR': np.percentile(data, 75) - np.percentile(data, 25),
            'Min': np.min(data),
            'Max': np.max(data)
        }
        stats_df[name] = pd.Series(stats_dict)
    
    return stats_df

def find_best_transformation(data_dict):
    """
    Identify the best transformation based on skewness reduction
    """
    skewness_values = {name: abs(calculate_skewness(data)) 
                      for name, data in data_dict.items()}
    
    best_transform = min(skewness_values.items(), key=lambda x: x[1])[0]
    return best_transform, skewness_values

# Example usage
if __name__ == "__main__":
    # Generate sample left-skewed data
    data = generate_sample_left_skewed_data(1000)
    
    # Apply transformations and plot
    transformed_data = plot_transformations(data)
    
    # Get statistics
    stats = get_transformation_statistics(transformed_data)
    print("\nTransformation Statistics:")
    print(stats)
    
    # Find best transformation
    best_transform, skewness_values = find_best_transformation(transformed_data)
    print(f"\nBest transformation: {best_transform}")
    print("Skewness values for each transformation:")
    for name, skew in skewness_values.items():
        print(f"{name}: {skew:.3f}")

    # Example with your own data:
    # your_data = np.array([...])  # Your left-skewed data array
    # transformed_data = plot_transformations(your_data)
    # stats = get_transformation_statistics(transformed_data)
    # best_transform, skewness_values = find_best_transformation(transformed_data)
