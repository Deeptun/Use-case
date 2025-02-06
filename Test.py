import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

# Create some realistic data (let's simulate student scores or product ratings)
np.random.seed(42)
# Generate data with a right-skewed distribution
base_data = np.concatenate([
    np.random.normal(60, 20, 300),  # bulk of scores
    np.random.normal(150, 30, 100),  # high performers
    np.random.normal(180, 10, 50)    # exceptional cases
])

# Clip to our desired range [1, 200] and ensure no zeros
data = np.clip(base_data, 1, 200)

def plot_distributions(original, transformed, title):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot original distribution
    ax1.hist(original, bins=50, alpha=0.7)
    ax1.axvline(original.mean(), color='r', linestyle='dashed', linewidth=1)
    ax1.set_title(f'Original Distribution\nMean: {original.mean():.2f}, Std: {original.std():.2f}')
    
    # Plot transformed distribution
    ax2.hist(transformed, bins=50, alpha=0.7)
    ax2.axvline(transformed.mean(), color='r', linestyle='dashed', linewidth=1)
    ax2.set_title(f'Transformed Distribution\nMean: {transformed.mean():.2f}, Std: {transformed.std():.2f}')
    
    plt.suptitle(title)
    plt.tight_layout()
    plt.show()

# Let's implement several transformation approaches:

# 1. Log-based transformation with scaling back to original range
def log_scale_transform(x, target_max=100):
    # Apply log transformation
    log_data = np.log1p(x)
    
    # Scale back to desired range while keeping minimum at 1
    scaled = 1 + (target_max - 1) * (log_data - log_data.min()) / (log_data.max() - log_data.min())
    
    return scaled

# 2. Power transformation with controlled compression
def power_transform(x, power=0.5, target_max=100):
    # Apply power transformation
    powered = np.power(x, power)
    
    # Scale to desired range while keeping minimum at 1
    scaled = 1 + (target_max - 1) * (powered - powered.min()) / (powered.max() - powered.min())
    
    return scaled

# 3. Sigmoid-based transformation
def sigmoid_transform(x, target_max=100):
    # Center and scale the data
    z_scores = stats.zscore(x)
    
    # Apply sigmoid transformation
    sigmoid = 1 / (1 + np.exp(-z_scores))
    
    # Scale to desired range while keeping minimum at 1
    scaled = 1 + (target_max - 1) * (sigmoid - sigmoid.min()) / (sigmoid.max() - sigmoid.min())
    
    return scaled

# Apply transformations
target_max = 100  # Set our desired maximum value
log_transformed = log_scale_transform(data, target_max)
power_transformed = power_transform(data, power=0.5, target_max=target_max)
sigmoid_transformed = sigmoid_transform(data, target_max)

# Plot results
plot_distributions(data, log_transformed, 'Log-based Transformation')
plot_distributions(data, power_transformed, 'Power Transformation (sqrt)')
plot_distributions(data, sigmoid_transformed, 'Sigmoid Transformation')

# Print summary statistics
results = pd.DataFrame({
    'Original': data,
    'Log_Transform': log_transformed,
    'Power_Transform': power_transformed,
    'Sigmoid_Transform': sigmoid_transformed
})

print("\nSummary Statistics:")
print(results.describe())

# Calculate and print dispersion metrics
print("\nDispersion Metrics:")
for column in results.columns:
    iqr = results[column].quantile(0.75) - results[column].quantile(0.25)
    print(f"\n{column}:")
    print(f"Standard Deviation: {results[column].std():.2f}")
    print(f"IQR: {iqr:.2f}")
    print(f"Coefficient of Variation: {results[column].std() / results[column].mean():.2f}")
