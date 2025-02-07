import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.preprocessing import (PowerTransformer, QuantileTransformer, 
                                   RobustScaler, MinMaxScaler)

np.random.seed(42)

# Generate realistic skewed data
original_scores = np.exp(np.random.normal(5, 1.5, 11)).astype(int)
original_scores.sort()
print("Original Scores:", original_scores)

# Define all transformation functions
def min_max_scaling(data):
    scaler = MinMaxScaler(feature_range=(1, 100))
    return scaler.fit_transform(data.reshape(-1, 1)).flatten()

def zscore_scaling(data):
    z = stats.zscore(data)
    scaler = MinMaxScaler(feature_range=(1, 100))
    return scaler.fit_transform(z.reshape(-1, 1)).flatten()

def log_transform(data):
    log_data = np.log(data)
    scaler = MinMaxScaler(feature_range=(1, 100))
    return scaler.fit_transform(log_data.reshape(-1, 1)).flatten()

def sqrt_transform(data):
    sqrt_data = np.sqrt(data)
    scaler = MinMaxScaler(feature_range=(1, 100))
    return scaler.fit_transform(sqrt_data.reshape(-1, 1)).flatten()

def cbrt_transform(data):
    cbrt_data = np.cbrt(data)
    scaler = MinMaxScaler(feature_range=(1, 100))
    return scaler.fit_transform(cbrt_data.reshape(-1, 1)).flatten()

def boxcox_transform(data):
    transformed_data, _ = stats.boxcox(data)
    scaler = MinMaxScaler(feature_range=(1, 100))
    return scaler.fit_transform(transformed_data.reshape(-1, 1)).flatten()

def yeojohnson_transform(data):
    pt = PowerTransformer(method='yeo-johnson', standardize=False)
    transformed_data = pt.fit_transform(data.reshape(-1, 1)).flatten()
    scaler = MinMaxScaler(feature_range=(1, 100))
    return scaler.fit_transform(transformed_data.reshape(-1, 1)).flatten()

def quantile_transform(data):
    qt = QuantileTransformer(output_distribution='uniform', random_state=42)
    transformed_data = qt.fit_transform(data.reshape(-1, 1)).flatten()
    return transformed_data * 99 + 1

def sigmoid_transform(data):
    z = stats.zscore(data)
    sigmoid = 1 / (1 + np.exp(-z))
    return sigmoid * 99 + 1

def clip_transform(data):
    lower = np.percentile(data, 5)
    upper = np.percentile(data, 95)
    clipped = np.clip(data, lower, upper)
    scaler = MinMaxScaler(feature_range=(1, 100))
    return scaler.fit_transform(clipped.reshape(-1, 1)).flatten()

def robust_scaling(data):
    scaler = RobustScaler()
    scaled_data = scaler.fit_transform(data.reshape(-1, 1)).flatten()
    minmax_scaler = MinMaxScaler(feature_range=(1, 100))
    return minmax_scaler.fit_transform(scaled_data.reshape(-1, 1)).flatten()

def inverse_transform(data):
    inverted = 1 / data
    scaler = MinMaxScaler(feature_range=(1, 100))
    scaled_inverted = scaler.fit_transform(inverted.reshape(-1, 1)).flatten()
    return 101 - scaled_inverted  # Maintain original order

def exponential_transform(data):
    z = stats.zscore(data)
    exp_data = np.exp(z)
    scaler = MinMaxScaler(feature_range=(1, 100))
    return scaler.fit_transform(exp_data.reshape(-1, 1)).flatten()

def power_transform(data, gamma=0.3):
    powered_data = np.power(data, gamma)
    scaler = MinMaxScaler(feature_range=(1, 100))
    return scaler.fit_transform(powered_data.reshape(-1, 1)).flatten()

transformations = {
    'Min-Max': min_max_scaling,
    'Z-Score': zscore_scaling,
    'Log': log_transform,
    'Square Root': sqrt_transform,
    'Cube Root': cbrt_transform,
    'Box-Cox': boxcox_transform,
    'Yeo-Johnson': yeojohnson_transform,
    'Quantile': quantile_transform,
    'Sigmoid': sigmoid_transform,
    'Clipping': clip_transform,
    'Robust Scaling': robust_scaling,
    'Inverse': inverse_transform,
    'Exponential': exponential_transform,
    'Power (γ=0.5)': lambda data: power_transform(data, gamma=0.5)
}

# Apply all transformations
transformed_data = {name: func(original_scores) for name, func in transformations.items()}

# Calculate metrics
original_skew = stats.skew(original_scores)
original_std = np.std(original_scores)
original_percentages = (original_scores / original_scores.sum()) * 100

metrics = []
for name, data in transformed_data.items():
    skewness = stats.skew(data)
    std_dev = np.std(data)
    transformed_percent = (data / data.sum()) * 100
    metrics.append({
        'Transformation': name,
        'Skewness': skewness,
        'Std Dev': std_dev,
        'Transformed Percentages': transformed_percent
    })

# Find best transformation
low_indices = [0, 1, 2]
original_low_percent = original_percentages[low_indices].sum()

best_transformation = None
best_score = -np.inf
for metric in metrics:
    transformed_low_percent = metric['Transformed Percentages'][low_indices].sum()
    if (transformed_low_percent > original_low_percent and 
        metric['Std Dev'] < original_std):
        score = transformed_low_percent - original_low_percent - metric['Std Dev']
        if score > best_score:
            best_score = score
            best_transformation = metric['Transformation']

# Visualize distributions
plt.figure(figsize=(18, 20))
plt.subplot(5, 3, 1)
sns.histplot(original_scores, kde=True)
plt.title('Original Data\nSkew: {:.2f}, Std: {:.2f}'.format(original_skew, original_std))

for i, (name, data) in enumerate(transformed_data.items(), 1):
    plt.subplot(5, 3, i+1)
    sns.histplot(data, kde=True)
    plt.title(f'{name}\nSkew: {stats.skew(data):.2f}, Std: {np.std(data):.2f}')

plt.tight_layout()
plt.show()

# Plot original vs best transformation
best_data = transformed_data[best_transformation]
plt.figure(figsize=(12, 6))
plt.plot(original_scores, label='Original', marker='o', linestyle='--')
plt.plot(best_data, label=best_transformation, marker='s', linewidth=2)
plt.xticks(range(len(original_scores)), [f'Region {i+1}' for i in range(len(original_scores))])
plt.ylabel('Complexity Score')
plt.title(f'Best Transformation: {best_transformation}\n'
         f'Low Complexity Improvement: {metrics[metrics.index([m for m in metrics if m["Transformation"] == best_transformation][0])]["Transformed Percentages"][low_indices].sum()-original_low_percent:.1f}%')
plt.legend()
plt.show()

# Print metric comparison
metric_df = pd.DataFrame(metrics)
print("\nMetric Comparison:")
print(metric_df[['Transformation', 'Skewness', 'Std Dev']].sort_values('Std Dev'))

# Plot original vs ALL transformation
for metric in metric_df['Transformation']:
    # print(metric)
    t_data = transformed_data[metric]
    plt.figure(figsize=(12, 6))
    plt.plot(original_scores, label='Original', marker='o', linestyle='--')
    plt.plot(t_data, label=metric, marker='s', linewidth=2)
    plt.xticks(range(len(original_scores)), [f'Region {i+1}' for i in range(len(original_scores))])
    plt.ylabel('Complexity Score')
    plt.title(f'Transformation: {metric}\n'
             f'Low Complexity Improvement: {metrics[metrics.index([m for m in metrics if m["Transformation"] == metric][0])]["Transformed Percentages"][low_indices].sum()-original_low_percent:.1f}%')
    plt.legend()
    plt.show()