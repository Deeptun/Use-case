import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from datetime import timedelta

# ----------------------------
# 1. Generate Dummy Data
# ----------------------------
np.random.seed(42)

# Parameters
num_companies = 400
num_days = 365 * 4  # ~4 years
total_rows = num_companies * num_days

# Generate a date range
date_range = pd.date_range(start='2021-01-01', periods=num_days, freq='D')

# Create a list of company IDs (e.g., COMP001, COMP002, ..., COMP400)
company_ids = [f'COMP{str(i).zfill(3)}' for i in range(1, num_companies + 1)]

# Create a MultiIndex of company_id and date
index = pd.MultiIndex.from_product([company_ids, date_range], names=['company_id', 'date'])

# Create an empty DataFrame with the MultiIndex
df = pd.DataFrame(index=index).reset_index()

# Function to generate a column with realistic values and some noise
def generate_series(n, low, high, zero_prob, nan_prob, inf_prob):
    # Start with uniformly distributed values
    series = np.random.uniform(low, high, size=n)
    # Randomly set some values to 0
    mask_zero = np.random.rand(n) < zero_prob
    series[mask_zero] = 0
    # Randomly set some values to NaN
    mask_nan = np.random.rand(n) < nan_prob
    series[mask_nan] = np.nan
    # Randomly set some values to infinity
    mask_inf = np.random.rand(n) < inf_prob
    series[mask_inf] = np.inf
    return series

# Generate columns:
df['deposit_balance'] = generate_series(total_rows, 1000, 5000, zero_prob=0.2, nan_prob=0.1, inf_prob=0.01)
df['used_loan_amount'] = generate_series(total_rows, 500, 3000, zero_prob=0.15, nan_prob=0.1, inf_prob=0.005)
df['unused_loan_amount'] = generate_series(total_rows, 200, 1500, zero_prob=0.15, nan_prob=0.1, inf_prob=0.005)

# ----------------------------
# 2. Compute Loan Utilization
# ----------------------------
# loan_utilization = used / (used + unused), handling division by zero.
denom = df['used_loan_amount'] + df['unused_loan_amount']
df['loan_utilization'] = df['used_loan_amount'] / denom
df.loc[denom == 0, 'loan_utilization'] = np.nan  # avoid division by zero

# ----------------------------
# 3. Waterfall (Funnel) Chart
# ----------------------------
# Step A: Count companies that have any non-zero, valid loan data
def valid_loans(group):
    valid_used = ((group['used_loan_amount'] != 0) & group['used_loan_amount'].notna() & np.isfinite(group['used_loan_amount'])).any()
    valid_unused = ((group['unused_loan_amount'] != 0) & group['unused_loan_amount'].notna() & np.isfinite(group['unused_loan_amount'])).any()
    return valid_used or valid_unused

companies_with_loans = df.groupby('company_id').filter(valid_loans)['company_id'].nunique()

# Step B: Count companies with both valid loans and deposit data
def valid_deposit(group):
    return ((group['deposit_balance'] != 0) & group['deposit_balance'].notna() & np.isfinite(group['deposit_balance'])).any()

companies_with_both = df.groupby('company_id').filter(lambda g: valid_loans(g) and valid_deposit(g))['company_id'].nunique()

# Step C: Create company-level summary (for segmentation later)
company_summary = df.groupby('company_id').agg({
    'deposit_balance': 'mean',
    'used_loan_amount': 'mean',
    'unused_loan_amount': 'mean',
    'loan_utilization': 'mean'
}).reset_index()

# For segmentation, keep only companies that have both deposit and loan data (non-NaN average values)
company_summary = company_summary.dropna(subset=['deposit_balance', 'used_loan_amount', 'unused_loan_amount', 'loan_utilization'])

# Segment counts for each key metric – we’ll create quantile-based segments (low, medium, high)
company_summary['utilization_segment'] = pd.qcut(company_summary['loan_utilization'], q=3, labels=['low', 'medium', 'high'])
company_summary['used_loan_segment'] = pd.qcut(company_summary['used_loan_amount'], q=3, labels=['low', 'medium', 'high'])
company_summary['deposit_segment'] = pd.qcut(company_summary['deposit_balance'], q=3, labels=['low', 'medium', 'high'])

# For waterfall chart, we’ll show the funnel of counts:
waterfall_steps = {
    'All Companies with Loans': companies_with_loans,
    'Companies with Both Deposits & Loans': companies_with_both,
    'Segmented (Companies with valid summary)': company_summary.shape[0]
}

# Plot a simple waterfall chart
fig, ax = plt.subplots(figsize=(8, 5))
steps = list(waterfall_steps.keys())
values = list(waterfall_steps.values())
bars = []
cumulative = 0
for i, v in enumerate(values):
    if i == 0:
        bars.append(v)
        cumulative = v
    else:
        diff = v - cumulative
        bars.append(diff)
        cumulative = v

# For a waterfall effect, plot bars with appropriate colors
colors = ['skyblue' if v >= 0 else 'salmon' for v in bars]
# Compute positions
x_pos = np.arange(len(bars))
# We can plot the cumulative totals
cumulative_totals = np.cumsum(bars)
ax.bar(x_pos, cumulative_totals, color='lightgrey', edgecolor='black', width=0.5)
# Overlay individual step markers
for i, (xp, cum) in enumerate(zip(x_pos, cumulative_totals)):
    ax.text(xp, cum + (max(values)*0.01), f'{cum}', ha='center', va='bottom', fontsize=10)
ax.set_xticks(x_pos)
ax.set_xticklabels(steps, rotation=45, ha='right')
ax.set_ylabel("Number of Companies")
ax.set_title("Waterfall/Funnel Chart: Companies Filtering Steps")
plt.tight_layout()
plt.show()

# ----------------------------
# 4. Pivot Table Visualization of Segments
# ----------------------------
# Example: Cross-tabulate utilization segments vs. deposit segments
pivot_seg = pd.pivot_table(company_summary, index='utilization_segment', 
                           columns='deposit_segment', 
                           values='company_id', 
                           aggfunc='count')
plt.figure(figsize=(6, 4))
sns.heatmap(pivot_seg, annot=True, fmt='d', cmap='Blues')
plt.title("Pivot Table: Loan Utilization vs. Deposit Segments")
plt.ylabel("Loan Utilization Segment")
plt.xlabel("Deposit Segment")
plt.show()

# ----------------------------
# 5. Correlation Flag between Loan Utilization and Deposits
# ----------------------------
# Compute the Pearson correlation for each company’s time series (if enough valid data exists)
def compute_corr(group):
    # Only compute if we have at least 5 valid points in each series
    valid_dep = group['deposit_balance'].notna() & np.isfinite(group['deposit_balance'])
    valid_util = group['loan_utilization'].notna() & np.isfinite(group['loan_utilization'])
    if valid_dep.sum() < 5 or valid_util.sum() < 5:
        return np.nan
    return group.loc[valid_dep & valid_util, 'deposit_balance'].corr(group.loc[valid_dep & valid_util, 'loan_utilization'])

corr_df = df.groupby('company_id').apply(compute_corr).reset_index(name='corr')
company_summary = company_summary.merge(corr_df, on='company_id', how='left')

# Flag companies with high positive (>=0.7) or high negative (<= -0.7) correlations
def flag_corr(c):
    if pd.isna(c):
        return 'insufficient'
    if c >= 0.7:
        return 'high_positive'
    elif c <= -0.7:
        return 'high_negative'
    else:
        return 'none'

company_summary['corr_flag'] = company_summary['corr'].apply(flag_corr)
print("Correlation flag counts:\n", company_summary['corr_flag'].value_counts())

# ----------------------------
# 6. Risk Flags Based on Recent Trends
# ----------------------------
# For each company, compute trends (slope) in deposit_balance and loan_utilization for the last 6 months.
def compute_trends(group, months=6):
    last_date = group['date'].max()
    start_date = last_date - pd.DateOffset(months=months)
    sub = group[group['date'] >= start_date].copy()
    if len(sub) < 5:
        return pd.Series({'deposit_slope': np.nan, 'loan_utilization_slope': np.nan})
    # Convert dates to ordinal for regression
    sub['date_ord'] = sub['date'].apply(pd.Timestamp.toordinal)
    # For deposit_balance
    valid_dep = sub['deposit_balance'].notna() & np.isfinite(sub['deposit_balance'])
    if valid_dep.sum() >= 5:
        slope_dep = np.polyfit(sub.loc[valid_dep, 'date_ord'], sub.loc[valid_dep, 'deposit_balance'], 1)[0]
    else:
        slope_dep = np.nan
    # For loan_utilization
    valid_util = sub['loan_utilization'].notna() & np.isfinite(sub['loan_utilization'])
    if valid_util.sum() >= 5:
        slope_util = np.polyfit(sub.loc[valid_util, 'date_ord'], sub.loc[valid_util, 'loan_utilization'], 1)[0]
    else:
        slope_util = np.nan
    return pd.Series({'deposit_slope': slope_dep, 'loan_utilization_slope': slope_util})

# Apply to each company (we use the original df grouped by company)
trend_df = df.groupby('company_id').apply(compute_trends, months=6).reset_index()
company_summary = company_summary.merge(trend_df, on='company_id', how='left')

# Define risk flag conditions:
def risk_flag(row):
    # If slopes are not computed, return False
    if pd.isna(row['deposit_slope']) or pd.isna(row['loan_utilization_slope']):
        return False
    # Condition 1: Loan utilization increasing while deposits are flat or decreasing.
    if row['loan_utilization_slope'] > 0 and row['deposit_slope'] <= 0:
        return True
    # Condition 2: Loan utilization steady but deposits steadily decreasing.
    if abs(row['loan_utilization_slope']) < 1e-6 and row['deposit_slope'] < 0:
        return True
    # Condition 3: Loan utilization decreasing but deposits dropping even faster.
    if row['loan_utilization_slope'] < 0 and row['deposit_slope'] < row['loan_utilization_slope']:
        return True
    return False

company_summary['risk_flag'] = company_summary.apply(risk_flag, axis=1)
print("Risk flag counts:\n", company_summary['risk_flag'].value_counts())

# ----------------------------
# 7. Cluster Clients for Cohort Analysis
# ----------------------------
# For clustering, we use aggregated features. We first drop rows with missing values.
cluster_features = company_summary.dropna(subset=['deposit_balance', 'used_loan_amount', 'loan_utilization', 'deposit_slope', 'loan_utilization_slope'])
# Select features (you may scale/normalize in a production setting)
features = cluster_features[['deposit_balance', 'used_loan_amount', 'loan_utilization', 'deposit_slope', 'loan_utilization_slope']]
# For simplicity, we use KMeans with 3 clusters
kmeans = KMeans(n_clusters=3, random_state=42)
cluster_features['cluster'] = kmeans.fit_predict(features)

# Merge cluster labels back into the company_summary
company_summary = company_summary.merge(cluster_features[['company_id', 'cluster']], on='company_id', how='left')

print("Cluster counts:\n", company_summary['cluster'].value_counts())

# Visualize average trends over time by cluster
# Merge cluster labels back into the main dataframe
df = df.merge(company_summary[['company_id', 'cluster']], on='company_id', how='left')

# For each cluster and date, compute average deposit and loan utilization.
time_trend = df.groupby(['cluster', 'date']).agg({
    'deposit_balance': 'mean',
    'loan_utilization': 'mean'
}).reset_index()

# Plot trends for each cluster
plt.figure(figsize=(12, 5))
for cl in sorted(time_trend['cluster'].dropna().unique()):
    subset = time_trend[time_trend['cluster'] == cl]
    plt.plot(subset['date'], subset['deposit_balance'], label=f'Cluster {cl} - Deposit')
plt.title("Average Deposit Balance Over Time by Cluster")
plt.xlabel("Date")
plt.ylabel("Average Deposit Balance")
plt.legend()
plt.tight_layout()
plt.show()

plt.figure(figsize=(12, 5))
for cl in sorted(time_trend['cluster'].dropna().unique()):
    subset = time_trend[time_trend['cluster'] == cl]
    plt.plot(subset['date'], subset['loan_utilization'], label=f'Cluster {cl} - Loan Utilization')
plt.title("Average Loan Utilization Over Time by Cluster")
plt.xlabel("Date")
plt.ylabel("Average Loan Utilization")
plt.legend()
plt.tight_layout()
plt.show()

# ----------------------------
# End of Analysis
# ----------------------------
