# 5) FILL THE MISSING VALUES WITH PROPHET FORECASTS
# -------------------------------------------------
df_imputed = df.copy()
missing_mask = df_imputed['y'].isna()
df_imputed.loc[missing_mask, 'y'] = forecast.loc[missing_mask, 'yhat']


You can achieve this by applying different filling strategies for weekends and weekdays separately. Here's the Python code using pandas:

import pandas as pd

# Sample DataFrame structure
# Assuming 'df' has columns: 'company_id', 'date', 'deposit', 'loan'
df['date'] = pd.to_datetime(df['date'])

# Identify weekends (Saturday and Sunday)
df['is_weekend'] = df['date'].dt.weekday >= 5  # Saturday (5) & Sunday (6)

# Apply different filling strategies
df.loc[df['is_weekend'], ['deposit', 'loan']] = df.loc[df['is_weekend'], ['deposit', 'loan']].ffill()  # Forward fill weekends
df.loc[~df['is_weekend'], ['deposit', 'loan']] = df.loc[~df['is_weekend'], ['deposit', 'loan']].fillna(0)  # Fill weekdays with 0

# Drop the temporary 'is_weekend' column
df.drop(columns=['is_weekend'], inplace=True)

# Display the updated DataFrame
print(df)

Explanation:

1. Convert date to datetime format (if not already).


2. Identify weekends using dt.weekday. (Saturday = 5, Sunday = 6).


3. Forward fill (ffill) only for weekends to propagate the previous value.


4. Fill NaNs with 0 for weekdays using fillna(0).


5. Drop the temporary is_weekend column.



This ensures that weekends retain the last known values while random missing weekdays are replaced with zero. Let me know if you need modifications!

You can filter out companies that have only NaN or 0 loan values in the last 12 months while keeping those that have at least one non-zero loan entry. Here's the Python code:

import pandas as pd

# Convert 'date' to datetime format (if not already)
df['date'] = pd.to_datetime(df['date'])

# Define the cutoff date for the last 12 months
end_date = df['date'].max()
start_date = end_date - pd.DateOffset(years=1)

# Filter last 12 months' data
last_12_months = df[(df['date'] >= start_date) & (df['date'] <= end_date)]

# Identify companies where all loan values in the last 12 months are either NaN or 0
invalid_companies = last_12_months.groupby('company_id')['loan'].apply(lambda x: (x.isna() | (x == 0)).all())

# Filter out these companies from the main dataset
valid_companies = invalid_companies[~invalid_companies].index  # Companies with at least one non-zero loan entry
filtered_df = df[df['company_id'].isin(valid_companies)]

# Display the updated DataFrame
print(filtered_df)

Explanation:

1. Extract the last 12 months' data based on the date column.


2. Check if all loan values in the last 12 months are either NaN or 0 for each company_id.


3. Keep only those companies where at least one loan value is non-zero.


4. Filter the original dataset to include only these valid companies.



This ensures you retain only companies that had at least one non-zero loan entry in the last 12 months, regardless of their deposit or other KPI data. Let me know if you need modifications!

