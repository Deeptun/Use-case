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

