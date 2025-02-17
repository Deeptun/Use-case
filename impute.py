df.loc[missing_indices, 'y'] = np.nan

# Initialize and fit Prophet model
model = Prophet(
    yearly_seasonality=True,  # Match our annual seasonality pattern
    weekly_seasonality=False,
    daily_seasonality=False
)
model.fit(df.dropna())  # Fit on available data

# Create future dataframe with all dates (including missing ones)
future = model.make_future_dataframe(periods=0, freq='M')

# Generate forecasts
forecast = model.predict(future)

# Merge predictions with original data
result = pd.merge(df, forecast[['ds', 'yhat']], on='ds', how='left')
result['imputed'] = result['y'].combine_first(result['yhat'])

#************************************************************************

# --------- Imputation using Prophet ---------
df_prophet = df.reset_index().rename(columns={'date': 'ds', 'value': 'y'})
df_train = df_prophet.dropna()  # Drop NaNs for training

# Train Prophet Model
prophet = Prophet()
prophet.fit(df_train)

# Create future dataframe
future = prophet.make_future_dataframe(periods=0, freq='D')
forecast = prophet.predict(future)

# Merge predicted values into original dataframe
df_prophet.set_index('ds', inplace=True)
df_prophet.loc[df_prophet['y'].isna(), 'y'] = forecast.set_index('ds').loc[df_prophet['y'].isna(), 'yhat']
