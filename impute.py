# 5) FILL THE MISSING VALUES WITH PROPHET FORECASTS
# -------------------------------------------------
df_imputed = df.copy()
missing_mask = df_imputed['y'].isna()
df_imputed.loc[missing_mask, 'y'] = forecast.loc[missing_mask, 'yhat']
