import pandas as pd
import numpy as np
from scipy.stats import ttest_ind
import matplotlib.pyplot as plt

# Load Data
file_path = "QVI_data.csv"  # Replace with your file path
data = pd.read_csv(r"/content/QVI_data.csv")

# Add YEARMONTH Column
data['DATE'] = pd.to_datetime(data['DATE'])
data['YEARMONTH'] = data['DATE'].dt.year * 100 + data['DATE'].dt.month

# Preprocess Metrics
metrics = data.groupby(['STORE_NBR', 'YEARMONTH']).agg(
    totSales=('TOT_SALES', 'sum'),
    nCustomers=('LYLTY_CARD_NBR', 'nunique'),
    nTxnPerCust=('TOT_SALES', 'count'),
    avgPricePerUnit=('TOT_SALES', 'mean')
).reset_index()

def calculate_correlation(input_table, metric_col, trial_store):
    correlation_table = []
    trial_data = input_table[input_table['STORE_NBR'] == trial_store][['YEARMONTH', metric_col]]
    for store in input_table['STORE_NBR'].unique():
        if store != trial_store:
            control_data = input_table[input_table['STORE_NBR'] == store][['YEARMONTH', metric_col]]
            merged = pd.merge(trial_data, control_data, on='YEARMONTH', suffixes=('_trial', '_control'))
            if not merged.empty:
                corr = merged[f'{metric_col}_trial'].corr(merged[f'{metric_col}_control'])
                correlation_table.append({'Store': store, 'Correlation': corr})
    return pd.DataFrame(correlation_table)

def calculate_magnitude_distance(input_table, metric_col, trial_store):
    magnitude_table = []
    trial_mean = input_table[input_table['STORE_NBR'] == trial_store][metric_col].mean()
    for store in input_table['STORE_NBR'].unique():
        if store != trial_store:
            control_mean = input_table[input_table['STORE_NBR'] == store][metric_col].mean()
            magnitude = abs(trial_mean - control_mean)
            magnitude_table.append({'Store': store, 'Magnitude': magnitude})
    return pd.DataFrame(magnitude_table)

trial_store = 77

# Correlation for Sales and Customers
corr_sales = calculate_correlation(metrics, 'totSales', trial_store)
corr_customers = calculate_correlation(metrics, 'nCustomers', trial_store)

# Magnitude Distance for Sales and Customers
mag_sales = calculate_magnitude_distance(metrics, 'totSales', trial_store)
mag_customers = calculate_magnitude_distance(metrics, 'nCustomers', trial_store)

# Normalize Magnitude
for mag in [mag_sales, mag_customers]:
    min_val, max_val = mag['Magnitude'].min(), mag['Magnitude'].max()
    mag['Normalized Magnitude'] = 1 - (mag['Magnitude'] - min_val) / (max_val - min_val)

# Combine Scores for Sales
combined_sales = pd.merge(corr_sales, mag_sales, on='Store')
combined_sales['Score'] = 0.5 * combined_sales['Correlation'] + 0.5 * combined_sales['Normalized Magnitude']

# Combine Scores for Customers
combined_customers = pd.merge(corr_customers, mag_customers, on='Store')
combined_customers['Score'] = 0.5 * combined_customers['Correlation'] + 0.5 * combined_customers['Normalized Magnitude']

# Final Control Store Selection
combined_scores = pd.merge(combined_sales, combined_customers, on='Store', suffixes=('_sales', '_customers'))
combined_scores['FinalScore'] = (combined_scores['Score_sales'] + combined_scores['Score_customers']) / 2

control_store = combined_scores.sort_values('FinalScore', ascending=False).iloc[0]['Store']
print(f"Selected Control Store for Trial Store {trial_store}: {control_store}")

# Visualize Sales Trends
trial_data_sales = metrics[metrics['STORE_NBR'] == trial_store]
control_data_sales = metrics[metrics['STORE_NBR'] == control_store]

plt.figure(figsize=(12, 6))
plt.plot(trial_data_sales['YEARMONTH'], trial_data_sales['totSales'], label=f'Trial Store {trial_store}', marker='o')
plt.plot(control_data_sales['YEARMONTH'], control_data_sales['totSales'], label=f'Control Store {control_store}', marker='o')
plt.axvline(x=201902, color='red', linestyle='--', label='Trial Start')
plt.title('Total Sales: Trial Store vs Control Store')
plt.xlabel('Year-Month')
plt.ylabel('Total Sales')
plt.legend()
plt.grid()
plt.show()

# Scale Control Sales
scaling_factor = trial_data_sales[trial_data_sales['YEARMONTH'] < 201902]['totSales'].sum() / \
                 control_data_sales[control_data_sales['YEARMONTH'] < 201902]['totSales'].sum()

# Create a new column safely using .loc
control_data_sales.loc[:, 'ScaledSales'] = control_data_sales['totSales'] * scaling_factor

# Calculate Percentage Difference
trial_period_sales = trial_data_sales[trial_data_sales['YEARMONTH'] >= 201902]['totSales']
control_period_sales = control_data_sales[control_data_sales['YEARMONTH'] >= 201902]['ScaledSales']
percentage_diff_sales = abs(trial_period_sales.values - control_period_sales.values) / control_period_sales.values * 100

print(f"Percentage Difference in Sales: {percentage_diff_sales}")

trial_store = 86

# Correlation for Sales and Customers
corr_sales = calculate_correlation(metrics, 'totSales', trial_store)
corr_customers = calculate_correlation(metrics, 'nCustomers', trial_store)

# Magnitude Distance for Sales and Customers
mag_sales = calculate_magnitude_distance(metrics, 'totSales', trial_store)
mag_customers = calculate_magnitude_distance(metrics, 'nCustomers', trial_store)

# Normalize Magnitude
for mag in [mag_sales, mag_customers]:
    min_val, max_val = mag['Magnitude'].min(), mag['Magnitude'].max()
    mag['Normalized Magnitude'] = 1 - (mag['Magnitude'] - min_val) / (max_val - min_val)

# Combine Scores for Sales
combined_sales = pd.merge(corr_sales, mag_sales, on='Store')
combined_sales['Score'] = 0.5 * combined_sales['Correlation'] + 0.5 * combined_sales['Normalized Magnitude']

# Combine Scores for Customers
combined_customers = pd.merge(corr_customers, mag_customers, on='Store')
combined_customers['Score'] = 0.5 * combined_customers['Correlation'] + 0.5 * combined_customers['Normalized Magnitude']

# Final Control Store Selection
combined_scores = pd.merge(combined_sales, combined_customers, on='Store', suffixes=('_sales', '_customers'))
combined_scores['FinalScore'] = (combined_scores['Score_sales'] + combined_scores['Score_customers']) / 2

control_store = combined_scores.sort_values('FinalScore', ascending=False).iloc[0]['Store']
print(f"Selected Control Store for Trial Store {trial_store}: {control_store}")

# Visualize Sales Trends
trial_data_sales = metrics[metrics['STORE_NBR'] == trial_store]
control_data_sales = metrics[metrics['STORE_NBR'] == control_store]

plt.figure(figsize=(12, 6))
plt.plot(trial_data_sales['YEARMONTH'], trial_data_sales['totSales'], label=f'Trial Store {trial_store}', marker='o')
plt.plot(control_data_sales['YEARMONTH'], control_data_sales['totSales'], label=f'Control Store {control_store}', marker='o')
plt.axvline(x=201902, color='red', linestyle='--', label='Trial Start')
plt.title('Total Sales: Trial Store vs Control Store')
plt.xlabel('Year-Month')
plt.ylabel('Total Sales')
plt.legend()
plt.grid()
plt.show()

# Scale Control Sales
scaling_factor = trial_data_sales[trial_data_sales['YEARMONTH'] < 201902]['totSales'].sum() / \
                 control_data_sales[control_data_sales['YEARMONTH'] < 201902]['totSales'].sum()

# Create a new column safely using .loc
control_data_sales.loc[:, 'ScaledSales'] = control_data_sales['totSales'] * scaling_factor

# Calculate Percentage Difference
trial_period_sales = trial_data_sales[trial_data_sales['YEARMONTH'] >= 201902]['totSales']
control_period_sales = control_data_sales[control_data_sales['YEARMONTH'] >= 201902]['ScaledSales']
percentage_diff_sales = abs(trial_period_sales.values - control_period_sales.values) / control_period_sales.values * 100

print(f"Percentage Difference in Sales: {percentage_diff_sales}")

trial_store = 88

# Correlation for Sales and Customers
corr_sales = calculate_correlation(metrics, 'totSales', trial_store)
corr_customers = calculate_correlation(metrics, 'nCustomers', trial_store)

# Magnitude Distance for Sales and Customers
mag_sales = calculate_magnitude_distance(metrics, 'totSales', trial_store)
mag_customers = calculate_magnitude_distance(metrics, 'nCustomers', trial_store)

# Normalize Magnitude
for mag in [mag_sales, mag_customers]:
    min_val, max_val = mag['Magnitude'].min(), mag['Magnitude'].max()
    mag['Normalized Magnitude'] = 1 - (mag['Magnitude'] - min_val) / (max_val - min_val)

# Combine Scores for Sales
combined_sales = pd.merge(corr_sales, mag_sales, on='Store')
combined_sales['Score'] = 0.5 * combined_sales['Correlation'] + 0.5 * combined_sales['Normalized Magnitude']

# Combine Scores for Customers
combined_customers = pd.merge(corr_customers, mag_customers, on='Store')
combined_customers['Score'] = 0.5 * combined_customers['Correlation'] + 0.5 * combined_customers['Normalized Magnitude']

# Final Control Store Selection
combined_scores = pd.merge(combined_sales, combined_customers, on='Store', suffixes=('_sales', '_customers'))
combined_scores['FinalScore'] = (combined_scores['Score_sales'] + combined_scores['Score_customers']) / 2

control_store = combined_scores.sort_values('FinalScore', ascending=False).iloc[0]['Store']
print(f"Selected Control Store for Trial Store {trial_store}: {control_store}")

# Visualize Sales Trends
trial_data_sales = metrics[metrics['STORE_NBR'] == trial_store]
control_data_sales = metrics[metrics['STORE_NBR'] == control_store]

plt.figure(figsize=(12, 6))
plt.plot(trial_data_sales['YEARMONTH'], trial_data_sales['totSales'], label=f'Trial Store {trial_store}', marker='o')
plt.plot(control_data_sales['YEARMONTH'], control_data_sales['totSales'], label=f'Control Store {control_store}', marker='o')
plt.axvline(x=201902, color='red', linestyle='--', label='Trial Start')
plt.title('Total Sales: Trial Store vs Control Store')
plt.xlabel('Year-Month')
plt.ylabel('Total Sales')
plt.legend()
plt.grid()
plt.show()

# Scale Control Sales
scaling_factor = trial_data_sales[trial_data_sales['YEARMONTH'] < 201902]['totSales'].sum() / \
                 control_data_sales[control_data_sales['YEARMONTH'] < 201902]['totSales'].sum()

# Create a new column safely using .loc
control_data_sales.loc[:, 'ScaledSales'] = control_data_sales['totSales'] * scaling_factor

# Calculate Percentage Difference
trial_period_sales = trial_data_sales[trial_data_sales['YEARMONTH'] >= 201902]['totSales']
control_period_sales = control_data_sales[control_data_sales['YEARMONTH'] >= 201902]['ScaledSales']
percentage_diff_sales = abs(trial_period_sales.values - control_period_sales.values) / control_period_sales.values * 100

print(f"Percentage Difference in Sales: {percentage_diff_sales}")
