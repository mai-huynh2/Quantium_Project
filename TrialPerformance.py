import pandas as pd
import numpy as np
from scipy.stats import ttest_ind
import matplotlib.pyplot as plt

# Load dataset
file_path = "QVI_data.csv"
data = pd.read_csv(file_path)

# Add YEARMONTH column
data['DATE'] = pd.to_datetime(data['DATE'])
data['YEARMONTH'] = data['DATE'].dt.year * 100 + data['DATE'].dt.month

# Calculate measures over time
metrics = data.groupby(['STORE_NBR', 'YEARMONTH']).agg(
    totSales=('TOT_SALES', 'sum'),
    nCustomers=('LYLTY_CARD_NBR', 'nunique'),
    nTxnPerCust=('TXN_ID', lambda x: x.nunique() / x.nunique()),
    avgPricePerUnit=('TOT_SALES', lambda x: x.sum() / x.count())
).reset_index()

# Filter pre-trial period and stores with full observation
pre_trial = metrics[metrics['YEARMONTH'] < 201902]
stores_with_full_obs = pre_trial.groupby('STORE_NBR').filter(lambda x: len(x) == 12)['STORE_NBR'].unique()

# Function to calculate correlation
def calculate_correlation(input_table, metric_col, store_comparison):
    results = []
    for store in input_table['STORE_NBR'].unique():
        if store != store_comparison:
            corr = input_table[input_table['STORE_NBR'] == store_comparison][metric_col].corr(
                input_table[input_table['STORE_NBR'] == store][metric_col])
            results.append({'Store1': store_comparison, 'Store2': store, 'corr_measure': corr})
    return pd.DataFrame(results)

# Function to calculate magnitude distance
def calculate_magnitude_distance(input_table, metric_col, store_comparison):
    results = []
    for store in input_table['STORE_NBR'].unique():
        if store != store_comparison:
            trial_data = input_table[input_table['STORE_NBR'] == store_comparison][metric_col]
            control_data = input_table[input_table['STORE_NBR'] == store][metric_col]
            magnitude = abs(trial_data.mean() - control_data.mean())
            results.append({'Store1': store_comparison, 'Store2': store, 'measure': magnitude})
    return pd.DataFrame(results)

# Select control store for trial store 77
trial_store = 77
corr_nSales = calculate_correlation(pre_trial, 'totSales', trial_store)
magnitude_nSales = calculate_magnitude_distance(pre_trial, 'totSales', trial_store)

# Combine scores
corr_weight = 0.5
scores = pd.merge(corr_nSales, magnitude_nSales, on=['Store1', 'Store2'])
scores['final_score'] = corr_weight * scores['corr_measure'] + (1 - corr_weight) * scores['measure']
control_store = scores.sort_values('final_score', ascending=False).iloc[0]['Store2']

# Visualize results
trial_data = metrics[metrics['STORE_NBR'] == trial_store]
control_data = metrics[metrics['STORE_NBR'] == control_store]

plt.figure(figsize=(10, 6))
plt.plot(trial_data['YEARMONTH'], trial_data['totSales'], label='Trial Store')
plt.plot(control_data['YEARMONTH'], control_data['totSales'], label='Control Store')
plt.legend()
plt.title('Total Sales Comparison')
plt.xlabel('Year-Month')
plt.ylabel('Total Sales')
plt.show()

# Statistical analysis
pre_trial_trial = pre_trial[pre_trial['STORE_NBR'] == trial_store]['totSales']
pre_trial_control = pre_trial[pre_trial['STORE_NBR'] == control_store]['totSales']

t_stat, p_val = ttest_ind(pre_trial_trial, pre_trial_control)
print(f"T-Test Results: t-statistic={t_stat}, p-value={p_val}")
