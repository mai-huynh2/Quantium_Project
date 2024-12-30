import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Read data
transaction_data = pd.read_excel(r"/content/QVI_transaction_data.xlsx")
customer_data = pd.read_csv(r"/content/QVI_purchase_behaviour.csv")

# Check data structure
transaction_data.info()

# Convert DATE column to datetime
transaction_data['DATE'] = pd.to_datetime(transaction_data['DATE'], origin='1899-12-30', unit='D')

# Check for missing values and duplicates 
transaction_data_missing = transaction_data.isnull().sum()
transaction_data_duplicates = transaction_data.duplicated().sum()

# Produce summary of product names
products = transaction_data['PROD_NAME'].unique()

# Display the first 10 unique products
print(prod_name_summary)

product_series = pd.Series(products)

# Define a list of keywords that indicate non-chip products
non_chip_keywords = ['Salsa', 'Popd', 'Dip', 'Crackers', 'Papadums', 'Rings', 'Mystery', 'Salsa', 'Belly', 'Aioli']

# Filter out products containing any of the non-chip keywords
filtered_products = product_series[~product_series.str.contains('|'.join(non_chip_keywords), case=False, na=False)]

# Print the filtered list of products
print("Filtered products (chips only):")
print(filtered_products.tolist())

customer_data.info()

# Check for missing values and duplicates in customer data
customer_data_missing = customer_data.isnull().sum()
customer_data_duplicates = customer_data.duplicated().sum()

# Merge datasets
merged_data = pd.merge(transaction_data, customer_data, on='LYLTY_CARD_NBR', how='inner')

# Calculate total sales
merged_data['TOTAL_SALES'] = merged_data['PROD_QTY'] * merged_data['TOT_SALES']

# Extract pack size
merged_data['PACK_SIZE'] = merged_data['PROD_NAME'].str.extract('(\d+)').astype(float)

# Extract brand name
merged_data['BRAND_NAME'] = merged_data['PROD_NAME'].str.split().str[0]

# Clean brand name
merged_data['BRAND_NAME'] = merged_data['BRAND_NAME'].replace({'RED': 'RRD'})

# Compute z-scores for numerical columns
z_scores = transaction_data.select_dtypes(include=['float64', 'int64']).apply(zscore)

# Identify rows with z-scores greater than a threshold (e.g., 3)
outliers = transaction_data[(z_scores.abs() > 3).any(axis=1)]

print("Outliers based on z-scores:")
print(outliers)

# Plot a boxplot for the 'PROD_QTY' column
plt.figure(figsize=(8, 6))
sns.boxplot(data=transaction_data, y='PROD_QTY')
plt.title("Boxplot for Product Quantity")
plt.show()

# Summarize the data to check for nulls and possible outliers
def summarize_data(data):
    print("Summary Statistics:")
    print(data.describe())
    print("\nNull Counts:")
    print(data.isnull().sum())

# Look into outlier
outlier_transactions = transaction_data[transaction_data['PROD_QTY'] == 200]
print("\nTransactions with 200 packets of chips:")
print(outlier_transactions)

# Check if they had other transactions
if not outlier_transactions.empty:
    outlier_customer = outlier_transactions['LYLTY_CARD_NBR'].iloc[0]
    customer_transactions = transaction_data[transaction_data['LYLTY_CARD_NBR'] == outlier_customer]
    print(f"\nAll transactions for customer {outlier_customer}:")
    print(customer_transactions)

# Remove outlier
filtered_data = transaction_data[transaction_data['LYLTY_CARD_NBR'] != outlier_customer]

# Save filtered data sets
filtered_data.to_csv("cleaned_transaction_data.csv", index=False)

# Sales by LIFESTAGE and PREMIUM_CUSTOMER
sales_by_segment = merged_data.groupby(['PREMIUM_CUSTOMER', 'LIFESTAGE'])['TOTAL_SALES'].sum().reset_index()

# Visualization
plt.figure(figsize=(12,6))
sns.barplot(data=sales_by_segment, x='LIFESTAGE', y='TOTAL_SALES', hue='PREMIUM_CUSTOMER')
plt.title('Total Sales by Lifestage and Premium Status')
plt.xticks(rotation=45)
plt.legend(title= 'Premium Status')
plt.show()

# T-Test for average prices
mainstream_prices = merged_data.loc[merged_data['PREMIUM_CUSTOMER'] == 'Mainstream', 'TOT_SALES']
premium_prices = merged_data.loc[merged_data['PREMIUM_CUSTOMER'] == 'Premium', 'TOT_SALES']
t_stat, p_value = ttest_ind(mainstream_prices, premium_prices)
print(f"T-Test Results: t-statistic={t_stat}, p-value={p_value}")

merged_data['PREMIUM_CUSTOMER'] = merged_data['PREMIUM_CUSTOMER'].str.strip().str.upper()
merged_data['LIFESTAGE'] = merged_data['LIFESTAGE'].str.strip().str.upper()


# Filter for Mainstream Young Singles/Couples
mainstream_young = merged_data[
    (merged_data['PREMIUM_CUSTOMER'] == 'MAINSTREAM') &
    (merged_data['LIFESTAGE'] == 'YOUNG SINGLES/COUPLES')
]

# Brand preference
brand_preference = mainstream_young['BRAND_NAME'].value_counts()
print("Brand Preference for Mainstream Young Singles/Couples:")
print(brand_preference)

# Pack size preference
pack_size_preference = mainstream_young['PACK_SIZE'].value_counts()
print("Pack Size Preference for Mainstream Young Singles/Couples:")
print(pack_size_preference)

print("Brand Preference for Mainstream Young Singles/Couples:")
print(brand_preference)

print("Pack Size Preference for Mainstream Young Singles/Couples:")
print(pack_size_preference)

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(10, 6))
sns.barplot(x=brand_preference.index, y=brand_preference.values)
plt.title('Brand Preference for Mainstream Young Singles/Couples')
plt.xlabel('Brand')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.show()

# Pack size preference analysis
pack_size_preference = mainstream_young['PACK_SIZE'].value_counts()
print("Pack Size Preference for Mainstream Young Singles/Couples:")
print(pack_size_preference)

# Visualize pack size preferences
plt.figure(figsize=(10, 6))
sns.barplot(x=pack_size_preference.index, y=pack_size_preference.values)
plt.title('Pack Size Preference for Mainstream Young Singles/Couples')
plt.xlabel('Pack Size (g)')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.show()
