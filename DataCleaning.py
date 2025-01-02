import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import zscore
from scipy.stats import ttest_ind

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
print(products)

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


# Filter for Budget Young Singles/Couples
budget_young = merged_data[
    (merged_data['PREMIUM_CUSTOMER'] == 'BUDGET') &
    (merged_data['LIFESTAGE'] == 'YOUNG SINGLES/COUPLES')
]

# Brand preference
brand_preference = budget_young['BRAND_NAME'].value_counts()
print("Brand Preference for Budget Young Singles/Couples:")
print(brand_preference)

# Pack size preference
pack_size_preference = budget_young['PACK_SIZE'].value_counts()
print("Pack Size Preference for Budget Young Singles/Couples:")
print(pack_size_preference)

print("Brand Preference for Budget Young Singles/Couples:")
print(brand_preference)

print("Pack Size Preference for Budget Young Singles/Couples:")
print(pack_size_preference)

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(10, 6))
sns.barplot(x=brand_preference.index, y=brand_preference.values)
plt.title('Brand Preference for Budget Young Singles/Couples')
plt.xlabel('Brand')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.show()

# Pack size preference analysis
pack_size_preference = budget_young['PACK_SIZE'].value_counts()
print("Pack Size Preference for Budget Young Singles/Couples:")
print(pack_size_preference)

# Visualize pack size preferences
plt.figure(figsize=(10, 6))
sns.barplot(x=pack_size_preference.index, y=pack_size_preference.values)
plt.title('Pack Size Preference for Budget Young Singles/Couples')
plt.xlabel('Pack Size (g)')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.show()

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

# Filter for Premium Young Singles/Couples
premium_young = merged_data[
    (merged_data['PREMIUM_CUSTOMER'] == 'PREMIUM') &
    (merged_data['LIFESTAGE'] == 'YOUNG SINGLES/COUPLES')
]

# Brand preference
brand_preference = premium_young['BRAND_NAME'].value_counts()
print("Brand Preference for premium Young Singles/Couples:")
print(brand_preference)

# Pack size preference
pack_size_preference = premium_young['PACK_SIZE'].value_counts()
print("Pack Size Preference for Premium Young Singles/Couples:")
print(pack_size_preference)

print("Brand Preference for Premium Young Singles/Couples:")
print(brand_preference)

print("Pack Size Preference for Premium Young Singles/Couples:")
print(pack_size_preference)

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(10, 6))
sns.barplot(x=brand_preference.index, y=brand_preference.values)
plt.title('Brand Preference for Premium Young Singles/Couples')
plt.xlabel('Brand')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.show()

# Pack size preference analysis
pack_size_preference = premium_young['PACK_SIZE'].value_counts()
print("Pack Size Preference for Premium Young Singles/Couples:")
print(pack_size_preference)

# Visualize pack size preferences
plt.figure(figsize=(10, 6))
sns.barplot(x=pack_size_preference.index, y=pack_size_preference.values)
plt.title('Pack Size Preference for Premium Young Singles/Couples')
plt.xlabel('Pack Size (g)')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.show()

# Filter for Budget Young Families
budget_youngfam = merged_data[
    (merged_data['PREMIUM_CUSTOMER'] == 'BUDGET') &
    (merged_data['LIFESTAGE'] == 'YOUNG FAMILIES')
]

# Brand preference
brand_preference = budget_youngfam['BRAND_NAME'].value_counts()
print("Brand Preference for Budget Young Families:")
print(brand_preference)

print("Brand Preference for Budget Young Families:")
print(brand_preference)

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(10, 6))
sns.barplot(x=brand_preference.index, y=brand_preference.values)
plt.title('Brand Preference for Budget Young Families')
plt.xlabel('Brand')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.show()

# Filter for Budget Young Families
budget_youngfam = merged_data[
    (merged_data['PREMIUM_CUSTOMER'] == 'BUDGET') &
    (merged_data['LIFESTAGE'] == 'YOUNG FAMILIES')
]

# Brand preference
brand_preference = budget_youngfam['BRAND_NAME'].value_counts()
print("Brand Preference for Budget Young Families:")
print(brand_preference)

print("Brand Preference for Budget Young Families:")
print(brand_preference)

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(10, 6))
sns.barplot(x=brand_preference.index, y=brand_preference.values)
plt.title('Brand Preference for Budget Young Families')
plt.xlabel('Brand')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.show()

# Filter for Mainstream Young Families
mainstream_youngfam = merged_data[
    (merged_data['PREMIUM_CUSTOMER'] == 'MAINSTREAM') &
    (merged_data['LIFESTAGE'] == 'YOUNG FAMILIES')
]

# Brand preference
brand_preference = mainstream_youngfam['BRAND_NAME'].value_counts()
print("Brand Preference for Mainstream Young Families:")
print(brand_preference)

print("Brand Preference for Mainstream Young Families:")
print(brand_preference)

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(10, 6))
sns.barplot(x=brand_preference.index, y=brand_preference.values)
plt.title('Brand Preference for Mainstream Young Families')
plt.xlabel('Brand')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.show()

# Pack size preference analysis
pack_size_preference = mainstream_youngfam['PACK_SIZE'].value_counts()
print("Pack Size Preference for Mainstream Young Families:")
print(pack_size_preference)

# Visualize pack size preferences
plt.figure(figsize=(10, 6))
sns.barplot(x=pack_size_preference.index, y=pack_size_preference.values)
plt.title('Pack Size Preference for Mainstream Young Families')
plt.xlabel('Pack Size (g)')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.show()

# Filter for Premium Young Families
premium_youngfam = merged_data[
    (merged_data['PREMIUM_CUSTOMER'] == 'PREMIUM') &
    (merged_data['LIFESTAGE'] == 'YOUNG FAMILIES')
]

# Brand preference
brand_preference = premium_youngfam['BRAND_NAME'].value_counts()
print("Brand Preference for Premium Young Families:")
print(brand_preference)

print("Brand Preference for Premium Young Families:")
print(brand_preference)

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(10, 6))
sns.barplot(x=brand_preference.index, y=brand_preference.values)
plt.title('Brand Preference for Premium Young Families')
plt.xlabel('Brand')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.show()

# Pack size preference analysis
pack_size_preference = premium_youngfam['PACK_SIZE'].value_counts()
print("Pack Size Preference for Premium Young Families:")
print(pack_size_preference)

# Visualize pack size preferences
plt.figure(figsize=(10, 6))
sns.barplot(x=pack_size_preference.index, y=pack_size_preference.values)
plt.title('Pack Size Preference for Premium Young Families')
plt.xlabel('Pack Size (g)')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.show()

# Filter for Budget Retirees
budget_retirees = merged_data[
    (merged_data['PREMIUM_CUSTOMER'] == 'BUDGET') &
    (merged_data['LIFESTAGE'] == 'RETIREES')
]

# Brand preference
brand_preference = budget_retirees['BRAND_NAME'].value_counts()
print("Brand Preference for Budget Retirees:")
print(brand_preference)

print("Brand Preference for Budget Retirees:")
print(brand_preference)

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(10, 6))
sns.barplot(x=brand_preference.index, y=brand_preference.values)
plt.title('Brand Preference for Budget Retirees')
plt.xlabel('Brand')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.show()

# Pack size preference analysis
pack_size_preference = budget_retirees['PACK_SIZE'].value_counts()
print("Pack Size Preference for Budget Retirees:")
print(pack_size_preference)

# Visualize pack size preferences
plt.figure(figsize=(10, 6))
sns.barplot(x=pack_size_preference.index, y=pack_size_preference.values)
plt.title('Pack Size Preference for Budget Retirees')
plt.xlabel('Pack Size (g)')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.show()

# Filter for Mainstream Retirees
mainstream_retirees = merged_data[
    (merged_data['PREMIUM_CUSTOMER'] == 'MAINSTREAM') &
    (merged_data['LIFESTAGE'] == 'RETIREES')
]

# Brand preference
brand_preference = mainstream_retirees['BRAND_NAME'].value_counts()
print("Brand Preference for Mainstream Retirees:")
print(brand_preference)

print("Brand Preference for Mainstream Retirees:")
print(brand_preference)

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(10, 6))
sns.barplot(x=brand_preference.index, y=brand_preference.values)
plt.title('Brand Preference for Mainstream Retirees')
plt.xlabel('Brand')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.show()

# Pack size preference analysis
pack_size_preference = mainstream_retirees['PACK_SIZE'].value_counts()
print("Pack Size Preference for Mainstream Retirees:")
print(pack_size_preference)

# Visualize pack size preferences
plt.figure(figsize=(10, 6))
sns.barplot(x=pack_size_preference.index, y=pack_size_preference.values)
plt.title('Pack Size Preference for Mainstream Retirees')
plt.xlabel('Pack Size (g)')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.show()

# Filter for Premium Retirees
premium_retirees = merged_data[
    (merged_data['PREMIUM_CUSTOMER'] == 'PREMIUM') &
    (merged_data['LIFESTAGE'] == 'RETIREES')
]

# Brand preference
brand_preference = premium_retirees['BRAND_NAME'].value_counts()
print("Brand Preference for Premium Retirees:")
print(brand_preference)

print("Brand Preference for Premium Retirees:")
print(brand_preference)

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(10, 6))
sns.barplot(x=brand_preference.index, y=brand_preference.values)
plt.title('Brand Preference for Premium Retirees')
plt.xlabel('Brand')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.show()

# Pack size preference analysis
pack_size_preference = premium_retirees['PACK_SIZE'].value_counts()
print("Pack Size Preference for Premium Retirees:")
print(pack_size_preference)

# Visualize pack size preferences
plt.figure(figsize=(10, 6))
sns.barplot(x=pack_size_preference.index, y=pack_size_preference.values)
plt.title('Pack Size Preference for Premium Retirees')
plt.xlabel('Pack Size (g)')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.show()

# Filter for Budget Older Singles/Couples
budget_older = merged_data[
    (merged_data['PREMIUM_CUSTOMER'] == 'BUDGET') &
    (merged_data['LIFESTAGE'] == 'OLDER SINGLES/COUPLES')
]

# Brand preference
brand_preference = budget_older['BRAND_NAME'].value_counts()
print("Brand Preference for Older Singles/Couples:")
print(brand_preference)

print("Brand Preference for Older Singles/Couples:")
print(brand_preference)

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(10, 6))
sns.barplot(x=brand_preference.index, y=brand_preference.values)
plt.title('Brand Preference for Budget Older Singles/Couples')
plt.xlabel('Brand')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.show()

# Pack size preference analysis
pack_size_preference = budget_older['PACK_SIZE'].value_counts()
print("Pack Size Preference for Budget Older Singles/Couples:")
print(pack_size_preference)

# Visualize pack size preferences
plt.figure(figsize=(10, 6))
sns.barplot(x=pack_size_preference.index, y=pack_size_preference.values)
plt.title('Pack Size Preference for Budget Older Singles/Couples')
plt.xlabel('Pack Size (g)')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.show()

# Filter for Mainstream Older Singles/Couples
mainstream_older = merged_data[
    (merged_data['PREMIUM_CUSTOMER'] == 'MAINSTREAM') &
    (merged_data['LIFESTAGE'] == 'OLDER SINGLES/COUPLES')
]

# Brand preference
brand_preference = mainstream_older['BRAND_NAME'].value_counts()
print("Brand Preference for Older Singles/Couples:")
print(brand_preference)

print("Brand Preference for Older Singles/Couples:")
print(brand_preference)

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(10, 6))
sns.barplot(x=brand_preference.index, y=brand_preference.values)
plt.title('Brand Preference for Mainstream Older Singles/Couples')
plt.xlabel('Brand')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.show()
# Pack size preference analysis
pack_size_preference = mainstream_older['PACK_SIZE'].value_counts()
print("Pack Size Preference for Mainstream Older Singles/Couples:")
print(pack_size_preference)

# Visualize pack size preferences
plt.figure(figsize=(10, 6))
sns.barplot(x=pack_size_preference.index, y=pack_size_preference.values)
plt.title('Pack Size Preference for Mainstream Older Singles/Couples')
plt.xlabel('Pack Size (g)')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.show()

# Filter for Premium Older Singles/Couples
premium_older = merged_data[
    (merged_data['PREMIUM_CUSTOMER'] == 'PREMIUM') &
    (merged_data['LIFESTAGE'] == 'OLDER SINGLES/COUPLES')
]

# Brand preference
brand_preference = premium_older['BRAND_NAME'].value_counts()
print("Brand Preference for Older Singles/Couples:")
print(brand_preference)

print("Brand Preference for Older Singles/Couples:")
print(brand_preference)

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(10, 6))
sns.barplot(x=brand_preference.index, y=brand_preference.values)
plt.title('Brand Preference for Premium Older Singles/Couples')
plt.xlabel('Brand')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.show()

# Pack size preference analysis
pack_size_preference = premium_older['PACK_SIZE'].value_counts()
print("Pack Size Preference for Premium Older Singles/Couples:")
print(pack_size_preference)

# Visualize pack size preferences
plt.figure(figsize=(10, 6))
sns.barplot(x=pack_size_preference.index, y=pack_size_preference.values)
plt.title('Pack Size Preference for Premium Older Singles/Couples')
plt.xlabel('Pack Size (g)')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.show()

# Filter for Budget Older Families
budget_olderfam = merged_data[
    (merged_data['PREMIUM_CUSTOMER'] == 'BUDGET') &
    (merged_data['LIFESTAGE'] == 'OLDER FAMILIES')
]

# Brand preference
brand_preference = budget_olderfam['BRAND_NAME'].value_counts()
print("Brand Preference for Budget Older Families:")
print(brand_preference)

print("Brand Preference for Budget Older Families:")
print(brand_preference)

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(10, 6))
sns.barplot(x=brand_preference.index, y=brand_preference.values)
plt.title('Brand Preference for Budget Older Families')
plt.xlabel('Brand')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.show()

# Pack size preference analysis
pack_size_preference = budget_olderfam['PACK_SIZE'].value_counts()
print("Pack Size Preference for Budget Older Families:")
print(pack_size_preference)

# Visualize pack size preferences
plt.figure(figsize=(10, 6))
sns.barplot(x=pack_size_preference.index, y=pack_size_preference.values)
plt.title('Pack Size Preference for Budget Older Families')
plt.xlabel('Pack Size (g)')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.show()

# Filter for Mainstream Older Families
mainstream_olderfam = merged_data[
    (merged_data['PREMIUM_CUSTOMER'] == 'MAINSTREAM') &
    (merged_data['LIFESTAGE'] == 'OLDER FAMILIES')
]

# Brand preference
brand_preference = mainstream_olderfam['BRAND_NAME'].value_counts()
print("Brand Preference for Mainstream Older Families:")
print(brand_preference)

print("Brand Preference for Mainstream Older Families:")
print(brand_preference)

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(10, 6))
sns.barplot(x=brand_preference.index, y=brand_preference.values)
plt.title('Brand Preference for Mainstream Older Families')
plt.xlabel('Brand')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.show()

# Pack size preference analysis
pack_size_preference = mainstream_olderfam['PACK_SIZE'].value_counts()
print("Pack Size Preference for Mainstream Older Families:")
print(pack_size_preference)

# Visualize pack size preferences
plt.figure(figsize=(10, 6))
sns.barplot(x=pack_size_preference.index, y=pack_size_preference.values)
plt.title('Pack Size Preference for Mainstream Older Families')
plt.xlabel('Pack Size (g)')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.show()

# Filter for Premium Older Families
premium_olderfam = merged_data[
    (merged_data['PREMIUM_CUSTOMER'] == 'PREMIUM') &
    (merged_data['LIFESTAGE'] == 'OLDER FAMILIES')
]

# Brand preference
brand_preference = premium_olderfam['BRAND_NAME'].value_counts()
print("Brand Preference for Premium Older Families:")
print(brand_preference)

print("Brand Preference for Premium Older Families:")
print(brand_preference)

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(10, 6))
sns.barplot(x=brand_preference.index, y=brand_preference.values)
plt.title('Brand Preference for Premium Older Families')
plt.xlabel('Brand')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.show()

# Pack size preference analysis
pack_size_preference = premium_olderfam['PACK_SIZE'].value_counts()
print("Pack Size Preference for Premium Older Families:")
print(pack_size_preference)

# Visualize pack size preferences
plt.figure(figsize=(10, 6))
sns.barplot(x=pack_size_preference.index, y=pack_size_preference.values)
plt.title('Pack Size Preference for Premium Older Families')
plt.xlabel('Pack Size (g)')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.show()

# Filter for Budget New Families
budget_newfam = merged_data[
    (merged_data['PREMIUM_CUSTOMER'] == 'BUDGET') &
    (merged_data['LIFESTAGE'] == 'NEW FAMILIES')
]

# Brand preference
brand_preference = budget_newfam['BRAND_NAME'].value_counts()
print("Brand Preference for Budget New Families:")
print(brand_preference)

print("Brand Preference for Budget New Families:")
print(brand_preference)

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(10, 6))
sns.barplot(x=brand_preference.index, y=brand_preference.values)
plt.title('Brand Preference for Budget New Families')
plt.xlabel('Brand')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.show()

# Pack size preference analysis
pack_size_preference = budget_newfam['PACK_SIZE'].value_counts()
print("Pack Size Preference for Budget New Families:")
print(pack_size_preference)

# Visualize pack size preferences
plt.figure(figsize=(10, 6))
sns.barplot(x=pack_size_preference.index, y=pack_size_preference.values)
plt.title('Pack Size Preference for Budget New Families')
plt.xlabel('Pack Size (g)')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.show()

# Filter for Mainstream New Families
mainstream_newfam = merged_data[
    (merged_data['PREMIUM_CUSTOMER'] == 'MAINSTREAM') &
    (merged_data['LIFESTAGE'] == 'NEW FAMILIES')
]

# Brand preference
brand_preference = mainstream_newfam['BRAND_NAME'].value_counts()
print("Brand Preference for Mainstream New Families:")
print(brand_preference)

print("Brand Preference for Mainstream New Families:")
print(brand_preference)

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(10, 6))
sns.barplot(x=brand_preference.index, y=brand_preference.values)
plt.title('Brand Preference for Mainstream New Families')
plt.xlabel('Brand')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.show()# Filter for Mainstream New Families
mainstream_newfam = merged_data[
    (merged_data['PREMIUM_CUSTOMER'] == 'MAINSTREAM') &
    (merged_data['LIFESTAGE'] == 'NEW FAMILIES')
]

# Pack size preference analysis
pack_size_preference = mainstream_newfam['PACK_SIZE'].value_counts()
print("Pack Size Preference for Mainstream New Families:")
print(pack_size_preference)

# Visualize pack size preferences
plt.figure(figsize=(10, 6))
sns.barplot(x=pack_size_preference.index, y=pack_size_preference.values)
plt.title('Pack Size Preference for Mainstream New Families')
plt.xlabel('Pack Size (g)')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.show()

# Filter for premium New Families
premium_newfam = merged_data[
    (merged_data['PREMIUM_CUSTOMER'] == 'PREMIUM') &
    (merged_data['LIFESTAGE'] == 'NEW FAMILIES')
]

# Brand preference
brand_preference = premium_newfam['BRAND_NAME'].value_counts()
print("Brand Preference for Premium New Families:")
print(brand_preference)

print("Brand Preference for Premium New Families:")
print(brand_preference)

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(10, 6))
sns.barplot(x=brand_preference.index, y=brand_preference.values)
plt.title('Brand Preference for Premium New Families')
plt.xlabel('Brand')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.show()

# Pack size preference analysis
pack_size_preference = premium_newfam['PACK_SIZE'].value_counts()
print("Pack Size Preference for Premium New Families:")
print(pack_size_preference)

# Visualize pack size preferences
plt.figure(figsize=(10, 6))
sns.barplot(x=pack_size_preference.index, y=pack_size_preference.values)
plt.title('Pack Size Preference for Premium New Families')
plt.xlabel('Pack Size (g)')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.show()

# Filter for Budget Midage Singles/Couples
budget_midage = merged_data[
    (merged_data['PREMIUM_CUSTOMER'] == 'BUDGET') &
    (merged_data['LIFESTAGE'] == 'MIDAGE SINGLES/COUPLES')
]

# Brand preference
brand_preference = budget_midage['BRAND_NAME'].value_counts()
print("Brand Preference for Midage Singles/Couples:")
print(brand_preference)

print("Brand Preference for Midage Singles/Couples:")
print(brand_preference)

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(10, 6))
sns.barplot(x=brand_preference.index, y=brand_preference.values)
plt.title('Brand Preference for Budget Midage Singles/Couples')
plt.xlabel('Brand')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.show()

# Pack size preference analysis
pack_size_preference = budget_midage['PACK_SIZE'].value_counts()
print("Pack Size Preference for Budget Midage Singles/Couples:")
print(pack_size_preference)

# Visualize pack size preferences
plt.figure(figsize=(10, 6))
sns.barplot(x=pack_size_preference.index, y=pack_size_preference.values)
plt.title('Pack Size Preference for Budget Midage Singles/Couples')
plt.xlabel('Pack Size (g)')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.show()

# Filter for Mainstream Midage Singles/Couples
mainstream_midage = merged_data[
    (merged_data['PREMIUM_CUSTOMER'] == 'MAINSTREAM') &
    (merged_data['LIFESTAGE'] == 'MIDAGE SINGLES/COUPLES')
]

# Brand preference
brand_preference = mainstream_midage['BRAND_NAME'].value_counts()
print("Brand Preference for Budget Midage Singles/Couples:")
print(brand_preference)

print("Brand Preference for Budget Midage Singles/Couples:")
print(brand_preference)

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(10, 6))
sns.barplot(x=brand_preference.index, y=brand_preference.values)
plt.title('Brand Preference for Mainstream Midage Singles/Couples')
plt.xlabel('Brand')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.show()

# Pack size preference analysis
pack_size_preference = mainstream_midage['PACK_SIZE'].value_counts()
print("Pack Size Preference for Mainstream Midage Singles/Couples:")
print(pack_size_preference)

# Visualize pack size preferences
plt.figure(figsize=(10, 6))
sns.barplot(x=pack_size_preference.index, y=pack_size_preference.values)
plt.title('Pack Size Preference for Mainstream Midage Singles/Couples')
plt.xlabel('Pack Size (g)')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.show()

# Filter for Premium Midage Singles/Couples
premium_midage = merged_data[
    (merged_data['PREMIUM_CUSTOMER'] == 'PREMIUM') &
    (merged_data['LIFESTAGE'] == 'MIDAGE SINGLES/COUPLES')
]

# Brand preference
brand_preference = premium_midage['BRAND_NAME'].value_counts()
print("Brand Preference for Premium Midage Singles/Couples:")
print(brand_preference)

print("Brand Preference for Premium Midage Singles/Couples:")
print(brand_preference)

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(10, 6))
sns.barplot(x=brand_preference.index, y=brand_preference.values)
plt.title('Brand Preference for Premium Midage Singles/Couples')
plt.xlabel('Brand')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.show()

# Pack size preference analysis
pack_size_preference = premium_midage['PACK_SIZE'].value_counts()
print("Pack Size Preference for Premium Midage Singles/Couples:")
print(pack_size_preference)

# Visualize pack size preferences
plt.figure(figsize=(10, 6))
sns.barplot(x=pack_size_preference.index, y=pack_size_preference.values)
plt.title('Pack Size Preference for Premium Midage Singles/Couples')
plt.xlabel('Pack Size (g)')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.show()

# Calculate averages for pack size and brand preference
overall_avg_pack_size = merged_data['PACK_SIZE'].mean()
overall_avg_brand_pref = merged_data['BRAND_NAME'].value_counts(normalize=True) * 100

avg_pack_by_premium = merged_data.groupby('PREMIUM_CUSTOMER')['PACK_SIZE'].mean()
avg_pack_by_lifestage = merged_data.groupby('LIFESTAGE')['PACK_SIZE'].mean()

brand_pref_by_premium = merged_data.groupby('PREMIUM_CUSTOMER')['BRAND_NAME'].value_counts(normalize=True).unstack() * 100
brand_pref_by_lifestage = merged_data.groupby('LIFESTAGE')['BRAND_NAME'].value_counts(normalize=True).unstack() * 100

# Overall Pack Size Distribution
plt.figure(figsize=(8, 6))
sns.histplot(merged_data['PACK_SIZE'], bins=20, kde=True, color='blue')
plt.title('Overall Pack Size Distribution')
plt.xlabel('Pack Size (grams)')
plt.ylabel('Frequency')
plt.show()

print(overall_avg_pack_size)

# Average Pack Size by Affluence
plt.figure(figsize=(8, 6))
avg_pack_by_premium.plot(kind='bar', color='green')
plt.title('Average Pack Size by Affluence')
plt.xlabel('Premium Customer Status')
plt.ylabel('Average Pack Size (grams)')
plt.show()

print(avg_pack_by_premium)

# Average Pack Size by Lifestage
plt.figure(figsize=(8, 6))
avg_pack_by_lifestage.plot(kind='bar', color='orange')
plt.title('Average Pack Size by Lifestage')
plt.xlabel('Lifestage')
plt.ylabel('Average Pack Size (grams)')
plt.show()

print(avg_pack_by_lifestage)

# Brand Preference by Affluence
plt.figure(figsize=(10, 6))
brand_pref_by_premium.T.plot(kind='bar', stacked=True, figsize=(10, 6), colormap='tab20')
plt.title('Brand Preference by Affluence')
plt.xlabel('BRAND_NAME')
plt.ylabel('Percentage')
plt.legend(title='Premium Status')
plt.show()

print(brand_pref_by_premium)

# Brand Preference by Lifestage
plt.figure(figsize=(10, 6))
BRAND_NAME_pref_by_lifestage.T.plot(kind='bar', stacked=True, figsize=(10, 6), colormap='tab20')
plt.title('Brand Preference by Lifestage')
plt.xlabel('BRAND_NAME')
plt.ylabel('Percentage')
plt.legend(title='Lifestage')
plt.show()

print(brand_pref_by_lifestage)

# Overall Brand Preference
plt.figure(figsize=(10, 6))
overall_avg_BRAND_NAME_pref.plot(kind='bar', color='purple')
plt.title('Overall Brand Preference')
plt.xlabel('BRAND_NAME')
plt.ylabel('Percentage')
plt.show()

overall_avg_brand_pref
