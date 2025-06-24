import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm

# Load data
df = pd.read_csv('International_Education_Costs.csv')

# Convert relevant columns to numeric
num_cols = ['Tuition_USD', 'Rent_USD', 'Insurance_USD', 'Duration_Years', 'Visa_Fee_USD', 'Living_Cost_Index']
for col in num_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Calculate total cost (full program duration)
df['Total_Cost_USD'] = (
    (df['Tuition_USD'] + df['Rent_USD'] * 12 + df['Insurance_USD']) * df['Duration_Years'] + df['Visa_Fee_USD']
)

# Basic info
print('--- Dataset Info ---')
print(df.info())
print('\n--- Head ---')
print(df.head())
print('\n--- Missing Values ---')
print(df.isnull().sum())
print('\n--- Shape ---')
print(df.shape)

# Descriptive statistics
print('\n--- Descriptive Statistics ---')
print(df.describe(include='all'))

# Distribution plots for key numeric columns
for col in num_cols + ['Total_Cost_USD']:
    plt.figure(figsize=(8,4))
    sns.histplot(df[col].dropna(), kde=True, bins=30)
    plt.title(f'Distribution of {col}')
    plt.xlabel(col)
    plt.ylabel('Count')
    plt.tight_layout()
    plt.savefig(f'eda_{col}_hist.png')
    plt.close()

# Bar chart: count of entries per country
plt.figure(figsize=(12,6))
df['Country'].value_counts().plot(kind='bar')
plt.title('Number of Entries per Country')
plt.xlabel('Country')
plt.ylabel('Count')
plt.tight_layout()
plt.savefig('eda_country_counts.png')
plt.close()

# Boxplot: total cost by country (top 10 by count)
top_countries = df['Country'].value_counts().head(10).index
plt.figure(figsize=(12,6))
sns.boxplot(x='Country', y='Total_Cost_USD', data=df[df['Country'].isin(top_countries)])
plt.title('Total Cost Distribution by Country (Top 10)')
plt.xlabel('Country')
plt.ylabel('Total Cost (USD)')
plt.tight_layout()
plt.savefig('eda_total_cost_by_country_boxplot.png')
plt.close()

# Correlation heatmap for numeric columns
plt.figure(figsize=(10,8))
corr = df[num_cols + ['Total_Cost_USD']].corr()
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Heatmap (Numeric Columns)')
plt.tight_layout()
plt.savefig('eda_correlation_heatmap.png')
plt.close()

# Key findings (example, you can expand)
print('\n--- Key Findings ---')
print(f"Countries in dataset: {df['Country'].nunique()}")
print(f"Programs: {df['Program'].nunique()}")
print(f"Degree levels: {df['Level'].unique()}")
print(f"Average total cost (USD): {df['Total_Cost_USD'].mean():,.0f}")
print(f"Median total cost (USD): {df['Total_Cost_USD'].median():,.0f}")
print(f"Country with most entries: {df['Country'].value_counts().idxmax()}")

# Prepare data for OLS regression
predictors = ['Tuition_USD', 'Rent_USD', 'Insurance_USD', 'Duration_Years', 'Visa_Fee_USD', 'Living_Cost_Index']
df_ols = df.dropna(subset=predictors + ['Total_Cost_USD'])
X = df_ols[predictors]
y = df_ols['Total_Cost_USD']
X = sm.add_constant(X)

# Fit OLS regression
ols_model = sm.OLS(y, X).fit()

print('\n--- OLS Regression Summary (MLR) ---')
print(ols_model.summary()) 