# %%
pip install kagglehub pandas numpy matplotlib seaborn scikit-learn statsmodels


# %%
# Cost of International Education - Multiple Linear Regression Analysis
# Project: Measuring Impact of Various Features on International Education Costs


import kagglehub
from kagglehub import KaggleDatasetAdapter
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import statsmodels.api as sm
from statsmodels.stats.diagnostic import het_breuschpagan
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# %%
# Set style for better visualizations
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

print("=" * 60)
print("INTERNATIONAL EDUCATION COST ANALYSIS")
print("Multiple Linear Regression Study")
print("=" * 60)

# %%

# 1. DATA LOADING AND EXPLORATION
print("\n1. LOADING DATASET...")
try:
    # Load the dataset
    file_path = ""  # Leave empty to load main file
    df = kagglehub.load_dataset(
        KaggleDatasetAdapter.PANDAS,
        "adilshamim8/cost-of-international-education",
        file_path
    )
    
    print(f"✓ Dataset loaded successfully!")
    print(f"Dataset shape: {df.shape}")
    print(f"\nFirst 5 records:")
    print(df.head())
    
except Exception as e:
    print(f"Error loading dataset: {e}")
    print("Creating sample dataset for demonstration...")
    
    # Create a sample dataset for demonstration
    np.random.seed(42)
    n_samples = 1000
    
    countries = ['USA', 'UK', 'Canada', 'Australia', 'Germany', 'France', 'Netherlands', 
                'Sweden', 'Switzerland', 'Japan', 'Singapore', 'New Zealand']
    
    df = pd.DataFrame({
        'Country': np.random.choice(countries, n_samples),
        'University_Ranking': np.random.randint(1, 500, n_samples),
        'Tuition_Fee_USD': np.random.normal(25000, 15000, n_samples),
        'Living_Cost_USD': np.random.normal(15000, 8000, n_samples),
        'GDP_Per_Capita': np.random.normal(45000, 20000, n_samples),
        'University_Type': np.random.choice(['Public', 'Private'], n_samples),
        'Program_Duration_Years': np.random.choice([1, 2, 3, 4], n_samples),
        'City_Tier': np.random.choice([1, 2, 3], n_samples),
        'Language_Requirement_Score': np.random.normal(6.5, 1.5, n_samples)
    })
        # Create total cost as our target variable
    df['Total_Annual_Cost_USD'] = (df['Tuition_Fee_USD'] + df['Living_Cost_USD'] + 
                                  np.random.normal(0, 2000, n_samples))
    
    # Ensure positive values
    df = df[df['Total_Annual_Cost_USD'] > 0]
    df = df[df['Tuition_Fee_USD'] > 0]
    df = df[df['Living_Cost_USD'] > 0]
    
    print("Sample dataset created for demonstration")
    print(f"Dataset shape: {df.shape}")

# %%
# 2. EXPLORATORY DATA ANALYSIS
print("\n2. EXPLORATORY DATA ANALYSIS")
print("=" * 40)

# Basic info
print("\nDataset Info:")
print(df.info())

print("\nDescriptive Statistics:")
print(df.describe())

# Check for missing values
print(f"\nMissing Values:")
print(df.isnull().sum())

# %%
# Identify target variable and features
# Assuming the main cost variable is our target
cost_columns = [col for col in df.columns if 'cost' in col.lower() or 'fee' in col.lower() or 'tuition' in col.lower()]
print(f"\nPotential target variables: {cost_columns}")

# For this analysis, let's assume we're predicting total education cost
if 'Total_Annual_Cost_USD' in df.columns:
    target_var = 'Total_Annual_Cost_USD'
elif 'Tuition_Fee_USD' in df.columns:
    target_var = 'Tuition_Fee_USD'
else:
    # Use the first numeric column as target
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    target_var = numeric_cols[0]

print(f"Target variable selected: {target_var}")

# %%
# 3. DATA PREPROCESSING
print("\n3. DATA PREPROCESSING")
print("=" * 40)

# Separate features and target
X = df.drop(columns=[target_var])
y = df[target_var]

# Handle categorical variables
categorical_cols = X.select_dtypes(include=['object']).columns
numerical_cols = X.select_dtypes(include=[np.number]).columns

print(f"Categorical columns: {list(categorical_cols)}")
print(f"Numerical columns: {list(numerical_cols)}")

# Encode categorical variables
label_encoders = {}
X_processed = X.copy()

for col in categorical_cols:
    le = LabelEncoder()
    X_processed[col] = le.fit_transform(X[col].astype(str))
    label_encoders[col] = le

# Create dummy variables for better interpretation (alternative approach)
X_dummies = pd.get_dummies(X, columns=categorical_cols, drop_first=True)

print(f"Features after preprocessing: {X_processed.shape[1]}")
print(f"Features with dummies: {X_dummies.shape[1]}")

# %%
# 4. CORRELATION ANALYSIS
print("\n4. CORRELATION ANALYSIS")
print("=" * 40)

# Calculate correlation matrix
correlation_matrix = X_processed.corrwith(y).sort_values(ascending=False)
print("Feature correlations with target variable:")
print(correlation_matrix.round(3))

# Visualize correlations
plt.figure(figsize=(12, 8))
plt.subplot(2, 2, 1)
sns.heatmap(X_processed.corr(), annot=True, cmap='coolwarm', center=0, fmt='.2f')
plt.title('Feature Correlation Matrix')

plt.subplot(2, 2, 2)
correlation_matrix.plot(kind='barh')
plt.title('Feature Correlations with Target')
plt.xlabel('Correlation Coefficient')

plt.tight_layout()
plt.show()

# %%

# 5. MODEL SELECTION RATIONALE
print("\n5. MODEL SELECTION RATIONALE")
print("=" * 40)

print("""
RECOMMENDED APPROACH: Multiple Linear Regression (OLS)

Why OLS is appropriate for this analysis:
1. Linear Relationship: Education costs typically have linear relationships with features
2. Interpretability: Coefficients directly show impact of each feature
3. Statistical Inference: P-values and confidence intervals for significance testing
4. Assumptions: Can be tested and validated

When to consider alternatives:
- GLM: If non-normal residuals or non-linear relationships
- Ridge/Lasso: If multicollinearity or feature selection needed
- Random Forest: If complex interactions or non-linear patterns exist
""")

# %%
# 6. MULTIPLE LINEAR REGRESSION ANALYSIS
print("\n6. MULTIPLE LINEAR REGRESSION ANALYSIS")
print("=" * 40)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X_dummies, y, test_size=0.2, random_state=42
)

# Scale features for better interpretation
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Convert back to DataFrame for feature names
X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=X_train.columns, index=X_train.index)
X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=X_test.columns, index=X_test.index)

# Method 1: Scikit-learn Linear Regression
print("\nMethod 1: Scikit-learn Linear Regression")
lr_model = LinearRegression()
lr_model.fit(X_train_scaled, y_train)

# Predictions
y_pred_train = lr_model.predict(X_train_scaled)
y_pred_test = lr_model.predict(X_test_scaled)

# Model performance
train_r2 = r2_score(y_train, y_pred_train)
test_r2 = r2_score(y_test, y_pred_test)
train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))

print(f"Training R²: {train_r2:.4f}")
print(f"Testing R²: {test_r2:.4f}")
print(f"Training RMSE: ${train_rmse:,.2f}")
print(f"Testing RMSE: ${test_rmse:,.2f}")

# Feature importance
feature_importance = pd.DataFrame({
    'Feature': X_train.columns,
    'Coefficient': lr_model.coef_,
    'Abs_Coefficient': np.abs(lr_model.coef_)
}).sort_values('Abs_Coefficient', ascending=False)

print("\nTop 10 Most Important Features:")
print(feature_importance.head(10))

# Method 2: Statsmodels OLS for detailed statistics
print("\nMethod 2: Statsmodels OLS (Detailed Statistics)")
X_train_sm = sm.add_constant(X_train_scaled_df)
X_test_sm = sm.add_constant(X_test_scaled_df)

ols_model = sm.OLS(y_train, X_train_sm).fit()
print(ols_model.summary())


# %%
# 7. MODEL DIAGNOSTICS
print("\n7. MODEL DIAGNOSTICS")
print("=" * 40)

# Residual analysis
residuals = y_train - y_pred_train
residuals_test = y_test - y_pred_test

# Plotting diagnostics
plt.figure(figsize=(15, 10))

# Residuals vs Fitted
plt.subplot(2, 3, 1)
plt.scatter(y_pred_train, residuals, alpha=0.6)
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel('Fitted Values')
plt.ylabel('Residuals')
plt.title('Residuals vs Fitted Values')

# Q-Q plot for normality
plt.subplot(2, 3, 2)
stats.probplot(residuals, dist="norm", plot=plt)
plt.title('Q-Q Plot (Normality Check)')

# Actual vs Predicted
plt.subplot(2, 3, 3)
plt.scatter(y_test, y_pred_test, alpha=0.6)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Actual vs Predicted (Test Set)')

# Feature importance
plt.subplot(2, 3, 4)
top_features = feature_importance.head(10)
plt.barh(top_features['Feature'], top_features['Coefficient'])
plt.xlabel('Coefficient Value')
plt.title('Top 10 Feature Coefficients')
plt.gca().invert_yaxis()

# Distribution of residuals
plt.subplot(2, 3, 5)
plt.hist(residuals, bins=30, alpha=0.7, density=True)
plt.xlabel('Residuals')
plt.ylabel('Density')
plt.title('Distribution of Residuals')

# Cook's distance (outlier detection)
plt.subplot(2, 3, 6)
influence = ols_model.get_influence()
cooks_d = influence.cooks_distance[0]
plt.stem(range(len(cooks_d)), cooks_d)
plt.xlabel('Observation')
plt.ylabel("Cook's Distance")
plt.title("Cook's Distance (Outlier Detection)")

plt.tight_layout()
plt.show()

# Statistical tests
print("\nStatistical Tests:")

# Heteroscedasticity test
lm, lm_pvalue, fvalue, f_pvalue = het_breuschpagan(residuals, X_train_sm)
print(f"Breusch-Pagan Test for Heteroscedasticity: p-value = {lm_pvalue:.4f}")
if lm_pvalue < 0.05:
    print("⚠️  Heteroscedasticity detected (consider robust standard errors)")
else:
    print("✓ Homoscedasticity assumption satisfied")

# Normality test
shapiro_stat, shapiro_p = stats.shapiro(residuals[:5000])  # Sample for large datasets
print(f"Shapiro-Wilk Test for Normality: p-value = {shapiro_p:.4f}")
if shapiro_p < 0.05:
    print("⚠️  Residuals may not be normally distributed")
else:
    print("✓ Normality assumption satisfied")

# %%
#  8. RESULTS INTERPRETATION
print("\n8. RESULTS INTERPRETATION AND RECOMMENDATIONS")
print("=" * 50)

print(f"""
MODEL PERFORMANCE SUMMARY:
- R² Score: {test_r2:.4f} ({test_r2*100:.1f}% of variance explained)
- RMSE: ${test_rmse:,.2f}
- Model explains {test_r2*100:.1f}% of the variation in education costs

KEY FINDINGS:
""")

# Interpret top features
for i, row in feature_importance.head(5).iterrows():
    direction = "increases" if row['Coefficient'] > 0 else "decreases"
    print(f"- {row['Feature']}: ${abs(row['Coefficient']):,.2f} {direction} in cost per unit change")

print(f"""
RECOMMENDATIONS FOR YOUR CLASS PRESENTATION:

1. MODEL CHOICE: ✓ OLS is appropriate for this analysis
   - Linear relationships are reasonable for education costs
   - Good interpretability for stakeholders
   - Statistical significance testing available

2. KEY INSIGHTS TO HIGHLIGHT:
   - Model R² of {test_r2:.3f} shows {'good' if test_r2 > 0.7 else 'moderate' if test_r2 > 0.5 else 'limited'} predictive power
   - Top cost drivers identified
   - Practical implications for students/policymakers

3. POTENTIAL IMPROVEMENTS:
   - Consider interaction terms between features
   - Explore polynomial features for non-linear relationships
   - Use cross-validation for more robust performance estimates
   - Consider regularization (Ridge/Lasso) if overfitting occurs

4. LIMITATIONS TO MENTION:
   - Assumes linear relationships
   - Sensitive to outliers
   - Requires assumption validation
""")

# %%
# 9. SAVE RESULTS
print("\n9. SAVING RESULTS")
print("=" * 40)

# Create results summary
results_summary = {
    'Model_Type': 'Multiple Linear Regression (OLS)',
    'Training_R2': train_r2,
    'Testing_R2': test_r2,
    'Training_RMSE': train_rmse,
    'Testing_RMSE': test_rmse,
    'Number_of_Features': len(X_train.columns),
    'Sample_Size': len(df)
}

results_df = pd.DataFrame([results_summary])
print("Model Results Summary:")
print(results_df.round(4))

print("\n" + "="*60)
print("ANALYSIS COMPLETE! Ready for your class presentation.")
print("="*60)


