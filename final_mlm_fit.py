import pandas as pd
import numpy as np
from final_data_prep import load_and_prepare_data
import statsmodels.formula.api as smf

# Load preprocessed data
df, scaler = load_and_prepare_data()

# Exclude columns that should not be predictors
drop_cols = {'Country', 'University', 'Total_Cost_USD'}
predictors = [col for col in df.columns if col not in drop_cols]

# Build formula string
dependent = 'Total_Cost_USD'
formula = dependent + ' ~ ' + ' + '.join(predictors)

try:
    # Fit Mixed Linear Model (random intercepts for Country)
    print("Fitting Mixed Linear Model...")
    print(f"Number of predictors: {len(predictors)}")
    print(f"Sample size: {len(df)}")
    
    model = smf.mixedlm(formula, df, groups=df['Country'])
    result = model.fit()
    
    # Print model summary
    print("\nModel Summary:")
    print(result.summary())
    
    # Print key metrics
    print("\nKey Metrics:")
    print(f"AIC: {result.aic:.2f}")
    print(f"BIC: {result.bic:.2f}")
    
except Exception as e:
    print(f"Error fitting model: {str(e)}")
    print("\nDebug information:")
    print(f"Formula: {formula}")
    print("\nColumn names:")
    print(df.columns.tolist()) 