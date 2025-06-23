import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from final_data_prep import load_and_prepare_data
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import os

# Create 'models' directory if it doesn't exist
if not os.path.exists('models'):
    os.makedirs('models')

def evaluate_model(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    print("\nModel Performance Metrics:")
    print(f"Root Mean Square Error: ${rmse:,.2f}")
    print(f"R-squared Score: {r2:.4f}")
    # Diagnostic plots
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    # Residuals vs Fitted
    residuals = y_true - y_pred
    sns.scatterplot(x=y_pred, y=residuals, ax=axes[0])
    axes[0].axhline(y=0, color='r', linestyle='--')
    axes[0].set_xlabel('Fitted values')
    axes[0].set_ylabel('Residuals')
    axes[0].set_title('Residuals vs Fitted')
    # Actual vs Predicted
    sns.scatterplot(x=y_true, y=y_pred, ax=axes[1])
    axes[1].plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--')
    axes[1].set_xlabel('Actual Total Cost (USD)')
    axes[1].set_ylabel('Predicted Total Cost (USD)')
    axes[1].set_title('Actual vs Predicted')
    plt.tight_layout()
    plt.savefig('models/diagnostic_plots.png')
    plt.close()

def main():
    print("Loading data...")
    X, y, scaler, feature_cols = load_and_prepare_data()
    print("Fitting Ridge regression model...")
    model = Ridge(alpha=1.0)
    model.fit(X, y)
    y_pred = model.predict(X)
    evaluate_model(y, y_pred)
    print("\nSaving model artifacts...")
    joblib.dump(model, 'models/ridge_model.joblib')
    joblib.dump(scaler, 'models/scaler.joblib')
    joblib.dump(feature_cols, 'models/ridge_features.joblib')
    print("\nDone! Model artifacts saved in 'models' directory.")
    print("Generated files:")
    print("- models/ridge_model.joblib (trained Ridge model)")
    print("- models/scaler.joblib (feature scaler)")
    print("- models/ridge_features.joblib (feature list)")
    print("- models/diagnostic_plots.png (model diagnostics)")

if __name__ == "__main__":
    main() 