import pandas as pd
from sklearn.preprocessing import StandardScaler
import re
import matplotlib.pyplot as plt

def load_and_prepare_data():
    df = pd.read_csv('International_Education_Costs.csv')

    # Compute total cost (sum of all cost components)
    df['Total_Cost_USD'] = (
        df['Tuition_USD'] +
        df['Rent_USD'] * 12 +  # Annualize rent
        df['Insurance_USD'] +
        df['Visa_Fee_USD']
    )

    # Select relevant columns
    cols_to_keep = [
        'Country', 'City', 'Level', 'Duration_Years', 'Tuition_USD',
        'Living_Cost_Index', 'Rent_USD', 'Insurance_USD', 'Visa_Fee_USD', 'Total_Cost_USD'
    ]
    cols_to_keep = [col for col in cols_to_keep if col in df.columns]
    df = df[cols_to_keep]
    df = df.dropna()

    # One-hot encode City and Level
    df = pd.get_dummies(df, columns=['City', 'Level'], drop_first=True)

    # Clean column names
    def clean_column_name(col):
        clean = re.sub(r'[^a-zA-Z0-9_]', '_', col)
        if not clean[0].isalpha() and clean[0] != '_':
            clean = 'x_' + clean
        clean = re.sub(r'_+', '_', clean)
        clean = clean.strip('_')
        return clean
    df.columns = [clean_column_name(col) for col in df.columns]

    print("Columns after cleaning:", df.columns.tolist())

    # Features and target
    feature_cols = [col for col in df.columns if col not in ['Country', 'Total_Cost_USD']]
    X = df[feature_cols]
    y = df['Total_Cost_USD']

    # Scale numeric features
    numeric_features = [
        'Duration_Years', 'Tuition_USD', 'Living_Cost_Index',
        'Rent_USD', 'Insurance_USD', 'Visa_Fee_USD'
    ]
    numeric_features = [col for col in numeric_features if col in X.columns]
    print("Numeric features to scale:", numeric_features)
    print("Dtypes before scaling:\n", X[numeric_features].dtypes)
    # Convert to numeric, coerce errors to NaN
    X[numeric_features] = X[numeric_features].apply(pd.to_numeric, errors='coerce')
    print("Non-numeric values in numeric_features (should be 0):\n", X[numeric_features].isnull().sum())
    scaler = StandardScaler()
    if numeric_features:
        X[numeric_features] = scaler.fit_transform(X[numeric_features])

    print(df.columns.tolist())
    print(df.head())
    print(numeric_features)
    print(X[numeric_features].head())

    return X, y, scaler, feature_cols

if __name__ == "__main__":
    X, y, scaler, feature_cols = load_and_prepare_data()
    print("\nFeature columns:")
    print(feature_cols)
    print("\nShape of X:", X.shape)
    print("\nFirst few rows:")
    print(X.head())

    # Example values for demonstration
    uni = "Example University"
    city_sel = "Example City"
    program = "Example Program"
    level = "Bachelor"
    country = "Example Country"
    duration = 4
    total_cost = 50000
    default_uni_city = "Example University"
    default_city = "Example City"
    options = ["Example University, Example City"]  # Example options list

    # Plotting example
    fig, ax = plt.subplots()
    full_title = (
        f"Cost Breakdown for {uni}, {city_sel}\n"
        f"{program} | {level} | {country}\n"
        f"Duration: {duration} years | Total Cost: ${total_cost:,.0f}"
    )
    ax.set_title(full_title, fontsize=16, pad=30)

    if default_uni_city and default_city:
        default_option = f"{default_uni_city}, {default_city}"
        options_list = list(options)
        if default_option in options_list:
            default_index = int(options_list.index(default_option))
        else:
            default_index = 0
    else:
        default_index = 0

    tuition = float(tuition)
    duration = float(duration)
    total_tuition = tuition * duration

    total_tuition = float(total_tuition)
    total_cost = float(total_cost)

    if default_uni_city and default_city:
        default_option = f"{default_uni_city}, {default_city}"
        options_list = list(options)
        if default_option in options_list:
            default_index = int(options_list.index(default_option))
        else:
            default_index = 0
    else:
        default_index = 0

    rent = float(df.loc[0, 'Rent_USD'])
    annual_housing = rent * 12
    total_housing = annual_housing * duration

    if default_uni_city and default_city:
        default_option = f"{default_uni_city}, {default_city}"
        options_list = list(options)
        if default_option in options_list:
            default_index = int(options_list.index(default_option))
        else:
            default_index = 0
    else:
        default_index = 0 