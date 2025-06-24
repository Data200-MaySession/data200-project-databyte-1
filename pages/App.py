import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# Set page config
st.set_page_config(
    page_title="International Education Cost Predictor",
    page_icon="🎓",
    layout="wide"
)

# Load model artifacts
@st.cache_resource
def load_artifacts():
    model = joblib.load('models/ridge_model.joblib')
    features = joblib.load('models/ridge_features.joblib')
    scaler = joblib.load('models/scaler.joblib')
    return model, features, scaler

# Load the original dataset to get unique values
@st.cache_data
def load_data():
    df = pd.read_csv('International_Education_Costs.csv')
    df['Total_Cost_USD'] = (
        df['Tuition_USD'] +
        df['Rent_USD'] * 12 +
        df['Insurance_USD'] +
        df['Visa_Fee_USD']
    )
    return df

def clean_column_name(col):
    import re
    clean = re.sub(r'[^a-zA-Z0-9_]', '_', col)
    if not clean[0].isalpha() and clean[0] != '_':
        clean = 'x_' + clean
    clean = re.sub(r'_+', '_', clean)
    clean = clean.strip('_')
    return clean

def main():
    # Load model and data
    try:
        model, features, scaler = load_artifacts()
        df = load_data()
    except Exception as e:
        st.error(f"Error loading model artifacts: {str(e)}")
        return

    # Title and description
    st.title("🎓 International Education Cost Predictor")
    st.write("""
    Compare estimated total costs for your chosen country, degree level, and major.\nSee all available cities/universities and filter by your budget.
    """)

    # --- User Inputs ---
    # 1. Country
    countries = sorted([c for c in df['Country'].dropna().unique() if str(c).strip() not in ["", "====", "NA", "NaN"]])
    country = st.selectbox("Select Country", countries, index=0)

    # 2. Level (filtered by country)
    levels = sorted([l for l in df[df['Country'] == country]['Level'].dropna().unique() if str(l).strip() not in ["", "====", "NA", "NaN"]])
    level = st.selectbox("Select Degree Level", levels, index=0)

    # 3. Major/Program (filtered by country and level)
    programs = sorted([p for p in df[(df['Country'] == country) & (df['Level'] == level)]['Program'].dropna().unique() if str(p).strip() not in ["", "====", "NA", "NaN"]])
    program = st.selectbox("Select Major/Program", programs, index=0)

    # --- Filtered Data ---
    filtered = df[(df['Country'] == country) & (df['Level'] == level) & (df['Program'] == program)]

    if filtered.empty:
        st.warning("No programs found for this selection.")
        return

    # --- Prepare prediction table ---
    display_df = filtered.copy()
    # Prepare input for model prediction
    def prepare_input(row):
        input_dict = {
            'Country': row['Country'],
            'City': row['City'],
            'Level': row['Level'],
            'Duration_Years': row['Duration_Years'],
            'Tuition_USD': row['Tuition_USD'],
            'Living_Cost_Index': row['Living_Cost_Index'],
            'Rent_USD': row['Rent_USD'],
            'Insurance_USD': row['Insurance_USD'],
            'Visa_Fee_USD': row['Visa_Fee_USD']
        }
        input_data = pd.DataFrame([input_dict])
        input_encoded = pd.get_dummies(input_data, columns=['City', 'Level'], drop_first=True)
        input_encoded.columns = [clean_column_name(col) for col in input_encoded.columns]
        for col in features:
            if col not in input_encoded.columns:
                input_encoded[col] = 0
        input_encoded = input_encoded[features]
        numeric_cols = [
            'Duration_Years', 'Tuition_USD', 'Living_Cost_Index',
            'Rent_USD', 'Insurance_USD', 'Visa_Fee_USD'
        ]
        numeric_cols = [col for col in numeric_cols if col in input_encoded.columns]
        if numeric_cols:
            input_encoded[numeric_cols] = scaler.transform(input_encoded[numeric_cols])
        return input_encoded

    # Predict costs for all rows
    preds = []
    for idx, row in display_df.iterrows():
        X_row = prepare_input(row)
        pred = model.predict(X_row)[0]
        preds.append(pred)
    display_df['Estimated_Total_Cost'] = preds

    # --- Budget slider ---
    min_cost = int(display_df['Estimated_Total_Cost'].min())
    max_cost = int(display_df['Estimated_Total_Cost'].max())
    if min_cost < max_cost:
        st.write("\n#### Filter by your budget range (USD)")
        budget = st.slider("Budget Threshold", min_value=min_cost, max_value=max_cost, value=(min_cost, min(max_cost, min_cost + 50000)), step=1000)
        display_df = display_df[(display_df['Estimated_Total_Cost'] >= budget[0]) & (display_df['Estimated_Total_Cost'] <= budget[1])]
    else:
        st.info(f"Typical cost for this selection: ${min_cost:,.0f}")

    # --- Show table ---
    st.write(f"### Programs in {country} for {level} - {program}")
    show_cols = ['University', 'City', 'Estimated_Total_Cost', 'Duration_Years']
    show_cols = [col for col in show_cols if col in display_df.columns]
    st.dataframe(display_df[show_cols].sort_values('Estimated_Total_Cost'))

    # --- Detailed breakdown on selection ---
    st.write("\n#### See cost breakdown for a specific university/city:")
    options = display_df['University'] + ", " + display_df['City']
    selected = st.selectbox("Select University, City", options)
    if selected:
        uni, city_sel = selected.split(", ")
        row = display_df[(display_df['University'] == uni) & (display_df['City'] == city_sel)].iloc[0]
        with st.expander(f"Cost breakdown for {uni}, {city_sel}"):
            duration = float(pd.to_numeric(row['Duration_Years'], errors='coerce'))
            tuition = float(pd.to_numeric(row['Tuition_USD'], errors='coerce'))
            rent = float(pd.to_numeric(row['Rent_USD'], errors='coerce'))
            insurance = float(pd.to_numeric(row['Insurance_USD'], errors='coerce'))
            visa = float(pd.to_numeric(row['Visa_Fee_USD'], errors='coerce'))
            annual_housing = rent * 12
            total_tuition = tuition * duration
            total_housing = annual_housing * duration
            total_insurance = insurance * duration
            st.write(f"**Tuition:** ${total_tuition:,.2f}")
            st.write(f"**Housing (Total):** ${total_housing:,.2f} (${annual_housing:,.2f}/year)")
            st.write(f"**Insurance (Total):** ${total_insurance:,.2f}")
            st.write(f"**Visa Fee:** ${visa:,.2f}")
            st.write(f"**Duration:** {duration} years")
        # Add a button to view cost breakdown page with pre-filled values
        if st.button("View Cost Breakdown Page", key="view_cb"):
            st.session_state['cb_country'] = country
            st.session_state['cb_level'] = level
            st.session_state['cb_program'] = program
            st.session_state['cb_university'] = uni
            st.session_state['cb_city'] = city_sel
            st.success("Selections saved! Please click the 'Cost Breakdown' page in the sidebar to view the detailed chart.")
    st.info("""
    **Note:** This is an estimate based on historical data and may not reflect current costs.\nActual costs can vary based on specific programs, living arrangements, and other factors.
    """)

if __name__ == "__main__":
    main() 