import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import io

st.set_page_config(page_title="Country Comparison", page_icon="🌍")
st.title("🌍 Country-wise Cost Comparison")

@st.cache_data
def load_data():
    df = pd.read_csv('International_Education_Costs.csv')
    for col in ['Tuition_USD', 'Rent_USD', 'Insurance_USD', 'Duration_Years', 'Visa_Fee_USD']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df['Total_Cost_USD'] = (
        (df['Tuition_USD'] + df['Rent_USD'] * 12 + df['Insurance_USD']) * df['Duration_Years']
        + df['Visa_Fee_USD']
    )
    return df

def main():
    df = load_data()
    st.write("""
    Compare the average total cost of studying in different countries for your chosen degree level and major.
    """)
    # User selects level only
    levels = sorted([
        l for l in df['Level'].dropna().unique()
        if isinstance(l, str) and str(l).strip() not in ["", "====", "NA", "NaN", "Level"]
    ])
    level = st.selectbox("Select Degree Level", levels)
    filtered = df[df['Level'] == level]
    if filtered.empty:
        st.warning("No data for this selection.")
        return
    # Group by country and compute average total cost, show only top 10
    country_costs = filtered.groupby('Country')['Total_Cost_USD'].mean().sort_values(ascending=False).head(10)
    # Bar chart
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(country_costs.index, country_costs.values, color='skyblue')
    ax.set_ylabel('Average Total Cost (USD)')
    ax.set_xlabel('Country')
    ax.set_title(f"Top 10 Countries by Average Total Cost\n{level}")
    ax.tick_params(axis='x', rotation=45)
    # Annotate bars
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'${height:,.0f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 8),  # 8 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=8)
    st.pyplot(fig)
    # Download button
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight')
    st.download_button(
        label="Download Chart as PNG",
        data=buf.getvalue(),
        file_name=f"country_comparison_{level}.png",
        mime="image/png"
    )

if __name__ == "__main__":
    main() 