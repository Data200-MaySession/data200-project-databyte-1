import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import io

st.set_page_config(page_title="Country Comparison", page_icon="🌍")
st.title("🌍 Country-wise Cost Comparison")

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

def main():
    df = load_data()
    st.write("""
    Compare the average total cost of studying in different countries for your chosen degree level and major.
    """)
    # User selects level and program
    levels = sorted(df['Level'].unique())
    level = st.selectbox("Select Degree Level", levels)
    programs = sorted(df[df['Level'] == level]['Program'].unique())
    program = st.selectbox("Select Major/Program", programs)
    filtered = df[(df['Level'] == level) & (df['Program'] == program)]
    if filtered.empty:
        st.warning("No data for this selection.")
        return
    # Group by country and compute average total cost
    country_costs = filtered.groupby('Country')['Total_Cost_USD'].mean().sort_values(ascending=False)
    # Bar chart
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(country_costs.index, country_costs.values, color='skyblue')
    ax.set_ylabel('Average Total Cost (USD)')
    ax.set_xlabel('Country')
    ax.set_title(f"Average Total Cost by Country\n{program} | {level}")
    ax.tick_params(axis='x', rotation=60)
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
        file_name=f"country_comparison_{program}_{level}.png",
        mime="image/png"
    )

if __name__ == "__main__":
    main() 