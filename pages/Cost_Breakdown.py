import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import io

st.set_page_config(page_title="Cost Breakdown", page_icon="💸")
st.title("💸 Cost Breakdown Pie Chart")

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
    Select a country, degree level, major, and university/city to see a pie chart of the cost breakdown.
    """)

    # User selections
    country = st.selectbox("Select Country", sorted(df['Country'].unique()), index=sorted(df['Country'].unique()).index(st.session_state.get('cb_country', sorted(df['Country'].unique())[0])))
    levels = sorted(df[df['Country'] == country]['Level'].unique())
    level = st.selectbox("Select Degree Level", levels, index=levels.index(st.session_state.get('cb_level', levels[0])))
    programs = sorted(df[(df['Country'] == country) & (df['Level'] == level)]['Program'].unique())
    program = st.selectbox("Select Major/Program", programs, index=programs.index(st.session_state.get('cb_program', programs[0])))
    filtered = df[(df['Country'] == country) & (df['Level'] == level) & (df['Program'] == program)]
    if filtered.empty:
        st.warning("No programs found for this selection.")
        return
    # University/city selection
    options = filtered['University'] + ", " + filtered['City']
    default_uni_city = st.session_state.get('cb_university', None)
    default_city = st.session_state.get('cb_city', None)
    options_list = list(options)
    if default_uni_city and default_city:
        default_option = f"{default_uni_city}, {default_city}"
        if default_option in options_list:
            default_index = int(options_list.index(default_option))
        else:
            default_index = 0
    else:
        default_index = 0
    selected = st.selectbox("Select University, City", options, index=default_index)
    if not selected:
        st.info("Please select a university/city.")
        return
    uni, city_sel = selected.split(", ")
    row = filtered[(filtered['University'] == uni) & (filtered['City'] == city_sel)].iloc[0]
    # Pie chart data
    duration = row['Duration_Years']
    tuition = row['Tuition_USD'] * duration
    rent = row['Rent_USD'] * 12 * duration
    insurance = row['Insurance_USD'] * duration
    visa = row['Visa_Fee_USD']
    labels = ['Tuition', 'Housing (Rent)', 'Insurance']
    values = [tuition, rent, insurance]
    total_cost = tuition + rent + insurance + visa
    # Pie chart with legend
    fig, ax = plt.subplots(figsize=(7, 7))
    wedges, texts, autotexts = ax.pie(
        values,
        autopct='%1.1f%%',
        startangle=90,
        counterclock=False
    )
    title = f"Cost Breakdown for {uni}, {city_sel}"
    ax.set_title(title, fontsize=16, pad=30)
    # Add subtitle/context inside the image, below the pie chart
    fig.text(
        0.5, 0.05,
        f"{program} | {level} | {country}\nDuration: {duration} years | Total Cost: ${total_cost:,.0f}",
        ha='center', fontsize=12
    )
    ax.legend(wedges, labels, title="Cost Components", loc="center left", bbox_to_anchor=(1, 0, 0.5, 1))
    st.pyplot(fig)
    # Download button for PNG
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight')
    st.download_button(
        label="Download Chart as PNG",
        data=buf.getvalue(),
        file_name=f"cost_breakdown_{uni}_{city_sel}.png",
        mime="image/png"
    )
    # Show actual values
    st.write("#### Cost Details:")
    st.write(f"**Tuition:** ${tuition:,.2f}")
    st.write(f"**Housing (Total):** ${rent:,.2f}")
    st.write(f"**Insurance (Total):** ${insurance:,.2f}")
    st.write(f"**Visa Fee:** ${visa:,.2f}")
    st.write(f"**Duration:** {duration} years")

if __name__ == "__main__":
    main() 