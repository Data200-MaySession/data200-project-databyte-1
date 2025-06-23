import streamlit as st

st.set_page_config(page_title="International Education Cost Predictor", page_icon="🎓")
st.title("🎓 International Education Cost Predictor")

st.markdown("""
:earth_asia: **Welcome to the International Education Cost Predictor!**  
This app helps you estimate and compare the total cost of studying abroad for different programs, cities, and universities.

---

### :rocket: How to Use

1. Select **International Education Cost App** from the sidebar.
2. Choose your **Country**, **Degree Level**, and **Major/Program**.
3. Browse a table of all matching cities/universities and their estimated total costs.
4. Use the **budget slider** to filter options by your budget.
5. Click on a university/city for a detailed cost breakdown (tuition, rent, insurance, visa, etc.).

---

### 📊 Project Background

- Developed for **DATA200: Applied Statistical Analysis**.
- The goal: Help students and families make informed decisions about international education costs using real data and machine learning.

---

### 📚 Dataset

- **Source:** [Kaggle - Cost of International Education](https://www.kaggle.com/datasets/adilshamim8/cost-of-international-education)
- **Features:** Country, City, University, Program, Level, Tuition, Rent, Insurance, Visa, and more.
- **Target:** Total estimated cost (tuition + rent + insurance + visa, etc.)

---

### 🤖 About the Model

We use a **Ridge Regression** model (a regularized linear regression) for robust, interpretable cost prediction:

- Handles many features (including one-hot encoded cities/levels) without overfitting.
- Provides fast, reliable estimates for a wide range of programs and locations.

Example structure:

```python
Pipeline(steps=[
    ("scaler", StandardScaler()),
    ("regressor", Ridge(alpha=1.0))
])
```

---

### :information_source: Disclaimer

This app provides **estimates** based on historical data. Actual costs may vary depending on program, year, and personal circumstances.

---
""") 