import streamlit as st
import pandas as pd
import numpy as np
from pickle import load

# Load the trained XGBoost model
model = load(open("MedicalCost_Prediction_XGBoost.pkl", "rb"))

# App configuration
st.set_page_config(page_title="Medical Cost Estimator", page_icon="💰", layout="centered")

# Main title
st.title("💡 Medical Insurance Cost Prediction")
st.markdown(
    """
    Welcome to the **Medical Cost Estimator** app!  
    This tool helps insurance companies and individuals **predict medical insurance costs** based on:
    - Age 🧓
    - BMI 📉
    - Number of children 👶
    - Smoking status 🚬
    - Region 📍
    """
)

# Sidebar for input
st.sidebar.header("🔧 Input Patient Details")

age = st.sidebar.slider("🧓 Age", 18, 65, 30)
bmi = st.sidebar.number_input("📉 BMI (Body Mass Index)", min_value=10.0, max_value=50.0, value=25.0)
children = st.sidebar.slider("👶 Number of Children", 0, 5, 1)
smoker = st.sidebar.radio("🚬 Do you Smoke?", ["No", "Yes"])
region = st.sidebar.selectbox("📍 Region", ["northeast", "southeast", "southwest", "northwest"])

# Info for user
with st.sidebar.expander("ℹ️ What is BMI?"):
    st.write("BMI is a measure of body fat based on height and weight that applies to adult men and women.")

# Feature engineering
bmi_cap = np.clip(bmi, 15.0, 40.0)
smoker_yes = 1 if smoker == "Yes" else 0
region_southeast = 1 if region == "southeast" else 0

# Prepare input
input_df = pd.DataFrame({
    'age': [age],
    'children': [children],
    'bmi_cap': [bmi_cap],
    'smoker_yes': [smoker_yes],
    'region_southeast': [region_southeast]
})

# Prediction
if st.button("📊 Predict Medical Cost"):
    log_prediction = model.predict(input_df)[0]
    predicted_cost = np.expm1(log_prediction)

    st.subheader("🧾 Predicted Medical Expense")
    st.success(f"💰 Estimated Cost: ₹ {predicted_cost:,.2f}")

    with st.expander("🔍 See Prediction Details"):
        st.write("Input Features Used:")
        st.json(input_df.to_dict(orient='records')[0])

        st.markdown("""
        - The model used here is **XGBoost Regressor**.
        - It is trained on historical data and provides the best accuracy.
        """)

# Footer
st.markdown("---")
st.markdown("© 2025 Medical Cost Estimator | Built with Streamlit")
