import streamlit as st
import pandas as pd
import numpy as np
import pickle

st.set_page_config(page_title="COVID-19 Mortality Rate Predictor", page_icon="🦠", layout="centered")

# Load the trained model
@st.cache_data
def load_model():
    with open('models/best_dt_model.pkl', 'rb') as f:
        model = pickle.load(f)
    return model    

model = load_model()

st.markdown(
    """
    <h1 style='text-align: center; color: #d6336c;'>COVID-19 Mortality Rate Prediction</h1>
    <p style='text-align: center; font-size: 18px;'>
    Predict the COVID-19 mortality rate for a country or region using key demographic, health, and economic indicators.
    </p>
    """, unsafe_allow_html=True
)

st.image("https://images.unsplash.com/photo-1584036561566-baf8f5f1b144", use_container_width =True)

st.markdown("### Enter Country-Level Features")

# List your actual features from your notebook (replace below with your real feature names)
feature_names = [
    "population_density", "median_age", "gdp_per_capita", "cardiovasc_death_rate",
    "diabetes_prevalence", "hospital_beds_per_thousand", "life_expectancy",
    "human_development_index", "stringency_index", "handwashing_facilities",
    "female_smokers", "male_smokers", "continent"
]

# Input form
with st.form("input_form"):
    col1, col2 = st.columns(2)
    with col1:
        population = st.number_input("Population", min_value=0.0, value=100.0)
        total_cases_per_million = st.number_input("Total Cases per Million", min_value=0.0, value=1000.0)
        median_age = st.number_input("Median Age", min_value=0.0, value=30.0)
        gdp_per_capita = st.number_input("GDP per Capita", min_value=0.0, value=10000.0)
        extreme_poverty = st.number_input("Extreme Poverty (%)", min_value=0.0, max_value=100.0, value=10.0)
        cardiovasc_death_rate = st.number_input("Cardiovascular Death Rate", min_value=0.0, value=200.0)
        diabetes_prevalence = st.number_input("Diabetes Prevalence (%)", min_value=0.0, value=5.0)
        hospital_beds_per_thousand = st.number_input("Hospital Beds per Thousand", min_value=0.0, value=2.0)
    with col2:
        life_expectancy = st.number_input("Life Expectancy", min_value=0.0, value=70.0)
        human_development_index = st.number_input("Human Development Index", min_value=0.0, max_value=1.0, value=0.7)
        stringency_index = st.number_input("Stringency Index", min_value=0.0, max_value=100.0, value=50.0)
        female_smokers = st.number_input("Female Smokers", min_value=0.0, max_value=100.0, value=10.0)
        male_smokers = st.number_input("Male Smokers", min_value=0.0, max_value=100.0, value=30.0)
        continent = st.selectbox("Continent", ["Africa", "Asia", "Europe", "North America", "Oceania", "South America"])
    submitted = st.form_submit_button("Predict")

input_data = pd.DataFrame([{
    "population": population,
    "median_age": median_age,
    "gdp_per_capita": gdp_per_capita,
    "cardiovasc_death_rate": cardiovasc_death_rate,
    "extreme_poverty": extreme_poverty,
    "diabetes_prevalence": diabetes_prevalence,
    "hospital_beds_per_thousand": hospital_beds_per_thousand,
    "life_expectancy": life_expectancy,
    "human_development_index": human_development_index,
    "stringency_index": stringency_index,
    "female_smokers": female_smokers,
    "male_smokers": male_smokers,
    "continent": continent
}])

if submitted:
    st.markdown("### Prediction Result")
    prediction = model.predict(input_data)
    st.success(f"Predicted COVID-19 Mortality Rate: {prediction[0]:.2f}%")

    st.markdown(
        """
        <div style='background-color: #f8f9fa; padding: 15px; border-radius: 10px;'>
        <b>Note:</b> This prediction is based on the input features and the trained Decision Tree model. For more accurate results, ensure all inputs reflect real country data.
        </div>
        """, unsafe_allow_html=True
    )

st.markdown("---")
