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

st.image("https://images.unsplash.com/photo-1584036561566-baf8f5f1b144", use_column_width=True)

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
    pop_density = st.number_input("Population Density", min_value=0.0, value=100.0)
    median_age = st.number_input("Median Age", min_value=0.0, value=30.0)
    gdp = st.number_input("GDP per Capita", min_value=0.0, value=10000.0)
    cardio = st.number_input("Cardiovascular Death Rate", min_value=0.0, value=200.0)
    diabetes = st.number_input("Diabetes Prevalence (%)", min_value=0.0, value=5.0)
    beds = st.number_input("Hospital Beds per Thousand", min_value=0.0, value=2.0)
    life_exp = st.number_input("Life Expectancy", min_value=0.0, value=70.0)
    hdi = st.number_input("Human Development Index", min_value=0.0, max_value=1.0, value=0.7)
    stringency = st.number_input("Stringency Index", min_value=0.0, max_value=100.0, value=50.0)
    handwash = st.number_input("Handwashing Facilities (%)", min_value=0.0, max_value=100.0, value=80.0)
    female_smoke = st.number_input("Female Smokers (%)", min_value=0.0, max_value=100.0, value=10.0)
    male_smoke = st.number_input("Male Smokers (%)", min_value=0.0, max_value=100.0, value=30.0)
    continent = st.selectbox("Continent", ["Africa", "Asia", "Europe", "North America", "Oceania", "South America"])
    submitted = st.form_submit_button("Predict")

input_data = pd.DataFrame([{
    "population_density": pop_density,
    "median_age": median_age,
    "gdp_per_capita": gdp,
    "cardiovasc_death_rate": cardio,
    "diabetes_prevalence": diabetes,
    "hospital_beds_per_thousand": beds,
    "life_expectancy": life_exp,
    "human_development_index": hdi,
    "stringency_index": stringency,
    "handwashing_facilities": handwash,
    "female_smokers": female_smoke,
    "male_smokers": male_smoke,
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
st.markdown(
    """
    <small>
    <i>Powered by Decision Tree Regression. Data source: <a href='https://www.kaggle.com/datasets/sandhyakrishnan02/latest-covid-19-dataset-worldwide' target='_blank'>Kaggle COVID-19 Dataset</a></i>
    </small>
    """, unsafe_allow_html=True
)