import streamlit as st
import pandas as pd
import numpy as np

import pickle

# Load the trained model
@st.cache_data
def load_model():
    with open('models/best_dt_model.pkl', 'rb') as f:
        model = pickle.load(f)
    return model    

model = load_model()

st.title("COVID-19 Mortality Rate Prediction")
st.write("""
This app predicts the COVID-19 mortality rate based on various country-level features.
""")

# Input features
def user_input_features(): 
    feature1 = st.number_input('Feature 1', value=0.0)
    feature2 = st.number_input('Feature 2', value=0.0)
    feature3 = st.number_input('Feature 3', value=0.0)
    feature4 = st.number_input('Feature 4', value=0.0)
    feature5 = st.number_input('Feature 5', value=0.0)
    feature6 = st.number_input('Feature 6', value=0.0)
    feature7 = st.number_input('Feature 7', value=0.0)
    feature8 = st.number_input('Feature 8', value=0.0)
    feature9 = st.number_input('Feature 9', value=0.0)
    feature10 = st.number_input('Feature 10', value=0.0)
    feature11 = st.number_input('Feature 11', value=0.0)
    feature12 = st.number_input('Feature 12', value=0.0)
    feature13 = st.number_input('Feature 13', value=0.0)        
    data = {
        'feature1': feature1,
        'feature2': feature2,
        'feature3': feature3,
        'feature4': feature4,
        'feature5': feature5
    }
    features = pd.DataFrame(data, index=[0])
    return features

input_df = user_input_features()
st.subheader('User Input features')
st.write(input_df)

# Prediction
if st.button('Predict'):
    prediction = model.predict(input_df)
    st.subheader('Prediction')
    st.write(f'Predicted COVID-19 Mortality Rate: {prediction[0]:.2f}') 

