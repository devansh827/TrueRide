import streamlit as st
import pickle
import pandas as pd
import numpy as np
import requests

# Load the machine learning model and data
car = pd.read_csv('cleaned_csv.csv')

# Set page config
st.set_page_config(page_title="TRUE VALUE", page_icon="🚗", layout="wide")

st.markdown("""
    <h1 style="text-align: center;">🚗 True Value</h1>
    <p style="text-align: center;">
        This app predicts the price of a car based on various factors like the company, model, year, fuel type, 
        and kilometers driven. Select the options below to get your car price prediction!
    </p>
""", unsafe_allow_html=True)
# Create a single-column layout for the inputs
with st.form(key='car_input_form'):
    # Company selection
    company = st.selectbox('Select Company', sorted(car['company'].unique()))

    # Filter car models based on the selected company
    filtered_car_models = car[car['name'].str.contains(company, case=False, na=False)]['name'].unique()

    # Car Model selection
    car_model = st.selectbox('Select Car Model', sorted(filtered_car_models))

    # Year selection
    year = st.selectbox('Select Year', sorted(car['year'].unique(), reverse=True))

    # Fuel type selection
    fuel_type = st.selectbox('Select Fuel Type', car['fuel_type'].unique())

    # Kilometers driven input
    kilo_driven = st.number_input('Enter Kilometers Driven', min_value=0, step=1000)

    # Submit button
    submit_button = st.form_submit_button(label='Predict Price')

# Trigger prediction when button is pressed
if submit_button:
    # Prepare data for prediction
    data = {
        'company': str(company),
        'car_model': str(car_model),
        'year': int(year),               # Convert to native int
        'fuel_type': str(fuel_type),
        'kilo_driven': int(kilo_driven)  # Convert to native int
    }

    # Make a POST request to FastAPI backend for prediction
    response = requests.post('http://127.0.0.1:8000/predict', json=data)
    if response.status_code == 200:
        prediction = response.json()
        st.success(f"Predicted Price: ₹{prediction['prediction']:,.2f}")
    else:
        st.error(f"Prediction failed. Status code: {response.status_code}")
        st.json(response.json())  # Optional: for debugging

# Add a footer or additional information
st.markdown("""
    ---
    <p style='text-align: center; font-size: 14px;'>Developed by DEVANSH SINGH</p>
    <p style='text-align: center; font-size: 12px;'>Contact: devanshs288.ds@gmail.com</p>
""", unsafe_allow_html=True)
