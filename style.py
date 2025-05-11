import streamlit as st
import pickle
import pandas as pd
import numpy as np
import requests

# Load the machine learning model and data
car = pd.read_csv('cleaned_csv.csv')

# Set page config
st.set_page_config(page_title="True Value", page_icon="🚗", layout="wide")

# Title with a better look
st.title("🚗 True Value")
st.markdown("""
    This app predicts the price of a car based on various factors like the company, model, year, fuel type, 
    and kilometers driven. Select the options below to get your car price prediction!
""")

# Create a two-column layout for input fields
col1, col2 = st.columns(2)

# Company selection
with col1:
    company = st.selectbox('Select Company', sorted(car['company'].unique()))

# Filter car models based on the selected company being part of the car model name
filtered_car_models = car[car['name'].str.contains(company, case=False, na=False)]['name'].unique()

# Car Model selection
with col2:
    car_model = st.selectbox('Select Car Model', sorted(filtered_car_models))

# Create a second row for year, fuel type, and kilometers driven
col3, col4 = st.columns(2)

# Year selection
with col3:
    year = st.selectbox('Select Year', sorted(car['year'].unique(), reverse=True))

# Fuel type selection
with col4:
    fuel_type = st.selectbox('Select Fuel Type', car['fuel_type'].unique())

# Kilometers driven input
kilo_driven = st.number_input('Enter Kilometers Driven', min_value=0, step=1000)

# Button to trigger prediction
if st.button('Predict Price'):
    # Make a POST request to FastAPI backend for prediction
    data = {
        'company': str(company),
        'car_model': str(car_model),
        'year': int(year),               # Convert to native int
        'fuel_type': str(fuel_type),
        'kilo_driven': int(kilo_driven)  # Convert to native int
    }

    response = requests.post('https://trueride-production.up.railway.app/predict', json=data)
    if response.status_code == 200:
        prediction = response.json()
        st.success(f"Predicted Price: ₹{prediction['prediction']:,.2f}")
    else:
        st.error(f"Prediction failed. Status code: {response.status_code}")
        st.json(response.json())  # Optional: for debugging

# Add a footer or additional information
st.markdown("""
    ---
    <p style='text-align: center; font-size: 14px;'>Developed by Devansh Singh</p>
    <p style='text-align: center; font-size: 12px;'>Contact: Devanshs288.ds@gmail.com</p>
""", unsafe_allow_html=True)
