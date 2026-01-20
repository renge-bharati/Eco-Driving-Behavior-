import streamlit as st
import joblib
import numpy as np

# Load model files
model = joblib.load("models/eco_model_small.pkl")
label_encoder = joblib.load("models/label_encoder.pkl")
features = joblib.load("models/model_features.pkl")

st.title("🚗 Eco Driving Score Classification")

st.write("Enter vehicle driving parameters:")

# Input fields
rpm_variation = st.number_input("RPM Variation", 0.0, 10000.0)
harsh_braking_count = st.number_input("Harsh Braking Count", 0, 100)
idling_time = st.number_input("Idling Time (minutes)", 0.0, 500.0)
fuel_consumption = st.number_input("Fuel Consumption", 0.0, 50.0)
acceleration_smoothness = st.number_input("Acceleration Smoothness", 0.0, 10.0)

if st.button("Predict Eco Class"):
    input_data = np.array([
        rpm_variation,
        harsh_braking_count,
        idling_time,
        fuel_consumption,
        acceleration_smoothness
    ]).reshape(1, -1)

    prediction = model.predict(input_data)
    result = label_encoder.inverse_transform(prediction)

    st.success(f"🌱 Eco Driving Category: **{result[0]}**")
