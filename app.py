import streamlit as st
import joblib
import numpy as np
import os

# Page config
st.set_page_config(page_title="Eco Driving Classification", layout="centered")

st.title("🚗 Eco Driving Behavior Classification")

# ---- Load model files safely ----
MODEL_PATH = "models/eco_model_small.pkl"
ENCODER_PATH = "models/label_encoder.pkl"
FEATURES_PATH = "models/model_features.pkl"

if not os.path.exists(MODEL_PATH):
    st.error("❌ Model file not found. Upload eco_model_small.pkl in models folder.")
    st.stop()

if not os.path.exists(ENCODER_PATH):
    st.error("❌ Label encoder not found.")
    st.stop()

if not os.path.exists(FEATURES_PATH):
    st.error("❌ Feature list not found.")
    st.stop()

model = joblib.load(MODEL_PATH)
label_encoder = joblib.load(ENCODER_PATH)
features = joblib.load(FEATURES_PATH)

st.success("✅ Model loaded successfully!")

st.subheader("Enter Driving Parameters")

rpm_variation = st.number_input("RPM Variation", min_value=0.0, step=1.0)
harsh_braking_count = st.number_input("Harsh Braking Count", min_value=0, step=1)
idling_time = st.number_input("Idling Time (minutes)", min_value=0.0, step=1.0)
fuel_consumption = st.number_input("Fuel Consumption", min_value=0.0, step=0.1)
acceleration_smoothness = st.number_input("Acceleration Smoothness", min_value=0.0, step=0.1)

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
