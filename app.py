import streamlit as st
import pandas as pd
import joblib

st.title("Tourism Package Prediction App")

model = joblib.load("models/model.pkl")

def predict(input_dict):
    df = pd.DataFrame([input_dict])
    df = pd.get_dummies(df)
    # Align with training features
    trained_features = pd.read_csv("artifacts/train_X.csv").columns
    for col in trained_features:
        if col not in df.columns:
            df[col] = 0
    df = df[trained_features]
    return model.predict(df)[0]

st.header("Enter Customer Details")
age = st.number_input("Age", min_value=18, max_value=100, value=30)
income = st.number_input("Monthly Income", min_value=0, value=50000)
passport = st.selectbox("Passport", [0,1])
owncar = st.selectbox("Own Car", [0,1])
trips = st.number_input("Number of Trips per year", min_value=0, value=2)

if st.button("Predict"):
    input_data = {
        "Age": age,
        "MonthlyIncome": income,
        "Passport": passport,
        "OwnCar": owncar,
        "NumberOfTrips": trips
    }
    result = predict(input_data)
    if result == 1:
        st.success("✅ Customer is likely to purchase the package.")
    else:
        st.error("❌ Customer is unlikely to purchase the package.")
