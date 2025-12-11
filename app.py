import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load trained model and scaler
model = joblib.load('credit card fraud detection model.pkl')
scaler = joblib.load('scaler.pkl')

# App Title
st.title("Credit Card Fraud Detection System")
st.write("This web app predicts whether a transaction is **Fraudulent** or **Legitimate** using a trained Machine Learning Model.")

# Create user inputs
st.header("Enter Transaction Details")

# Input fields
credit_card_number = st.text_input("Credit Card Number", placeholder="1234567812345678")
cc_numeric = None
if credit_card_number.isdigit():
    cc_numeric = int(credit_card_number)
    masked = "**** **** **** " + credit_card_number[-4:]
    st.info(f"Card entered: {masked}")
else:
    st.warning("Please enter digits only")
amount = st.number_input("Transaction Amount ($)", min_value=0.0, value=0.00)
time = st.number_input("Transaction Time (hours)", min_value=0, max_value=23)
is_night_transaction = 1 if st.checkbox("Transaction by Night") else 0

# Create Dataframe with same structure as training data (with numeric cc_num)
sample_data = pd.DataFrame([[cc_numeric, amount, time, is_night_transaction]], 
                           columns=['cc_num', 'amt', 'trans_time_hrs', 'trans_time_is_night'])

# Scale input data
scaled_data = scaler.transform(sample_data)

# Predict
if st.button("Predict Transaction"):
    prediction = model.predict(scaled_data)
    
    st.subheader("Prediction Result")
    if prediction[0] == 1:
        st.error("Fraudulent Transaction Detected!")
    else:
        st.success("Legitimate Transaction.")
        
st.markdown("---")
st.caption("Developed by Okafor Gift Chukwudi - IT (Industrial Training) Project on Data Analytics & Machine Learning.")





