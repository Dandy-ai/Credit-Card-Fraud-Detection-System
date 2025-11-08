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
credit_card_number = st.text_input("Credit Card Number", max_chars=16, 
                                   placeholder="1234 5678 9012 3456")
if credit_card_number:
    cc_clean = credit_card_number.replace(" ", "")
    
    if not cc_clean.isdigit():
        st.error("Please enter only digits")
    else:
        st.success("Card Format looks good")
        cc_numeric = int(cc_clean)
amount = st.number_input("Transaction Amountd($)", min_value=0.0, value=0.00)
time = st.number_input("Transaction Time (hours)", min_value=0, max_value=23)
is_night_transaction = st.number_input("Transaction by Night", min_value=0, max_value=1)

# Create Dataframe with same structure as training data
sample_data = pd.DataFrame([[credit_card_number, amount, time, is_night_transaction]], 
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
st.caption("Developed by Okafor Gift Chukwudi - IT Industrial Training Project on Data Analytics & Machine Learning.")




