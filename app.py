import streamlit as st
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import pandas as pd
import pickle

# --- 1. LOAD MODELS AND ENCODERS ---
# It's a good practice to wrap this in a function with st.cache_resource
# to prevent reloading on every interaction.

@st.cache_resource
def load_resources():
    """Loads the trained model, encoders, and scaler."""
    model = tf.keras.models.load_model('model.h5')
    with open('label_encoder_gender.pkl', 'rb') as file:
        label_encoder_gender = pickle.load(file)
    with open('onehot_encoder_geo.pkl', 'rb') as file:
        onehot_encoder_geo = pickle.load(file)
    with open('scaler.pkl', 'rb') as file:
        scaler = pickle.load(file)
    return model, label_encoder_gender, onehot_encoder_geo, scaler

model, label_encoder_gender, onehot_encoder_geo, scaler = load_resources()


# --- 2. STREAMLIT APP LAYOUT ---
st.title("Customer Churn Prediction")
st.markdown("Enter the customer's details to predict whether they will churn.")

# Create columns for a more organized layout
col1, col2 = st.columns(2)

with col1:
    geography = st.selectbox('Geography', onehot_encoder_geo.categories_[0])
    gender = st.selectbox('Gender', label_encoder_gender.classes_)
    age = st.slider('Age', 18, 92, 38) # Added a default value
    tenure = st.slider('Tenure (years)', 0, 10, 5)
    num_of_products = st.slider('Number Of Products', 1, 4, 1)


with col2:
    credit_score = st.number_input('Credit Score', min_value=300, max_value=850, value=650)
    balance = st.number_input('Balance', value=0.0, format="%.2f")
    estimated_salary = st.number_input('Estimated Salary', value=50000.0, format="%.2f")
    has_cr_card = st.selectbox('Has Credit Card?', [1, 0], format_func=lambda x: 'Yes' if x == 1 else 'No')
    is_active_member = st.selectbox('Is Active Member?', [1, 0], format_func=lambda x: 'Yes' if x == 1 else 'No')


# --- 3. PREPARE INPUT DATA FOR PREDICTION ---
if st.button('Predict Churn'):
    # Create the initial DataFrame from user inputs
    input_data = pd.DataFrame({
        'CreditScore': [credit_score],
        'Gender': [label_encoder_gender.transform([gender])[0]],
        'Age': [age],
        'Tenure': [tenure],
        'Balance': [balance],
        'NumOfProducts': [num_of_products],
        'HasCrCard': [has_cr_card],  # <-- CORRECTION 1: Fixed typo from 'HsCrCard'
        'IsActiveMember': [is_active_member],
        'EstimatedSalary': [estimated_salary]
    })

    # One-hot encode the 'Geography' feature
    geo_encoded = onehot_encoder_geo.transform([[geography]]).toarray()
    geo_encoded_df = pd.DataFrame(geo_encoded, columns=onehot_encoder_geo.get_feature_names_out(['Geography']))

    # Combine the numerical and encoded categorical features
    input_data = pd.concat([input_data.reset_index(drop=True), geo_encoded_df], axis=1)

    # --- CORRECTION 2: Ensure the column order matches the scaler's expectations ---
    # This is a robust way to prevent errors if the order of inputs changes.
    try:
        expected_columns = scaler.get_feature_names_out()
        input_data = input_data[expected_columns]

        # Scale the data
        input_data_scaled = scaler.transform(input_data)

        # --- 4. PREDICT AND DISPLAY RESULT ---
        prediction = model.predict(input_data_scaled)
        prediction_proba = prediction[0][0]

        st.subheader("Prediction Result")
        if prediction_proba > 0.5:
            st.error(f'The Customer is likely to Churn (Probability: {prediction_proba:.2%})')
        else:
            st.success(f'The Customer is not likely to Churn (Probability: {1-prediction_proba:.2%})')

    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")
        st.warning("Please ensure all input values are correct. The column order or names might not match the training data.")
