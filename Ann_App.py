# Integrating ANN model with Streamlit web app

import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# -----------------------------
# Load dataset to rebuild scaler
# -----------------------------
df = pd.read_csv("Churn_Modelling.csv")

X = df.drop(["RowNumber","CustomerId","Surname","Exited"], axis=1)

# Recreate encoders
label_encoder_gender = LabelEncoder()
X["Gender"] = label_encoder_gender.fit_transform(X["Gender"])

onehot_encoder_geo = OneHotEncoder(sparse_output=False)
geo_encoded = onehot_encoder_geo.fit_transform(X[["Geography"]])

geo_encoded_df = pd.DataFrame(
    geo_encoded,
    columns=onehot_encoder_geo.get_feature_names_out(["Geography"])
)

X = pd.concat([X.drop("Geography", axis=1), geo_encoded_df], axis=1)

# Recreate scaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# -----------------------------
# Load ANN model
# -----------------------------
model = Sequential([
    Dense(64, activation='relu', input_shape=(12,)),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid')
])

model.load_weights("model.weights.h5")

# -----------------------------
# Streamlit UI
# -----------------------------
st.title("Customer Churn Prediction")

geography = st.selectbox("Geography", ["France","Germany","Spain"])
gender = st.selectbox("Gender", ["Male","Female"])

age = st.slider("Age", 18, 92)
tenure = st.slider("Tenure", 0, 10)

balance = st.number_input("Balance")
credit_score = st.number_input("Credit Score")
estimated_salary = st.number_input("Estimated Salary")

num_of_products = st.slider("Number of Products", 1, 4)
has_cr_card = st.selectbox("Has Credit Card", [0,1])
is_active_member = st.selectbox("Is Active Member", [0,1])

# -----------------------------
# Prepare input
# -----------------------------
gender_encoded = label_encoder_gender.transform([gender])[0]

geo_encoded = onehot_encoder_geo.transform([[geography]])

geo_encoded_df = pd.DataFrame(
    geo_encoded,
    columns=onehot_encoder_geo.get_feature_names_out(["Geography"])
)

input_data = pd.DataFrame({
    "CreditScore":[credit_score],
    "Gender":[gender_encoded],
    "Age":[age],
    "Tenure":[tenure],
    "Balance":[balance],
    "NumOfProducts":[num_of_products],
    "HasCrCard":[has_cr_card],
    "IsActiveMember":[is_active_member],
    "EstimatedSalary":[estimated_salary]
})

input_data = pd.concat([input_data, geo_encoded_df], axis=1)

input_scaled = scaler.transform(input_data)

# -----------------------------
# Prediction
# -----------------------------
if st.button("Predict"):

    prediction = model.predict(input_scaled)
    prob = prediction[0][0]

    st.subheader(f"Churn Probability: {prob:.2f}")

    if prob > 0.5:
        st.error("Customer is likely to churn")
    else:
        st.success("Customer is not likely to churn")