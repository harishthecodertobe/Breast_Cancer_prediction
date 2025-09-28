# breast_cancer_app.py
import streamlit as st
import numpy as np
import pickle
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression

# Load dataset
data = load_breast_cancer()
feature_names = data.feature_names

# Train a model (or you can load a pre-trained model)
model = LogisticRegression(max_iter=10000)
model.fit(data.data, data.target)

# Streamlit UI
st.title("Breast Cancer Prediction")
st.write("Enter the tumor features to predict whether it's malignant or benign:")

# Create input fields for all 30 features dynamically
user_input = []
for feature in feature_names:
    value = st.number_input(feature, value=float(data.data[:, list(feature_names).index(feature)].mean()))
    user_input.append(value)

# Convert input to array
input_array = np.array(user_input).reshape(1, -1)

# Prediction button
if st.button("Predict"):
    prediction = model.predict(input_array)[0]
    prediction_label = data.target_names[prediction]
    st.success(f"The tumor is likely: **{prediction_label}**")
