import streamlit as st
import joblib
import numpy as np

# Load the saved model
model = joblib.load('Downloads/svc_model.joblib')

# Streamlit app title
st.title("Heart Disease Prediction with SVC")

# Collect user input for each feature
age = st.number_input("Age", min_value=0, max_value=120, value=30)
sex = st.selectbox("Sex", options=["Male", "Female"])
cp = st.selectbox("Chest Pain Type (1: typical angina, 2: atypical angina, 3: non-anginal pain, 4: asymptomatic)", options=[1, 2, 3, 4])
trestbps = st.number_input("Resting Blood Pressure (mm Hg)", min_value=80, max_value=200, value=120)
restecg = st.selectbox("Resting ECG Results (0: normal, 1: ST-T wave abnormality, 2: left ventricular hypertrophy)", options=[0, 1, 2])
thalach = st.number_input("Maximum Heart Rate Achieved", min_value=60, max_value=220, value=150)
exang = st.selectbox("Exercise Induced Angina", options=[0, 1], format_func=lambda x: 'Yes' if x == 1 else 'No')
oldpeak = st.number_input("ST Depression Induced by Exercise Relative to Rest", min_value=0.0, max_value=6.0, value=1.0, step=0.1)
slope = st.selectbox("Slope of the Peak Exercise ST Segment (0: upsloping, 1: flat, 2: downsloping)", options=[0, 1, 2])
ca = st.selectbox("Number of Major Vessels Colored by Fluoroscopy (0-4)", options=[0, 1, 2, 3, 4])
thal = st.selectbox("Thalassemia (1: normal, 2: fixed defect, 3: reversible defect)", options=[1, 2, 3])

# Convert categorical inputs to numerical if necessary
sex = 1 if sex == "Male" else 0

# Create feature array for prediction
features = np.array([[age, sex, cp, trestbps, restecg, thalach, exang, oldpeak, slope, ca, thal]])

# Prediction
if st.button("Predict"):
    prediction = model.predict(features)
    if prediction[0] == 1:
        st.write("The model predicts the presence of heart disease.")
    else:
        st.write("The model predicts no heart disease.")