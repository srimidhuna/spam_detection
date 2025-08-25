import streamlit as st
import pickle
import numpy as np

with open("svm_model.pkl", "rb") as f:
    model = pickle.load(f)

st.title(" SVM Classifier Frontend")

st.subheader("Enter Feature Values:")

f1 = st.number_input("Feature 1", value=0.0)
f2 = st.number_input("Feature 2", value=0.0)
f3 = st.number_input("Feature 3", value=0.0)
f4 = st.number_input("Feature 4", value=0.0)

if st.button("Predict"):
    input_data = np.array([[f1, f2, f3, f4]])
    prediction = model.predict(input_data)
    st.success(f"Predicted Class: {prediction[0]}")