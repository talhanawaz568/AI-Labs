import streamlit as st
import pickle
import numpy as np
import os

# --- Page Config ---
st.set_page_config(page_title="Model Deployment Lab")

# --- Task 2: Load the Pre-trained Model ---
if os.path.exists('model.pkl'):
    with open('model.pkl', 'rb') as file:
        model = pickle.load(file)
else:
    st.error("Model file not found! Please run create_model.py first.")

# --- Task 1 & 3: UI and Logic ---
st.title("🚀 Machine Learning Model Deployment")
st.write("Enter the 4 features below (separated by commas) to get a prediction.")

# User Input Widget
user_input = st.text_input("Input Data (e.g., 5.1, 3.5, 1.4, 0.2):", "")

if st.button("Predict"):
    if user_input:
        try:
            # Convert string input to numpy array
            input_data = np.array([float(x) for x in user_input.split(',')])
            
            # Perform Inference
            prediction = model.predict([input_data])
            
            # Show Result
            st.success(f"The model predicts Class: {prediction[0]}")
            
            # Add a bit of flair
            if prediction[0] == 0:
                st.info("Species Predicted: Iris-Setosa")
            else:
                st.info("Species Predicted: Iris-Virginica")
                
        except ValueError:
            st.error("Please enter 4 valid numbers separated by commas.")
    else:
        st.warning("Please enter some data first.")

st.sidebar.markdown("### Lab 38: Deployment")
st.sidebar.info("This app uses Streamlit to serve a Scikit-Learn model locally.")
