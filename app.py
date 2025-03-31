import streamlit as st
import pandas as pd
import pickle
import tensorflow as tf

# Function to load model and preprocessor with caching
@st.cache_resource
def get_model_and_preprocessor():
    model_path = 'tf_bridge_model.h5'
    preprocessor_path = 'preprocessor.pkl'

    # Load preprocessor
    with open(preprocessor_path, 'rb') as file:
        preprocessor = pickle.load(file)
    
    # Load model without compiling
    model = tf.keras.models.load_model(model_path, compile=False)
    
    return model, preprocessor

# Load assets
model, preprocessor = get_model_and_preprocessor()

# UI
st.title("Bridge Maximum Load Prediction")
st.write("Input the bridge details below to predict its maximum load capacity (in tons).")

# Input form
span_ft = st.number_input("Span (ft):", min_value=0.0, value=250.0)
deck_width_ft = st.number_input("Deck Width (ft):", min_value=0.0, value=40.0)
age_years = st.number_input("Age (Years):", min_value=0, value=20)
num_lanes = st.number_input("Number of Lanes:", min_value=1, value=4)
condition_rating = st.slider("Condition Rating (1 to 5):", 1, 5, 4)
material = st.selectbox("Material:", options=["Steel", "Concrete", "Composite"])

# Run prediction
if st.button("Predict Maximum Load"):
    input_dict = {
        "Span_ft": [span_ft],
        "Deck_Width_ft": [deck_width_ft],
        "Age_Years": [age_years],
        "Num_Lanes": [num_lanes],
        "Condition_Rating": [condition_rating],
        "Material": [material]
    }
    
    input_df = pd.DataFrame(input_dict)
    input_processed = preprocessor.transform(input_df)
    prediction = model.predict(input_processed)

    st.success(f"Predicted Maximum Load Capacity: {prediction[0][0]:.2f} tons")
