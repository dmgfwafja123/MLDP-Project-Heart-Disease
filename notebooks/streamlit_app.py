# ==========================================
# Heart Disease Prediction App
# Streamlit Application
# ==========================================

import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Set page configuration (must be first Streamlit command)
st.set_page_config(
    page_title="Heart Disease Predictor",
    page_icon="❤️",
    layout="wide"
)

# ==========================================
# LOAD MODEL AND PREPROCESSING OBJECTS
# ==========================================

@st.cache_resource  # Cache the model to load only once
def load_model():
    """Load the trained model and preprocessing objects"""
    import os
    
    # Get the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Load files from the same directory as the script
    model_path = os.path.join(script_dir, "heart_disease_model.pkl")
    features_path = os.path.join(script_dir, "feature_columns.pkl")
    categorical_path = os.path.join(script_dir, "categorical_columns.pkl")
    
    model = joblib.load(model_path)
    feature_columns = joblib.load(features_path)
    categorical_columns = joblib.load(categorical_path)
    
    return model, feature_columns, categorical_columns

# Load everything
model, feature_columns, categorical_columns = load_model()

# ==========================================
# APP HEADER AND INTRODUCTION
# ==========================================

st.title("❤️ Heart Disease Prediction System")
st.markdown("""
This application predicts the likelihood of heart disease based on clinical parameters.
Please enter the patient information below to get a prediction.

**Disclaimer:** This is a predictive tool and should not replace professional medical advice.
Always consult with a healthcare provider for medical decisions.
""")

st.divider()

# ==========================================
# USER INPUT SECTION
# ==========================================

st.header("📋 Patient Information")

# Create two columns for better layout
col1, col2 = st.columns(2)

with col1:
    st.subheader("Demographics")
    
    age = st.slider(
        "Age (years)",
        min_value=20,
        max_value=100,
        value=50,
        help="Patient's age in years"
    )
    
    gender = st.selectbox(
        "Gender",
        options=["Female", "Male"],
        help="Patient's biological gender"
    )
    gender_encoded = 1 if gender == "Male" else 0

with col2:
    st.subheader("Vital Signs")
    
    restingBP = st.slider(
        "Resting Blood Pressure (mm Hg)",
        min_value=90,
        max_value=200,
        value=120,
        help="Blood pressure at rest"
    )
    
    maxheartrate = st.slider(
        "Maximum Heart Rate Achieved",
        min_value=70,
        max_value=220,
        value=150,
        help="Maximum heart rate during exercise"
    )

# Create another row
col3, col4 = st.columns(2)

with col3:
    st.subheader("Blood Tests")
    
    serumcholestrol = st.slider(
        "Serum Cholesterol (mg/dl)",
        min_value=100,
        max_value=600,
        value=200,
        help="Cholesterol level in blood"
    )
    
    fastingbloodsugar = st.selectbox(
        "Fasting Blood Sugar > 120 mg/dl",
        options=["No", "Yes"],
        help="Whether fasting blood sugar exceeds 120 mg/dl"
    )
    fastingbloodsugar_encoded = 1 if fastingbloodsugar == "Yes" else 0

with col4:
    st.subheader("Clinical Observations")
    
    chestpain = st.selectbox(
        "Chest Pain Type",
        options=[
            "Typical Angina",
            "Atypical Angina", 
            "Non-anginal Pain",
            "Asymptomatic"
        ],
        help="Type of chest pain experienced"
    )
    chestpain_map = {
        "Typical Angina": 0,
        "Atypical Angina": 1,
        "Non-anginal Pain": 2,
        "Asymptomatic": 3
    }
    chestpain_encoded = chestpain_map[chestpain]
    
    exerciseangia = st.selectbox(
        "Exercise Induced Angia",
        options=["No", "Yes"],
        help="Does exercise cause chest pain?"
    )
    exerciseangia_encoded = 1 if exerciseangia == "Yes" else 0

# Another row
col5, col6 = st.columns(2)

with col5:
    restingrelectro = st.selectbox(
        "Resting ECG Results",
        options=[
            "Normal",
            "ST-T Wave Abnormality",
            "Left Ventricular Hypertrophy"
        ],
        help="Resting electrocardiogram results"
    )
    restingrelectro_map = {
        "Normal": 0,
        "ST-T Wave Abnormality": 1,
        "Left Ventricular Hypertrophy": 2
    }
    restingrelectro_encoded = restingrelectro_map[restingrelectro]
    
    oldpeak = st.slider(
        "ST Depression (Oldpeak)",
        min_value=0.0,
        max_value=6.0,
        value=1.0,
        step=0.1,
        help="ST depression induced by exercise"
    )

with col6:
    slope = st.selectbox(
        "Slope of Peak Exercise ST Segment",
        options=[
            "Upsloping",
            "Flat",
            "Downsloping"
        ],
        help="Slope of the peak exercise ST segment"
    )
    slope_map = {
        "Upsloping": 1,
        "Flat": 2,
        "Downsloping": 3
    }
    slope_encoded = slope_map[slope]
    
    noofmajorvessels = st.selectbox(
        "Number of Major Vessels (0-3)",
        options=[0, 1, 2, 3],
        help="Number of major vessels colored by fluoroscopy"
    )

st.divider()

# ==========================================
# PREDICTION SECTION
# ==========================================

st.header("🔮 Get Prediction")

if st.button("Predict Heart Disease Risk", type="primary", use_container_width=True):
    
    # Create input DataFrame (raw data)
    input_data_raw = pd.DataFrame({
        'age': [age],
        'gender': [gender_encoded],
        'restingBP': [restingBP],
        'serumcholestrol': [serumcholestrol],
        'fastingbloodsugar': [fastingbloodsugar_encoded],
        'chestpain': [chestpain_encoded],
        'restingrelectro': [restingrelectro_encoded],
        'maxheartrate': [maxheartrate],
        'exerciseangia': [exerciseangia_encoded],
        'oldpeak': [oldpeak],
        'slope': [slope_encoded],
        'noofmajorvessels': [noofmajorvessels]
    })
    
    # Apply One-Hot Encoding (SAME as training!)
    input_data_encoded = pd.get_dummies(
        input_data_raw,
        columns=categorical_columns,
        drop_first=True
    )
    
    # Ensure all columns from training are present
    for col in feature_columns:
        if col not in input_data_encoded.columns:
            input_data_encoded[col] = 0
    
    # Reorder columns to match training data
    input_data_encoded = input_data_encoded[feature_columns]
    
    # Make prediction
    prediction = model.predict(input_data_encoded)[0]
    probability = model.predict_proba(input_data_encoded)[0]
    
    # Display results
    st.divider()
    st.header("📊 Results")
    
    # Create columns for results
    result_col1, result_col2 = st.columns(2)
    
    with result_col1:
        if prediction == 1:
            st.error("⚠️ **HIGH RISK** - Heart Disease Detected")
            st.markdown(f"""
            The model predicts a **{probability[1]*100:.1f}% probability** of heart disease.
            
            **Recommendation:** Consult a cardiologist immediately for further evaluation.
            """)
        else:
            st.success("✅ **LOW RISK** - No Heart Disease Detected")
            st.markdown(f"""
            The model predicts a **{probability[0]*100:.1f}% probability** of no heart disease.
            
            **Recommendation:** Maintain a healthy lifestyle and regular check-ups.
            """)
    
    with result_col2:
        st.metric(
            label="Disease Probability",
            value=f"{probability[1]*100:.1f}%",
            delta=f"{probability[1]*100 - 50:.1f}% from average"
        )
        
        st.metric(
            label="No Disease Probability",
            value=f"{probability[0]*100:.1f}%"
        )
    
    # Show warning
    st.warning("""
    **Important Notice:**
    - This prediction is based on a machine learning model and should not be used as a sole diagnostic tool.
    - Always consult with qualified healthcare professionals for medical advice.
    - This tool is for educational and screening purposes only.
    """)
    
    # Optional: Show input summary
    with st.expander("📋 View Input Summary"):
        st.write("Patient Information Submitted:")
        st.json({
            "Age": age,
            "Gender": gender,
            "Resting BP": restingBP,
            "Cholesterol": serumcholestrol,
            "Max Heart Rate": maxheartrate,
            "Chest Pain": chestpain,
            "Exercise Angia": exerciseangia,
            "Fasting Blood Sugar > 120": fastingbloodsugar,
            "Resting ECG": restingrelectro,
            "Oldpeak": oldpeak,
            "Slope": slope,
            "Major Vessels": noofmajorvessels
        })

# ==========================================
# FOOTER
# ==========================================

st.divider()
st.markdown("""
---
**Model Information:**
- Algorithm: Random Forest Classifier
- Accuracy: ~93% (on test set)
- Dataset: Cardiovascular Disease Dataset
- Developer: Lim Hong Yu - Temasek Polytechnic

*Last Updated: February 2026*
""")
