# app.py
import streamlit as st
import numpy as np
import pickle

# Load model, scaler & feature names
model = pickle.load(open("cancer_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))
feature_names = pickle.load(open("feature_names.pkl", "rb"))

st.set_page_config(page_title="Breast Cancer Prediction", page_icon="🌸", layout="wide")

# Custom CSS for Black & Pink Theme
st.markdown(
    """
    <style>
        body {background-color: #0e1117; color: #ffffff;}
        .stButton>button {
            background-color: #ff4b8b;
            color: white;
            border-radius: 10px;
            height: 3em;
            width: 100%;
        }
        .stSlider>div>div>div>div {background: #ff4b8b;}
    </style>
    """,
    unsafe_allow_html=True
)

st.title("🌸 Breast Cancer Prediction App 🌸")
st.write("Move the sliders to set feature values and predict if the tumor is **Malignant** or **Benign**.")

# Generate 30 sliders dynamically
input_data = []
cols = st.columns(3)  # 3 sliders per row

for i, feature in enumerate(feature_names):
    with cols[i % 3]:
        val = st.slider(feature, float(0), float(10), float(1))
        input_data.append(val)

# Convert input to numpy array
features = np.array([input_data])
features_scaled = scaler.transform(features)

# Predict
if st.button("🔍 Predict"):
    prediction = model.predict(features_scaled)
    result = "❌ Malignant (Cancerous)" if prediction[0] == 0 else "✅ Benign (Non-Cancerous)"
    st.success(f"🎯 Prediction: **{result}**")


#outpu in terminal:- streamlit run .\breast_Cancer_app.py
