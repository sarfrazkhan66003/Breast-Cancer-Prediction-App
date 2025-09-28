import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import warnings

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Cancer Prediction Dashboard",
    page_icon="🌸",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning)

# --- MODEL TRAINING (Cached) ---
@st.cache_resource
def train_model():
    """Trains and returns the ML model and scaler."""
    # (The data and model training logic remains the same)
    data = {
        'age': [45, 52, 58, 65, 38, 41, 68, 72, 35, 49, 55, 61, 75, 43, 50, 57, 63, 39, 48, 69],
        'height_cm': [160, 155, 162, 158, 165, 170, 153, 150, 168, 163, 157, 161, 154, 169, 164, 156, 159, 172, 166, 152],
        'weight_kg': [70, 85, 78, 90, 65, 72, 95, 100, 60, 75, 82, 88, 105, 68, 77, 86, 92, 63, 74, 98],
        'has_diabetes': [1, 1, 0, 1, 0, 0, 1, 1, 0, 1, 1, 0, 1, 0, 1, 1, 1, 0, 0, 1],
        'has_high_bp': [1, 1, 1, 1, 0, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 1, 1, 0, 0, 1],
        'family_history': [1, 0, 1, 1, 0, 0, 1, 0, 0, 1, 0, 1, 1, 0, 0, 0, 1, 0, 1, 0],
        'age_at_menarche': [12, 14, 13, 11, 15, 13, 12, 11, 14, 13, 12, 14, 12, 13, 15, 12, 11, 14, 13, 12],
        'has_cancer': [1, 1, 0, 1, 0, 0, 1, 1, 0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 0, 1]
    }
    df = pd.DataFrame(data)
    features = ['age', 'height_cm', 'weight_kg', 'has_diabetes', 'has_high_bp', 'family_history', 'age_at_menarche']
    X = df[features]
    y = df['has_cancer']
    scaler = StandardScaler().fit(X)
    X_scaled = scaler.transform(X)
    model = LogisticRegression(random_state=42).fit(X_scaled, y)
    return model, scaler

model, scaler = train_model()

# --- SESSION STATE ---
if 'history' not in st.session_state:
    st.session_state.history = []
if 'prediction_made' not in st.session_state:
    st.session_state.prediction_made = False

# --- STYLING ---
st.markdown("""
<style>
    /* Main app background */
    .stApp {
        background-color: #F0F2F6;
    }
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background-color: #FFFFFF;
        border-right: 1px solid #e0e0e0;
    }
    
    /* --- NEW STYLES FOR SIDEBAR WIDGETS --- */
    /* Style for the main button */
    [data-testid="stButton"] > button {
        background-color: #1E88E5;
        color: white;
        border-radius: 8px;
        border: none;
        padding: 12px 0;
        transition: all 0.2s ease-in-out;
        font-weight: bold;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    [data-testid="stButton"] > button:hover {
        background-color: #1565C0; /* Darker blue on hover */
        transform: scale(1.02);
        box-shadow: 0 6px 12px rgba(0,0,0,0.15);
    }
    /* Style for dropdowns and number inputs */
    [data-testid="stSelectbox"] > div, [data-testid="stNumberInput"] > div {
        border-radius: 8px;
    }
    [data-testid="stSelectbox"] div[data-baseweb="select"] > div,
    [data-testid="stNumberInput"] div[data-baseweb="input"] > div {
        background-color: #F0F2F6; /* Match app background */
        border-radius: 8px;
        border: 1px solid #F0F2F6;
    }
    /* --- END OF NEW STYLES --- */

    /* Metric card styling */
    .metric-card {
        background-color: #FFFFFF;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        text-align: center;
        border-left: 6px solid #1E88E5;
    }
    .metric-card h3 {
        color: #555555;
        font-size: 16px;
        margin-bottom: 5px;
    }
    .metric-card p {
        color: #0D47A1;
        font-size: 28px;
        font-weight: bold;
        margin: 0;
    }
    /* Prediction card styling */
    .prediction-card {
        padding: 30px;
        border-radius: 10px;
        color: white;
        text-align: center;
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    }
    .prediction-card h2 {
        font-size: 48px;
        margin: 0;
        font-weight: bold;
    }
    .prediction-card p {
        font-size: 18px;
        margin-top: 5px;
    }
    .high-risk { background: linear-gradient(135deg, #EF5350, #D32F2F); }
    .moderate-risk { background: linear-gradient(135deg, #FFEE58, #FBC02D); }
    .low-risk { background: linear-gradient(135deg, #66BB6A, #388E3C); }
</style>
""", unsafe_allow_html=True)

# --- SIDEBAR FOR INPUTS ---
with st.sidebar:
    st.title("Patient Details")
    st.markdown("Enter patient information to generate a prediction.")

    with st.expander("**Biometrics**", expanded=True):
        age = st.number_input("Age", 20, 90, 50, 1)
        height = st.number_input("Height (cm)", 140.0, 200.0, 165.0, 0.5, format="%.1f")
        weight = st.number_input("Weight (kg)", 40.0, 150.0, 70.0, 0.5, format="%.1f")

    with st.expander("**Medical History**", expanded=True):
        has_diabetes = st.selectbox("Has Diabetes?", ["No", "Yes"])
        has_high_bp = st.selectbox("Has High Blood Pressure?", ["No", "Yes"])
        family_history = st.selectbox("Family History of Cancer?", ["No", "Yes"])
        age_at_menarche = st.number_input("Age at First Menstruation", 8, 20, 13, 1)

    predict_button = st.button("Generate Prediction", use_container_width=True, type="primary")

# --- MAIN DASHBOARD AREA ---
st.title("🌸 Breast Cancer Prediction Dashboard")
st.markdown("This tool provides a demonstrative prediction based on the provided patient data.")

# --- Prediction Logic and Dashboard Display ---
# (This entire section remains the same as the previous version)
if predict_button:
    st.session_state.prediction_made = True
    diabetes_val = 1 if has_diabetes == "Yes" else 0
    bp_val = 1 if has_high_bp == "Yes" else 0
    family_history_val = 1 if family_history == "Yes" else 0
    st.session_state.current_inputs = {
        'Age': age, 'Height': f"{height} cm", 'Weight': f"{weight} kg",
        'Diabetes': has_diabetes, 'High BP': has_high_bp, 'Family History': family_history
    }
    input_data = np.array([[age, height, weight, diabetes_val, bp_val, family_history_val, age_at_menarche]])
    scaled_input_data = scaler.transform(input_data)
    probability = model.predict_proba(scaled_input_data)
    cancer_chance = probability[0][1] * 100
    if cancer_chance >= 70:
        risk_level, risk_class = "High", "high-risk"
    elif cancer_chance >= 30:
        risk_level, risk_class = "Moderate", "moderate-risk"
    else:
        risk_level, risk_class = "Low", "low-risk"
    st.session_state.current_prediction = {'chance': cancer_chance, 'level': risk_level, 'class': risk_class}
    new_entry = {'Age': age, 'Height': height, 'Weight': weight, 'Risk': risk_level, 'Chance (%)': cancer_chance}
    st.session_state.history.insert(0, new_entry)

if st.session_state.prediction_made:
    st.markdown("### Current Prediction Overview")
    pred_col, details_col = st.columns([1, 2])
    with pred_col:
        pred = st.session_state.current_prediction
        st.markdown(f"""<div class="prediction-card {pred['class']}"><p>{pred['level']} Risk</p><h2>{pred['chance']:.1f}%</h2><p>Chance of Cancer</p></div>""", unsafe_allow_html=True)
    with details_col:
        inputs = st.session_state.current_inputs
        cols = st.columns(3)
        cols[0].markdown(f'<div class="metric-card"><h3>Age</h3><p>{inputs["Age"]}</p></div>', unsafe_allow_html=True)
        cols[0].markdown(f'<div class="metric-card" style="margin-top: 15px;"><h3>Diabetes</h3><p>{inputs["Diabetes"]}</p></div>', unsafe_allow_html=True)
        cols[1].markdown(f'<div class="metric-card"><h3>Height</h3><p>{inputs["Height"]}</p></div>', unsafe_allow_html=True)
        cols[1].markdown(f'<div class="metric-card" style="margin-top: 15px;"><h3>High BP</h3><p>{inputs["High BP"]}</p></div>', unsafe_allow_html=True)
        cols[2].markdown(f'<div class="metric-card"><h3>Weight</h3><p>{inputs["Weight"]}</p></div>', unsafe_allow_html=True)
        cols[2].markdown(f'<div class="metric-card" style="margin-top: 15px;"><h3>Family History</h3><p>{inputs["Family History"]}</p></div>', unsafe_allow_html=True)
else:
    st.info("⬅️ Enter patient details in the sidebar and click 'Generate Prediction' to view the dashboard.")

st.markdown("---")
st.markdown("### Prediction History")
if not st.session_state.history:
    st.write("Prediction history will be displayed here.")
else:
    history_df = pd.DataFrame(st.session_state.history).head(5)
    def style_risk(row):
        risk = row['Risk']
        if risk == 'High': return ['background-color: #FFEBEE'] * len(row)
        elif risk == 'Moderate': return ['background-color: #FFFDE7'] * len(row)
        else: return ['background-color: #E8F5E9'] * len(row)
    styled_df = history_df.style.apply(style_risk, axis=1).format({'Chance (%)': '{:.2f}%', 'Height':'{:.1f}', 'Weight':'{:.1f}'})
    st.dataframe(styled_df, use_container_width=True)