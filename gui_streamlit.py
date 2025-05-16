import streamlit as st # type: ignore
import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
import time
from sklearn.preprocessing import StandardScaler


BASE_DIR   = r'C:\Users\majum\OneDrive\Pictures\Realtime_IDS'
CSV_PATH   = f"{BASE_DIR}/NSL-KDD/NSL-KDD-phase1-processed.csv"
MODEL_DIR  = f"{BASE_DIR}/trained_models"
AE_DIR     = f"{BASE_DIR}/AE_SVM/trained_models"

st.set_page_config(page_title="Live IDS | Realtime Packet Monitor", layout="wide")


st.markdown("""
    <style>
        .big-title { font-size:48px; font-weight:700; text-align:center; color:#f0f0f0; }
        .packet-card {
            background: #1e1e1e;
            padding: 2em;
            border-radius: 16px;
            margin-bottom: 1em;
            box-shadow: 0px 0px 10px rgba(0,0,0,0.6);
        }
        .normal-badge {
            background-color: #28a745;
            color: white;
            padding: 0.4em 1em;
            border-radius: 20px;
            display: inline-block;
            font-weight: bold;
        }
        .intrusion-badge {
            background-color: #dc3545;
            color: white;
            padding: 0.4em 1em;
            border-radius: 20px;
            display: inline-block;
            font-weight: bold;
        }
        .metric-box {
            border-radius: 12px;
            background-color: #2c2c2c;
            padding: 1em;
            margin: 0.5em 0;
            text-align: center;
            color: white;
        }
        .stApp {
            background-color: #121212;
        }
    </style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    return pd.read_csv(CSV_PATH)

df = load_data()

models = {
    name: joblib.load(f"{MODEL_DIR}/{name}.pkl")
    for name in ["RandomForest", "XGBoost", "LightGBM", "CatBoost", "MLP"]
}
encoder = tf.keras.models.load_model(f"{AE_DIR}/encoder_model.keras")
svm     = joblib.load(f"{AE_DIR}/svm_on_encoded.pkl")
scaler  = joblib.load(f"{MODEL_DIR}/scaler.pkl")


st.sidebar.title("⚙️ Settings")
model_name = st.sidebar.selectbox("Select Detection Model", list(models.keys()) + ["AE+SVM"])
delay = st.sidebar.slider("Packet Arrival Speed (sec)", 1, 5, 2)


st.markdown("<div class='big-title'>🔐 Real-Time Intrusion Detection System</div>", unsafe_allow_html=True)
st.markdown("---")


placeholder = st.empty()
for i in range(1000):
    with placeholder.container():
        row = df.sample(1, random_state=np.random.randint(999999))
        features = row.drop(columns="label")
        X = scaler.transform(features.values)

        if model_name != "AE+SVM":
            model = models[model_name]
            proba = model.predict_proba(X)[0]
            pred  = model.predict(X)[0]
        else:
            encoded = encoder.predict(X)
            proba = svm.predict_proba(encoded)[0]
            pred  = svm.predict(encoded)[0]

        label = "NORMAL" if pred == 0 else "INTRUSION"
        badge_class = "normal-badge" if pred == 0 else "intrusion-badge"
        confidence = f"{proba[pred]*100:.2f}%"

        st.markdown(f"<div class='packet-card'>", unsafe_allow_html=True)
        st.markdown(f"<h2>📦 Incoming Packet #{i+1}</h2>", unsafe_allow_html=True)
        st.markdown(f"<div class='{badge_class}'>{label}</div>", unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        col1.markdown(f"<div class='metric-box'><h4>Model</h4><p>{model_name}</p></div>", unsafe_allow_html=True)
        col2.markdown(f"<div class='metric-box'><h4>Confidence</h4><p>{confidence}</p></div>", unsafe_allow_html=True)
        with st.expander("📋 View Raw Features"):
            st.dataframe(features.T, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    time.sleep(delay)
