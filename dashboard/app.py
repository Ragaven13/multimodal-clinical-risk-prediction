from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Clinical Risk Dashboard", layout="wide")

st.title("Multimodal Clinical Risk Prediction Dashboard")
st.subheader("ICU Risk Prediction System")

st.write("Enter patient details below to estimate ICU risk using the trained XGBoost model.")

MODEL_PATH = Path("artifacts/xgboost_pipeline.joblib")
THRESHOLD = 0.35


@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        st.error(f"Model file not found: {MODEL_PATH}")
        st.stop()
    return joblib.load(MODEL_PATH)


model = load_model()

st.sidebar.header("Patient Input")

age = st.sidebar.slider("Age", min_value=18, max_value=100, value=50)
gender = st.sidebar.selectbox("Gender", ["F", "M"])
race = st.sidebar.selectbox(
    "Race",
    [
        "WHITE",
        "BLACK/AFRICAN AMERICAN",
        "ASIAN",
        "HISPANIC OR LATINO",
        "UNKNOWN",
        "OTHER",
    ],
)

input_df = pd.DataFrame(
    [
        {
            "anchor_age": age,
            "gender": gender,
            "race": race,
        }
    ]
)

st.markdown("## Patient Summary")
st.write(input_df)

if st.button("Predict ICU Risk"):
    risk_prob = model.predict_proba(input_df)[0, 1]
    prediction = "High ICU Risk" if risk_prob >= THRESHOLD else "Lower ICU Risk"

    st.markdown("## Prediction")
    st.metric("ICU Risk Probability", f"{risk_prob:.3f}")
    st.write(f"**Threshold used:** {THRESHOLD}")
    st.write(f"**Prediction:** {prediction}")

    if risk_prob >= THRESHOLD:
        st.warning("Patient flagged as high ICU risk.")
    else:
        st.success("Patient flagged as lower ICU risk.")