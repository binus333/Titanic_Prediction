import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

st.set_page_config(page_title="Titanic Survival Predictor", page_icon="🚢", layout="centered")

# ---------- Load model artifacts ----------
MODEL_PATH = "model.pkl"

if not os.path.exists(MODEL_PATH):
    st.error(
        "Model files not found. Please run all cells in "
        "`titanic_random_forest.ipynb` first — it saves model.pkl, "
        "encoders.pkl, and features.pkl into this folder."
    )
    st.stop()

model = joblib.load("model.pkl")
encoders = joblib.load("encoders.pkl")
features = joblib.load("features.pkl")

# ---------- Header ----------
st.title("🚢 Titanic Survival Predictor")
st.write(
    "A Random Forest classifier trained on the classic Titanic dataset. "
    "Enter passenger details below to predict survival probability."
)

st.divider()

# ---------- Sidebar: model info ----------
with st.sidebar:
    st.header("About this model")
    st.write("**Algorithm:** Random Forest Classifier")
    st.write("**Trees:** 300")
    st.write("**Max depth:** 6")
    st.write(f"**Features used:** {', '.join(features)}")

# ---------- Input form ----------
col1, col2 = st.columns(2)

with col1:
    pclass = st.selectbox("Passenger Class", options=[1, 2, 3], index=2,
                           format_func=lambda x: f"Class {x}")
    sex = st.selectbox("Sex", options=["male", "female"])
    age = st.slider("Age", min_value=0, max_value=90, value=30)
    sibsp = st.number_input("Siblings/Spouses aboard", min_value=0, max_value=10, value=0)

with col2:
    parch = st.number_input("Parents/Children aboard", min_value=0, max_value=10, value=0)
    fare = st.number_input("Fare paid ($)", min_value=0.0, max_value=600.0, value=32.0, step=1.0)
    embarked = st.selectbox(
        "Port of Embarkation", options=["S", "C", "Q"],
        format_func=lambda x: {"S": "Southampton", "C": "Cherbourg", "Q": "Queenstown"}[x]
    )

st.divider()

if st.button("Predict Survival", type="primary", use_container_width=True):
    # Build input row in the same order/encoding as training
    row = {
        "pclass": pclass,
        "sex": encoders["sex"].transform([sex])[0],
        "age": age,
        "sibsp": sibsp,
        "parch": parch,
        "fare": fare,
        "embarked": encoders["embarked"].transform([embarked])[0],
    }
    X_input = pd.DataFrame([row])[features]

    pred = model.predict(X_input)[0]
    proba = model.predict_proba(X_input)[0]

    survived_prob = proba[1] * 100
    died_prob = proba[0] * 100

    if pred == 1:
        st.success(f" Predicted: **Survived** ({survived_prob:.1f}% confidence)")
    else:
        st.error(f" Predicted: **Did not survive** ({died_prob:.1f}% confidence)")

    st.progress(int(survived_prob), text=f"Survival probability: {survived_prob:.1f}%")

    st.divider()

    # ---------- Feature importance ----------
    st.subheader("What drives this model's predictions?")
    importances = pd.Series(model.feature_importances_, index=features).sort_values(ascending=False)
    st.bar_chart(importances)

    with st.expander("See the exact input sent to the model"):
        st.dataframe(X_input)

st.divider()
st.caption("Made with lots of ❤️")
