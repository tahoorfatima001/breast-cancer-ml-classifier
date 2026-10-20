"""Streamlit demo app for the Breast Cancer Diagnosis Classifier.
Educational project only - NOT a medical tool."""
import pandas as pd
import streamlit as st
from predict import load_artifacts, predict_one

st.set_page_config(page_title="Breast Cancer ML Classifier", page_icon="🩺", layout="centered")
st.title("🩺 Breast Cancer Diagnosis Classifier")
st.warning("Educational machine learning project. This is **not** a medical tool and must not be used for diagnosis.")

model, meta = load_artifacts()
st.write(f"**Model:** {meta['best_model']}  |  **Test accuracy:** {meta['test_accuracy']:.1%}")
st.write("Adjust the 10 most important tumour measurements. All other features are set to the dataset average.")

values = {}
for feat in meta["top_features"]:
    lo, hi = meta["feature_min"][feat], meta["feature_max"][feat]
    values[feat] = st.slider(feat, float(lo), float(hi), float(meta["feature_means"][feat]))

if st.button("Predict", type="primary"):
    label, probs = predict_one(model, meta, values)
    if label == "malignant":
        st.error(f"Model prediction: **{label.title()}**")
    else:
        st.success(f"Model prediction: **{label.title()}**")
    st.bar_chart(pd.DataFrame({"Probability": probs}))

with st.expander("Model comparison results"):
    st.dataframe(pd.read_csv("results.csv"))
st.image("images/feature_importance.png", caption="Most important features")
