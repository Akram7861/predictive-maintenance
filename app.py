from pathlib import Path
import pickle
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "best_model.pkl"
TRAIN_PATH = ROOT / "artifacts" / "train.csv"

st.set_page_config(
    page_title="Predictive Maintenance",
    page_icon="⚙️",
    layout="wide"
)

st.title("Engine Predictive Maintenance")
st.write(
    "Enter engine sensor measurements to obtain a prediction "
    "from the trained machine learning model."
)

if not MODEL_PATH.exists():
    st.error("Model file not found. Run the ML training pipeline first.")
    st.stop()

if not TRAIN_PATH.exists():
    st.error("Feature schema not found. Please provide artifacts/train.csv.")
    st.stop()

train = pd.read_csv(TRAIN_PATH)
target = "Engine Condition"

if target not in train.columns:
    st.error(f"Expected target column '{target}' was not found.")
    st.stop()

feature_names = [c for c in train.columns if c != target]

with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

st.subheader("Engine sensor inputs")
st.caption("Enter values in the same units and ranges as the training data.")

defaults = train[feature_names].median(numeric_only=True)
inputs = {}

with st.form("prediction_form"):
    columns = st.columns(2)

    for i, feature in enumerate(feature_names):
        with columns[i % 2]:
            inputs[feature] = st.number_input(
                feature,
                value=float(defaults[feature]),
                format="%.4f",
                key=f"input_{i}"
            )

    submitted = st.form_submit_button("Predict engine condition")

if submitted:
    input_df = pd.DataFrame([inputs], columns=feature_names)

    try:
        prediction = model.predict(input_df)[0]
        st.subheader("Prediction result")
        st.metric("Predicted class", str(prediction))

        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(input_df)[0]
            probability_df = pd.DataFrame({
                "Class": [str(c) for c in model.classes_],
                "Probability": probabilities
            })
            st.write("Model class probabilities")
            st.dataframe(
                probability_df.style.format({"Probability": "{:.2%}"}),
                use_container_width=True
            )

        st.info(
            "Class meanings must be confirmed from the dataset documentation. "
            "This model output is a prediction, not a confirmed diagnosis "
            "of engine health or a guarantee of failure."
        )
        st.write("Input submitted to the model")
        st.dataframe(input_df, use_container_width=True)

    except Exception as exc:
        st.error(f"Prediction failed: {exc}")
