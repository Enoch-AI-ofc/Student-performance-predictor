from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

FEATURE_ORDER = [
    "study_hours",
    "attendance",
    "previous_score",
    "assignment_score",
    "sleep_hours",
    "internet_access",
]

# Anchored to this file so the app works no matter which folder it is launched from.
MODEL_PATH = Path(__file__).resolve().parent / "Models" / "student_score_predictor.pkl"


@st.cache_resource
def load_model():
    system = joblib.load(MODEL_PATH)
    return system["polynomial_transformer"], system["ridge_prediction_engine"]


poly_engine, ridge_engine = load_model()

st.title("Student Performance Predictor")
st.caption(
    "Set a student profile below, then click **Predict final score** "
    "to run it through the saved model."
)

with st.form("student_profile"):
    left, right = st.columns(2)

    with left:
        study_hours = st.slider("Study hours per week", 0.0, 40.0, 18.0, 0.5)
        attendance = st.slider("Attendance (%)", 0, 100, 92)
        previous_score = st.slider("Previous score", 0, 100, 78)

    with right:
        assignment_score = st.slider("Assignment score", 0, 100, 85)
        sleep_hours = st.slider("Sleep hours per night", 4.0, 10.0, 7.5, 0.5)
        internet_access = st.checkbox("Internet access at home", value=True)

    submitted = st.form_submit_button("Predict final score")

if submitted:
    # Column order must match the order used when the model was trained.
    student = pd.DataFrame(
        [
            {
                "study_hours": study_hours,
                "attendance": float(attendance),
                "previous_score": float(previous_score),
                "assignment_score": float(assignment_score),
                "sleep_hours": sleep_hours,
                "internet_access": int(internet_access),
            }
        ],
        columns=FEATURE_ORDER,
    )

    student_poly = poly_engine.transform(student)
    # Training scores were capped at 0-100, but the raw model output can spill
    # past that; clamp for display so the score stays in the valid range.
    raw_score = float(ridge_engine.predict(student_poly)[0])
    predicted_score = min(max(raw_score, 0.0), 100.0)

    st.subheader("Prediction")
    st.metric("Predicted final score", f"{predicted_score:.1f} / 100")
    if raw_score != predicted_score:
        st.caption(
            f"The model's raw output was {raw_score:.1f}; the displayed score is "
            "clamped to the valid 0-100 range. (Training scores were capped at 100, "
            "so very strong profiles can drift slightly past it.)"
        )

    st.subheader("Profile used for this prediction")
    st.dataframe(student, hide_index=True)
else:
    st.info(
        "The default values are the notebook's demo student. "
        "Click **Predict final score** to reproduce that result, or adjust any values first."
    )

with st.expander("About this model"):
    st.markdown(
        "This app runs the saved model from the project notebook "
        "(polynomial features + Ridge regression).\n\n"
        "- **Accuracy** — on 200 students held out from training, the average "
        "prediction error (MAE) is 2.9 points and the model explains about 68% of "
        "the differences between scores (R² = 0.68).\n"
        "- **Why you might see 100** — the synthetic training data was generated "
        "with a hard cap at 100, so about 29% of its 1,000 students score exactly "
        "100. For very strong profiles the raw model output can land slightly past "
        "100 and is clamped for display."
    )
