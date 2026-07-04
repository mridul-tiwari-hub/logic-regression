import streamlit as st
import pandas as pd
from sklearn.linear_model import LogisticRegression

# ──────────────────────────────────────────────
# Page Configuration
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="Life Insurance Purchase Prediction",
    page_icon="🛡️",
    layout="centered",
)

# ──────────────────────────────────────────────
# Title & Description
# ──────────────────────────────────────────────
st.title("🛡️ Life Insurance Purchase Prediction")
st.write("Predict whether a person will buy life insurance based on their age using Logistic Regression.")

st.write("")

# ──────────────────────────────────────────────
# Load Data & Train Model
# ──────────────────────────────────────────────
# Creating or reading the dataset
data = {
    "age": [22, 25, 47, 52, 46, 56, 55, 60, 62, 61,
            18, 28, 27, 29, 49, 55, 25, 58, 19, 18,
            21, 26, 40, 45, 50, 54, 23],
    "bought_insurance": [0, 0, 1, 0, 1, 1, 0, 1, 1, 1,
                         0, 0, 0, 0, 1, 1, 1, 1, 0, 0,
                         0, 0, 1, 1, 1, 1, 0],
}
df = pd.DataFrame(data)

# Save/ensure insurance_data.csv exists
df.to_csv("insurance_data.csv", index=False)

# Train the model on the dataset
X = df[["age"]]
y = df["bought_insurance"]
model = LogisticRegression(solver="lbfgs")
model.fit(X, y)

# ──────────────────────────────────────────────
# Dataset Overview
# ──────────────────────────────────────────────
st.header("Dataset Overview")
st.dataframe(df.head(5), use_container_width=True)

st.write("")

# ──────────────────────────────────────────────
# Enter Customer Details
# ──────────────────────────────────────────────
st.header("Enter Customer Details")

# Input for Age
age_input = st.number_input("Age", min_value=1, max_value=120, value=30, step=1)

# Prediction execution
if st.button("Predict"):
    prediction = model.predict([[age_input]])[0]
    proba = model.predict_proba([[age_input]])[0]
    
    st.write("")
    if prediction == 1:
        st.success(f"Prediction: **Will Buy Insurance** (Probability: {proba[1]*100:.2f}%)")
    else:
        st.error(f"Prediction: **Will Not Buy Insurance** (Probability: {proba[0]*100:.2f}%)")
