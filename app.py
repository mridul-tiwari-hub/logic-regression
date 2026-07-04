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
st.markdown("Predict whether a person will buy life insurance based on their age using Logistic Regression.")

# ──────────────────────────────────────────────
# Load Data & Train Model
# ──────────────────────────────────────────────
# Creating the dataset
data = {
    "age": [22, 25, 47, 52, 46, 56, 55, 60, 62, 61,
            18, 28, 27, 29, 49, 55, 25, 58, 19, 18,
            21, 26, 40, 45, 50, 54, 23],
    "bought_insurance": [0, 0, 1, 0, 1, 1, 0, 1, 1, 1,
                         0, 0, 0, 0, 1, 1, 1, 1, 0, 0,
                         0, 0, 1, 1, 1, 1, 0],
}
df = pd.DataFrame(data)

# Train the model on the full dataset
X = df[["age"]]
y = df["bought_insurance"]
model = LogisticRegression(solver="lbfgs")
model.fit(X, y)

# Retrieve coefficients
m_coeff = model.coef_[0][0]
b_intercept = model.intercept_[0]

# ──────────────────────────────────────────────
# Dataset Overview
# ──────────────────────────────────────────────
st.header("Dataset Overview")
st.dataframe(df.head(5), use_container_width=True)

# ──────────────────────────────────────────────
# Enter Customer Details
# ──────────────────────────────────────────────
st.header("Enter Customer Details")

# Input for Age
age_input = st.number_input("Age of the Person", min_value=1, max_value=120, value=35, step=1)

# Predict button
predict_clicked = st.button("Predict Insurance Status")

# We want the prediction to run and be shown either if button is clicked, or by default (as in screenshot)
if True:
    prediction = model.predict([[age_input]])[0]
    proba = model.predict_proba([[age_input]])[0]
    
    st.write("")
    if prediction == 1:
        st.success(f"Prediction: This person WILL buy insurance.", icon="✅")
        st.info(f"Probability of buying: {proba[1]*100:.2f}%", icon="📊")
    else:
        st.warning(f"Prediction: This person WILL NOT buy insurance.", icon="⚠️")
        st.info(f"Probability of buying: {proba[1]*100:.2f}%", icon="📊")

# ──────────────────────────────────────────────
# Model Parameters
# ──────────────────────────────────────────────
st.header("Model Parameters")
st.markdown(f"**Coefficient (m):** `{m_coeff}`")
st.markdown(f"**Intercept (b):** `{b_intercept}`")
