import streamlit as st
import pandas as pd
import numpy as np
import math
import plotly.graph_objects as go
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# ──────────────────────────────────────────────
# Page Configuration
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="Insurance Purchase Predictor",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────
# Custom CSS for premium look
# ──────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

    /* Global font */
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* Main background */
    .stApp {
        background: linear-gradient(135deg, #0f0c29 0%, #1a1a3e 40%, #24243e 100%);
    }

    /* Header styling */
    .main-header {
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.15), rgba(118, 75, 162, 0.15));
        border: 1px solid rgba(102, 126, 234, 0.3);
        border-radius: 20px;
        padding: 2.5rem 2rem;
        margin-bottom: 2rem;
        text-align: center;
        backdrop-filter: blur(10px);
    }
    .main-header h1 {
        background: linear-gradient(135deg, #667eea, #764ba2, #f093fb);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.4rem;
        font-weight: 800;
        margin-bottom: 0.5rem;
    }
    .main-header p {
        color: rgba(255, 255, 255, 0.7);
        font-size: 1.05rem;
        font-weight: 300;
    }

    /* Metric cards */
    .metric-card {
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.1), rgba(118, 75, 162, 0.1));
        border: 1px solid rgba(102, 126, 234, 0.25);
        border-radius: 16px;
        padding: 1.5rem;
        text-align: center;
        backdrop-filter: blur(10px);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    .metric-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 12px 40px rgba(102, 126, 234, 0.2);
    }
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        background: linear-gradient(135deg, #667eea, #764ba2);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .metric-label {
        color: rgba(255, 255, 255, 0.6);
        font-size: 0.85rem;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-top: 0.5rem;
    }

    /* Prediction result cards */
    .prediction-card {
        border-radius: 20px;
        padding: 2rem;
        text-align: center;
        backdrop-filter: blur(10px);
        margin-top: 1rem;
    }
    .prediction-buy {
        background: linear-gradient(135deg, rgba(46, 213, 115, 0.15), rgba(39, 174, 96, 0.15));
        border: 1px solid rgba(46, 213, 115, 0.4);
    }
    .prediction-nobuy {
        background: linear-gradient(135deg, rgba(252, 92, 101, 0.15), rgba(214, 48, 49, 0.15));
        border: 1px solid rgba(252, 92, 101, 0.4);
    }
    .prediction-emoji {
        font-size: 3.5rem;
        margin-bottom: 0.5rem;
    }
    .prediction-text {
        font-size: 1.3rem;
        font-weight: 700;
        margin-bottom: 0.3rem;
    }
    .prediction-buy .prediction-text { color: #2ed573; }
    .prediction-nobuy .prediction-text { color: #fc5c65; }
    .prediction-prob {
        color: rgba(255, 255, 255, 0.6);
        font-size: 0.95rem;
    }

    /* Section headers */
    .section-header {
        color: rgba(255, 255, 255, 0.9);
        font-size: 1.4rem;
        font-weight: 700;
        margin: 2rem 0 1rem 0;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid rgba(102, 126, 234, 0.3);
    }

    /* Info box */
    .info-box {
        background: linear-gradient(135deg, rgba(52, 152, 219, 0.1), rgba(155, 89, 182, 0.1));
        border: 1px solid rgba(52, 152, 219, 0.25);
        border-radius: 12px;
        padding: 1.2rem;
        color: rgba(255, 255, 255, 0.8);
        font-size: 0.9rem;
        line-height: 1.6;
    }

    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a1a3e 0%, #0f0c29 100%);
        border-right: 1px solid rgba(102, 126, 234, 0.2);
    }
    section[data-testid="stSidebar"] .stMarkdown h1,
    section[data-testid="stSidebar"] .stMarkdown h2,
    section[data-testid="stSidebar"] .stMarkdown h3 {
        color: rgba(255, 255, 255, 0.9);
    }

    /* Dataframe styling */
    .stDataFrame {
        border-radius: 12px;
        overflow: hidden;
    }

    /* Slider */
    .stSlider > div > div > div > div {
        background: linear-gradient(135deg, #667eea, #764ba2);
    }

    /* Expander */
    .streamlit-expanderHeader {
        background: rgba(102, 126, 234, 0.1);
        border-radius: 10px;
    }

    /* Hide default streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────
# Sigmoid helper
# ──────────────────────────────────────────────
def sigmoid(x):
    return 1 / (1 + math.exp(-x))


# ──────────────────────────────────────────────
# Load & cache data
# ──────────────────────────────────────────────
@st.cache_data
def load_data():
    try:
        df = pd.read_csv("insurance_data.csv")
    except FileNotFoundError:
        # Fallback embedded data
        data = {
            "age": [22, 25, 47, 52, 46, 56, 55, 60, 62, 61,
                    18, 28, 27, 29, 49, 55, 25, 58, 19, 18,
                    21, 26, 40, 45, 50, 54, 23],
            "bought_insurance": [0, 0, 1, 0, 1, 1, 0, 1, 1, 1,
                                 0, 0, 0, 0, 1, 1, 1, 1, 0, 0,
                                 0, 0, 1, 1, 1, 1, 0],
        }
        df = pd.DataFrame(data)
    return df


@st.cache_resource
def train_model(train_size, random_state):
    df = load_data()
    X = df[["age"]]
    y = df["bought_insurance"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=train_size, random_state=random_state
    )
    model = LogisticRegression(solver="lbfgs", max_iter=200)
    model.fit(X_train, y_train)
    score = model.score(X_test, y_test)
    y_pred = model.predict(X_test)
    return model, X_train, X_test, y_train, y_test, y_pred, score


# ──────────────────────────────────────────────
# Sidebar Controls
# ──────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Model Settings")
    st.markdown("---")

    train_size = st.slider(
        "Training Size",
        min_value=0.5,
        max_value=0.9,
        value=0.8,
        step=0.05,
        help="Fraction of data used for training",
    )

    random_state = st.number_input(
        "Random State",
        min_value=0,
        max_value=100,
        value=42,
        step=1,
        help="Controls the shuffling of data before splitting",
    )

    st.markdown("---")
    st.markdown("## 🔮 Predict Insurance Purchase")

    input_age = st.slider(
        "Enter Age",
        min_value=10,
        max_value=80,
        value=35,
        step=1,
        help="Select the age to predict insurance purchase",
    )

    predict_btn = st.button("🚀  Predict", use_container_width=True)

    st.markdown("---")
    st.markdown(
        """
        <div style="text-align:center; color: rgba(255,255,255,0.4); font-size: 0.75rem;">
            Built with ❤️ using Streamlit<br>
            Logistic Regression ML App
        </div>
        """,
        unsafe_allow_html=True,
    )


# ──────────────────────────────────────────────
# Load data & train model
# ──────────────────────────────────────────────
df = load_data()
model, X_train, X_test, y_train, y_test, y_pred, score = train_model(
    train_size, random_state
)

# ──────────────────────────────────────────────
# Main Header
# ──────────────────────────────────────────────
st.markdown(
    """
    <div class="main-header">
        <h1>🛡️ Insurance Purchase Predictor</h1>
        <p>Predicting whether a person would buy life insurance based on their age using <strong>Logistic Regression</strong></p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ──────────────────────────────────────────────
# Top Metric Cards
# ──────────────────────────────────────────────
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-value">{len(df)}</div>
            <div class="metric-label">Total Samples</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col2:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-value">{len(X_train)}</div>
            <div class="metric-label">Training Samples</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col3:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-value">{len(X_test)}</div>
            <div class="metric-label">Test Samples</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col4:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-value">{score * 100:.1f}%</div>
            <div class="metric-label">Model Accuracy</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ──────────────────────────────────────────────
# Prediction Section
# ──────────────────────────────────────────────
if predict_btn:
    prediction = model.predict([[input_age]])[0]
    proba = model.predict_proba([[input_age]])[0]

    st.markdown('<div class="section-header">🎯 Prediction Result</div>', unsafe_allow_html=True)

    res_col1, res_col2, res_col3 = st.columns([1, 2, 1])
    with res_col2:
        if prediction == 1:
            st.markdown(
                f"""
                <div class="prediction-card prediction-buy">
                    <div class="prediction-emoji">✅</div>
                    <div class="prediction-text">Will Buy Insurance</div>
                    <div class="prediction-prob">
                        A person aged <strong>{input_age}</strong> is <strong>{proba[1]*100:.1f}%</strong> likely to buy insurance
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f"""
                <div class="prediction-card prediction-nobuy">
                    <div class="prediction-emoji">❌</div>
                    <div class="prediction-text">Will Not Buy Insurance</div>
                    <div class="prediction-prob">
                        A person aged <strong>{input_age}</strong> has only <strong>{proba[1]*100:.1f}%</strong> chance of buying insurance
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    # Sigmoid manual calculation
    m = model.coef_[0][0]
    b = model.intercept_[0]
    z = m * input_age + b
    y_manual = sigmoid(z)

    with st.expander("📐 Manual Sigmoid Calculation", expanded=False):
        st.markdown(
            f"""
            <div class="info-box">
                <strong>Logistic Regression Equation:</strong>  y = 1 / (1 + e<sup>-(mx + b)</sup>)<br><br>
                <strong>Coefficient (m):</strong> {m:.6f}<br>
                <strong>Intercept (b):</strong> {b:.6f}<br>
                <strong>z = m × age + b:</strong> {m:.6f} × {input_age} + ({b:.6f}) = {z:.6f}<br>
                <strong>σ(z) = 1 / (1 + e<sup>-z</sup>):</strong> {y_manual:.6f}<br><br>
                Since {y_manual:.4f} {"≥" if y_manual >= 0.5 else "<"} 0.5 → prediction: <strong>{"Will Buy ✅" if y_manual >= 0.5 else "Will Not Buy ❌"}</strong>
            </div>
            """,
            unsafe_allow_html=True,
        )


# ──────────────────────────────────────────────
# Charts Section
# ──────────────────────────────────────────────
st.markdown('<div class="section-header">📊 Data Visualization</div>', unsafe_allow_html=True)

chart_col1, chart_col2 = st.columns(2)

# ---- Scatter Plot ----
with chart_col1:
    fig_scatter = go.Figure()

    # Data points
    bought = df[df["bought_insurance"] == 1]
    not_bought = df[df["bought_insurance"] == 0]

    fig_scatter.add_trace(
        go.Scatter(
            x=not_bought["age"],
            y=not_bought["bought_insurance"],
            mode="markers",
            name="Did Not Buy",
            marker=dict(
                size=12,
                color="#fc5c65",
                symbol="x",
                line=dict(width=2, color="#fc5c65"),
            ),
        )
    )
    fig_scatter.add_trace(
        go.Scatter(
            x=bought["age"],
            y=bought["bought_insurance"],
            mode="markers",
            name="Bought",
            marker=dict(
                size=12,
                color="#2ed573",
                symbol="circle",
                line=dict(width=2, color="#2ed573"),
            ),
        )
    )

    fig_scatter.update_layout(
        title=dict(text="Age vs Insurance Purchase", font=dict(size=18, color="white")),
        xaxis=dict(
            title="Age",
            gridcolor="rgba(255,255,255,0.06)",
            color="rgba(255,255,255,0.7)",
        ),
        yaxis=dict(
            title="Bought Insurance (0 / 1)",
            gridcolor="rgba(255,255,255,0.06)",
            color="rgba(255,255,255,0.7)",
            tickvals=[0, 1],
        ),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        legend=dict(font=dict(color="rgba(255,255,255,0.7)")),
        height=420,
    )
    st.plotly_chart(fig_scatter, use_container_width=True)

# ---- Sigmoid Curve ----
with chart_col2:
    m = model.coef_[0][0]
    b = model.intercept_[0]
    ages_range = np.linspace(10, 80, 300)
    probs = [sigmoid(m * a + b) for a in ages_range]

    fig_sigmoid = go.Figure()

    fig_sigmoid.add_trace(
        go.Scatter(
            x=ages_range,
            y=probs,
            mode="lines",
            name="Sigmoid Curve",
            line=dict(
                color="#667eea",
                width=3,
                shape="spline",
            ),
            fill="tozeroy",
            fillcolor="rgba(102, 126, 234, 0.1)",
        )
    )

    # Decision boundary line
    fig_sigmoid.add_hline(
        y=0.5,
        line_dash="dash",
        line_color="rgba(255, 255, 255, 0.35)",
        annotation_text="Decision Boundary (0.5)",
        annotation_font_color="rgba(255,255,255,0.5)",
    )

    # Actual data points on the curve
    actual_probs = [sigmoid(m * a + b) for a in df["age"]]
    colors = ["#2ed573" if v == 1 else "#fc5c65" for v in df["bought_insurance"]]
    fig_sigmoid.add_trace(
        go.Scatter(
            x=df["age"],
            y=actual_probs,
            mode="markers",
            name="Data Points",
            marker=dict(size=9, color=colors, line=dict(width=1, color="white")),
        )
    )

    fig_sigmoid.update_layout(
        title=dict(text="Logistic Regression Sigmoid Curve", font=dict(size=18, color="white")),
        xaxis=dict(
            title="Age",
            gridcolor="rgba(255,255,255,0.06)",
            color="rgba(255,255,255,0.7)",
        ),
        yaxis=dict(
            title="Probability of Buying",
            gridcolor="rgba(255,255,255,0.06)",
            color="rgba(255,255,255,0.7)",
            range=[-0.05, 1.05],
        ),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        legend=dict(font=dict(color="rgba(255,255,255,0.7)")),
        height=420,
    )
    st.plotly_chart(fig_sigmoid, use_container_width=True)


# ──────────────────────────────────────────────
# Probability Distribution Bar Chart
# ──────────────────────────────────────────────
st.markdown('<div class="section-header">📈 Prediction Probability Distribution</div>', unsafe_allow_html=True)

test_ages = X_test["age"].values
test_proba = model.predict_proba(X_test)
test_preds = model.predict(X_test)

bar_colors = ["#2ed573" if p == 1 else "#fc5c65" for p in test_preds]

fig_bar = go.Figure()
fig_bar.add_trace(
    go.Bar(
        x=[f"Age {a}" for a in test_ages],
        y=test_proba[:, 1],
        marker_color=bar_colors,
        marker_line=dict(width=1.5, color="rgba(255,255,255,0.2)"),
        text=[f"{p:.1%}" for p in test_proba[:, 1]],
        textposition="outside",
        textfont=dict(color="rgba(255,255,255,0.8)", size=12),
        name="Buy Probability",
    )
)

fig_bar.add_hline(
    y=0.5,
    line_dash="dash",
    line_color="rgba(255, 255, 255, 0.35)",
    annotation_text="Threshold (0.5)",
    annotation_font_color="rgba(255,255,255,0.5)",
)

fig_bar.update_layout(
    title=dict(text="Test Set — Probability of Buying Insurance", font=dict(size=18, color="white")),
    xaxis=dict(
        title="Test Samples",
        gridcolor="rgba(255,255,255,0.06)",
        color="rgba(255,255,255,0.7)",
    ),
    yaxis=dict(
        title="Probability",
        gridcolor="rgba(255,255,255,0.06)",
        color="rgba(255,255,255,0.7)",
        range=[0, 1.15],
    ),
    plot_bgcolor="rgba(0,0,0,0)",
    paper_bgcolor="rgba(0,0,0,0)",
    height=400,
)
st.plotly_chart(fig_bar, use_container_width=True)


# ──────────────────────────────────────────────
# Data & Model Details
# ──────────────────────────────────────────────
st.markdown('<div class="section-header">📋 Data & Model Details</div>', unsafe_allow_html=True)

detail_col1, detail_col2 = st.columns(2)

with detail_col1:
    with st.expander("📄 Full Dataset", expanded=False):
        styled_df = df.copy()
        styled_df["bought_insurance"] = styled_df["bought_insurance"].map(
            {1: "✅ Yes", 0: "❌ No"}
        )
        st.dataframe(styled_df, use_container_width=True, height=350)

with detail_col2:
    with st.expander("🧪 Test Set — Actual vs Predicted", expanded=False):
        comparison = pd.DataFrame(
            {
                "Age": X_test["age"].values,
                "Actual": y_test.values,
                "Predicted": y_pred,
                "Match": ["✅" if a == p else "❌" for a, p in zip(y_test.values, y_pred)],
            }
        )
        st.dataframe(comparison, use_container_width=True, height=350)

with st.expander("🔬 Model Coefficients", expanded=False):
    coef_col1, coef_col2, coef_col3 = st.columns(3)
    with coef_col1:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-value" style="font-size:1.4rem;">{model.coef_[0][0]:.6f}</div>
                <div class="metric-label">Coefficient (m)</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with coef_col2:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-value" style="font-size:1.4rem;">{model.intercept_[0]:.6f}</div>
                <div class="metric-label">Intercept (b)</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with coef_col3:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-value" style="font-size:1.4rem;">y = σ(mx + b)</div>
                <div class="metric-label">Equation</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown(
        """
        <div class="info-box" style="margin-top:1rem;">
            <strong>How it works:</strong> Logistic Regression uses the sigmoid function
            <em>σ(z) = 1 / (1 + e<sup>-z</sup>)</em> where <em>z = m × age + b</em>.
            If the output ≥ 0.5, the model predicts the person <strong>will buy</strong> insurance;
            otherwise <strong>will not</strong>.
        </div>
        """,
        unsafe_allow_html=True,
    )
