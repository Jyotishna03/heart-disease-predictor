import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Streamlit configuration
st.set_page_config(page_title="Heart Disease Prediction", layout="centered")

# ---------------- HTML & CSS Styling ---------------- #
st.markdown("""
    <style>
        body {
            background-color: #f7f9fc;
        }
        .header {
            background: linear-gradient(to right, #c31432, #240b36);
            padding: 1.5rem;
            border-radius: 10px;
            text-align: center;
            color: white;
            margin-bottom: 20px;
        }
        .card {
            background-color: white;
            padding: 1.5rem;
            border-radius: 12px;
            box-shadow: 0 4px 10px rgba(0, 0, 0, 0.08);
            margin-top: 1rem;
        }
        .subheader {
            font-size: 1.3rem;
            margin-bottom: 0.5rem;
        }
        .download {
            margin-top: 20px;
        }
    </style>

    <div class="header">
        <h1>❤️ Heart Disease Prediction App</h1>
        <p>Predict the likelihood of heart disease based on health data.</p>
    </div>
""", unsafe_allow_html=True)

# ---------------- Load Dataset ---------------- #
@st.cache_data
def load_data():
    df = pd.read_csv("Heart_Disease_Full.csv")
    return df

df = load_data()
X = df.drop("target", axis=1)
y = df["target"]

# Train/test split and model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
accuracy = accuracy_score(y_test, model.predict(X_test))

# ---------------- Sidebar Input Form ---------------- #
st.sidebar.header("📝 Enter Patient Details")

def user_input_features():
    age = st.sidebar.slider("Age", 29, 77, 54)
    sex = st.sidebar.selectbox("Sex (1 = Male, 0 = Female)", [1, 0])
    cp = st.sidebar.slider("Chest Pain Type (0-3)", 0, 3, 1)
    trestbps = st.sidebar.slider("Resting Blood Pressure", 80, 200, 120)
    chol = st.sidebar.slider("Cholesterol", 100, 600, 240)
    fbs = st.sidebar.selectbox("Fasting Blood Sugar > 120 mg/dl", [1, 0])
    restecg = st.sidebar.slider("Resting ECG Results", 0, 2, 1)
    thalach = st.sidebar.slider("Max Heart Rate Achieved", 60, 220, 150)
    exang = st.sidebar.selectbox("Exercise Induced Angina", [1, 0])
    oldpeak = st.sidebar.slider("ST Depression", 0.0, 6.2, 1.0)
    slope = st.sidebar.slider("Slope of ST Segment", 0, 2, 1)
    ca = st.sidebar.slider("Number of Major Vessels (0-3)", 0, 3, 0)
    thal = st.sidebar.slider("Thalassemia (1 = normal; 2 = fixed defect; 3 = reversable defect)", 1, 3, 2)

    data = {
        "age": age, "sex": sex, "cp": cp, "trestbps": trestbps, "chol": chol,
        "fbs": fbs, "restecg": restecg, "thalach": thalach, "exang": exang,
        "oldpeak": oldpeak, "slope": slope, "ca": ca, "thal": thal
    }
    return pd.DataFrame(data, index=[0])

input_df = user_input_features()

# ---------------- Prediction ---------------- #
prediction = model.predict(input_df)[0]
prediction_proba = model.predict_proba(input_df)[0]

# ---------------- Display Results ---------------- #
st.markdown('<div class="card">', unsafe_allow_html=True)

st.markdown('<div class="subheader">📈 Prediction Result</div>', unsafe_allow_html=True)
if prediction == 1:
    st.error("⚠️ Likely to have Heart Disease 😟")
else:
    st.success("✅ Unlikely to have Heart Disease 😊")

st.markdown('<div class="subheader">📊 Prediction Probability</div>', unsafe_allow_html=True)
st.write(f"**Probability of No Disease:** {prediction_proba[0]*100:.2f}%")
st.write(f"**Probability of Disease:** {prediction_proba[1]*100:.2f}%")

st.markdown('<div class="subheader">📌 Model Accuracy</div>', unsafe_allow_html=True)
st.info(f"Model Accuracy on Test Data: **{accuracy*100:.2f}%**")

st.markdown('</div>', unsafe_allow_html=True)

# ---------------- Input Data Summary ---------------- #
st.markdown('<div class="card">', unsafe_allow_html=True)
st.markdown('<div class="subheader">🔎 View Your Input Data</div>', unsafe_allow_html=True)
st.dataframe(input_df)
st.markdown('</div>', unsafe_allow_html=True)

# ---------------- Save Prediction Logs ---------------- #
input_df["prediction"] = prediction
input_df["prob_0"] = prediction_proba[0]
input_df["prob_1"] = prediction_proba[1]
input_df.to_csv("prediction_log.csv", mode="a", header=False, index=False)

with open("prediction_log.csv", "rb") as f:
    st.download_button("📥 Download Prediction Logs", f, "prediction_log.csv", help="Click to download your predictions", type="primary")


