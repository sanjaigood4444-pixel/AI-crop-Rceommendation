import streamlit as st
import numpy as np
import os
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# ========================
# Crop Images
# ========================
def get_image_path(crop):
    for ext in ['.jpg', '.jpeg', '.jfif', '.png']:
        path = f"images/{crop.lower()}{ext}"
        if os.path.exists(path):
            return path
    return None


# ========================
# Load Dataset
# ========================
data = pd.read_csv("dataset.csv")

# Encode labels
le = LabelEncoder()
data['label'] = le.fit_transform(data['label'])

# ========================
# Split Data (for accuracy)
# ========================
X = data.drop(['label', 'Yield'], axis=1)
y = data['label']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

X_reg = data.drop(['label', 'Yield'], axis=1)
y_reg = data['Yield']

X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(
    X_reg, y_reg, test_size=0.2, random_state=42
)

# ========================
# CACHE MODELS (IMPORTANT 🔥)
# ========================
@st.cache_resource
def train_models():
    clf = RandomForestClassifier()
    clf.fit(X_train, y_train)

    reg = RandomForestRegressor()
    reg.fit(X_train_r, y_train_r)

    return clf, reg

clf, reg = train_models()

# Accuracy (correct way)
y_pred_test = clf.predict(X_test)
acc = accuracy_score(y_test, y_pred_test)

# ========================
# UI CONFIG
# ========================
st.set_page_config(page_title="AI Crop System", layout="wide")

st.title("🌾 AI Crop Recommendation & Yield Prediction")
st.markdown("---")

# ========================
# INPUT SECTION
# ========================
col1, col2 = st.columns(2)

with col1:
    st.subheader("🌱 Soil Inputs")
    N = st.slider("Nitrogen", 0, 150, 50)
    P = st.slider("Phosphorus", 0, 150, 50)
    K = st.slider("Potassium", 0, 150, 50)

with col2:
    st.subheader("🌦️ Weather Inputs")
    temp = st.slider("Temperature (°C)", 0.0, 50.0, 25.0)
    humidity = st.slider("Humidity (%)", 0.0, 100.0, 50.0)
    ph = st.slider("pH Value", 0.0, 14.0, 7.0)
    rainfall = st.slider("Rainfall (mm)", 0.0, 300.0, 100.0)

st.markdown("---")

# ========================
# PREDICTION
# ========================
if st.button("🚀 Predict"):

    input_data = np.array([[N, P, K, temp, humidity, ph, rainfall]])

    with st.spinner("Predicting..."):
        crop_pred = clf.predict(input_data)
        crop_name = le.inverse_transform(crop_pred)[0]

        yield_pred = reg.predict(input_data)[0]

    # Output
    st.success(f"🌾 Recommended Crop: **{crop_name.upper()}**")
    st.info(f"📈 Predicted Yield: **{yield_pred:.2f}**")

    # Show image
    img_path = get_image_path(crop_name)

    if img_path:
        st.image(img_path, caption=crop_name, width=300)
    else:
        st.warning("Image not available")
    # Chart
    st.subheader("📊 Input Feature Visualization")

    chart_data = pd.DataFrame({
        'Feature': ['N', 'P', 'K', 'Temp', 'Humidity', 'pH', 'Rainfall'],
        'Value': [N, P, K, temp, humidity, ph, rainfall]
    })

    st.bar_chart(chart_data.set_index('Feature'))

# ========================
# SIDEBAR
# ========================
st.sidebar.title("📌 Model Info")

st.sidebar.write(f"✔ Model Accuracy: {acc*100:.2f}%")

st.sidebar.write("""
### Features Used:
- Nitrogen (N)
- Phosphorus (P)
- Potassium (K)
- Temperature
- Humidity
- pH
- Rainfall
""")

st.sidebar.write("### Models Used:")
st.sidebar.write("- Random Forest (Classification)")
st.sidebar.write("- Random Forest (Regression)")