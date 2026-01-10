import streamlit as st
import joblib
import numpy as np
import requests
from streamlit_lottie import st_lottie

# 1. Page Configuration (Ye sabse upar hona chahiye)
st.set_page_config(page_title="Laptop Predictor", layout="centered", page_icon="💻")

# 2. Lottie Loader Function (Error Handling ke saath)
def load_lottieurl(url):
    try:
        r = requests.get(url)
        if r.status_code != 200:
            return None
        return r.json()
    except Exception:
        return None

# Laptop Animation ka link (Updated working link)
lottie_url = "https://lottie.host/85a1936c-2f96-4191-bc10-097587841c62/An2Bv8K763.json"
lottie_laptop = load_lottieurl(lottie_url)

# 3. Load Model aur Encoders
# Ensure karein ki ye files aapke folder mein hain
model = joblib.load("laptop_price_prediction.pkl") # Ab ye sahi hai
encoder_cpu = joblib.load("cpu_encoder.pkl")
encoder_gpu = joblib.load("gpu_encoder.pkl")

# --- UI Display Start ---

# Animation dikhao (agar load ho jaye)
if lottie_laptop:
    st_lottie(lottie_laptop, height=200, key="laptop_ani")
else:
    st.title("💻") # Fallback agar animation na chale

st.title("Advanced Laptop Price Predictor")
st.write("Enter the details below to estimate the laptop price.")

# 4. Input Fields
col1, col2 = st.columns(2)

with col1:
    ram = st.number_input("RAM (GB)", min_value=2, max_value=64, value=8, step=2)
with col2:
    weight = st.number_input("Weight (kg)", min_value=0.5, max_value=5.0, value=1.5, step=0.1)

cpu = st.selectbox("Select CPU Processor", encoder_cpu.classes_)
gpu = st.selectbox("Select GPU Brand", encoder_gpu.classes_)

# 5. Prediction Logic
if st.button("Predict Price", use_container_width=True):
    # Encoding text inputs to numbers
    cpu_encoded = encoder_cpu.transform([cpu])[0]
    gpu_encoded = encoder_gpu.transform([gpu])[0]
    
    # Making prediction
    input_data = np.array([[ram, weight, cpu_encoded, gpu_encoded]])
    prediction = model.predict(input_data)
    
    # Display Result
    st.balloons() # Celebration effect!
    st.success(f"### Estimated Price: ₹{int(prediction[0]):,}")

