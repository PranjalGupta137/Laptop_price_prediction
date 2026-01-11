import streamlit as st
import joblib
import numpy as np
import pandas as pd
import requests
import random
import os
import base64
from streamlit_lottie import st_lottie

# 1. Page Config
st.set_page_config(page_title="2026 Laptop Predictor", layout="wide", page_icon="💻")

# 2. Error-Free Audio Function
def play_audio(file_path):
    try:
        if os.path.exists(file_path):
            with open(file_path, "rb") as f:
                data = f.read()
                b64 = base64.b64encode(data).decode()
                md = f"""<audio autoplay loop><source src="data:audio/mp3;base64,{b64}" type="audio/mp3"></audio>"""
                st.markdown(md, unsafe_allow_html=True)
    except Exception:
        pass # Agar audio mein error aaye toh ignore karo

# 3. Custom CSS for Amazon Style UI
st.markdown("""
    <style>
    .main { background-color: #f4f6f9; }
    .card {
        background: white; border-radius: 15px; padding: 20px;
        border: 1px solid #e0e0e0; text-align: center; height: 500px;
        transition: 0.3s; position: relative; box-shadow: 0 4px 6px rgba(0,0,0,0.05);
    }
    .card:hover { transform: translateY(-5px); box-shadow: 0 12px 20px rgba(0,0,0,0.1); }
    .brand-tag {
        position: absolute; top: 15px; left: 15px;
        background: #232f3e; color: white; padding: 4px 12px;
        border-radius: 20px; font-weight: bold; font-size: 11px;
    }
    .card img { width: 100%; height: 200px; object-fit: contain; margin-bottom: 15px; }
    .price-tag { color: #B12704; font-size: 24px; font-weight: bold; }
    </style>
    """, unsafe_allow_html=True)

# --- Background Music (Optional) ---
play_audio("bg_music.mp3")

# 4. Load Model & Data with Safety Checks
@st.cache_resource
def load_all_resources():
    try:
        # Online Dataset load karna
        df = pd.read_csv("https://raw.githubusercontent.com/campusx-official/laptop-price-predictor-regression-project/main/laptop_data.csv")
        # Local files load karna (Names exactly match with your GitHub)
        model = joblib.load("laptop_price_prediction.pkl")
        enc_cpu = joblib.load("cpu_encoder.pkl")
        enc_gpu = joblib.load("gpu_encoder.pkl")
        return df, model, enc_cpu, enc_gpu
    except Exception as e:
        st.error(f"Critical Error: Files check karein! {e}")
        st.stop()

df, model, encoder_cpu, encoder_gpu = load_all_resources()

# 5. Processor Display Mapping (Full Names)
cpu_map = {
    "Intel Core i5": "Intel Core i5 (13th Gen / 2025 Series)",
    "Intel Core i7": "Intel Core i7 (High Performance)",
    "AMD Ryzen 5": "AMD Ryzen 5 (5000/7000 Series)",
    "AMD Ryzen 7": "AMD Ryzen 7 (Octa-Core Beast)",
    "Intel Core i3": "Intel Core i3 (Budget Series)",
    "Intel Core i9": "Intel Core i9 (Extreme Gaming)",
    "Other Intel Processor": "Intel Core Ultra / Celeron",
    "Other AMD Processor": "AMD Ryzen 3 / Athlon"
}

# --- UI Header ---
st.title("🚀 Next-Gen Laptop AI Predictor")

c1, c2 = st.columns([1, 2])
with c1:
    try:
        res = requests.get("https://lottie.host/85a1936c-2f96-4191-bc10-097587841c62/An2Bv8K763.json")
        if res.status_code == 200:
            st_lottie(res.json(), height=250)
    except:
        st.image("https://cdn-icons-png.flaticon.com/512/4213/4213511.png", width=200)

with c2:
    st.subheader("Configure Your Laptop")
    i1, i2 = st.columns(2)
    with i1:
        ram = st.selectbox("RAM (GB)", [4, 8, 16, 32, 64], index=1)
        weight = st.number_input("Weight (kg)", 0.5, 4.0, 1.6)
    with i2:
        # Processor ke full names dikhana
        cpu_display = st.selectbox("Processor", list(cpu_map.values()))
        # Original name wapis lena model ke liye
        cpu_orig = [k for k, v in cpu_map.items() if v == cpu_display][0]
        gpu = st.selectbox("GPU Brand", encoder_gpu.classes_)

# 6. Predict & Recommend
if st.button("Analyze Market Price"):
    # Click sound effect (Internet based)
    st.components.v1.html("<audio autoplay><source src='https://www.soundjay.com/buttons/sounds/button-4.mp3' type='audio/mp3'></audio>", height=0)
    
    # ML Prediction
    cpu_enc = encoder_cpu.transform([cpu_orig])[0]
    gpu_enc = encoder_gpu.transform([gpu])[0]
    pred = int(model.predict(np.array([[ram, weight, cpu_enc, gpu_enc]]))[0])
    
    st.balloons()
    st.markdown(f"<h2 style='text-align:center;'>Estimated Value: ₹{pred:,}</h2>", unsafe_allow_html=True)
    
    st.markdown("---")
    st.subheader("🛒 Best Deals & Similar Models")
    
    # Accurate Filtering
    df['diff'] = abs(df['Price'] - pred)
    matches = df.sort_values('diff').head(4)
    
    cols = st.columns(4)
    for i, (idx, row) in enumerate(matches.iterrows()):
        brand = row['Company']
        # Stable Realistic Images from Pixabay/Unsplash
        img = f"https://source.unsplash.com/400x300/?laptop,{brand.lower()}"
        
        with cols[i]:
            st.markdown(f"""
                <div class="card">
                    <div class="brand-tag">{brand}</div>
                    <img src="{img}">
                    <div style="font-weight:bold; height:50px;">{brand} {row['TypeName']}</div>
                    <p style="font-size:12px; color:gray;">{row['Cpu']} | {row['Ram']}</p>
                    <div class="price-tag">₹{int(row['Price']):,}</div>
                </div>
                """, unsafe_allow_html=True)
            search_query = f"https://www.amazon.in/s?k={brand}+laptop+{row['Ram']}".replace(" ", "+")
            st.link_button("View Deal", search_query, use_container_width=True)
