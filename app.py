import streamlit as st
import joblib
import numpy as np
import pandas as pd
import requests
import os
from streamlit_lottie import st_lottie

# 1. Page Config
st.set_page_config(page_title="Laptop Price Expert", layout="wide", page_icon="💻")

# 2. Custom CSS (Realistic UI)
st.markdown("""
    <style>
    .main { background-color: #f1f3f6; }
    .card {
        background-color: white; padding: 15px; border-radius: 8px;
        border: 1px solid #ddd; text-align: center; height: 500px;
    }
    .card img { width: 100%; height: 200px; object-fit: contain; }
    .price-tag { color: #B12704; font-size: 22px; font-weight: bold; }
    .stButton>button { background-color: #febd69; color: black; font-weight: bold; width: 100%; }
    </style>
    """, unsafe_allow_html=True)

# 3. Load Resources
@st.cache_resource
def load_all():
    df = pd.read_csv("https://raw.githubusercontent.com/campusx-official/laptop-price-predictor-regression-project/main/laptop_data.csv")
    model = joblib.load("laptop_price_prediction.pkl")
    enc_cpu = joblib.load("cpu_encoder.pkl")
    enc_gpu = joblib.load("gpu_encoder.pkl")
    return df, model, enc_cpu, enc_gpu

df, model, encoder_cpu, encoder_gpu = load_all()

# Safe Processor Mapping
cpu_safe_map = {
    "Intel Core i9": "Intel Core i7", "Intel Core i7": "Intel Core i7",
    "Intel Core i5": "Intel Core i5", "Intel Core i3": "Intel Core i3",
    "AMD Ryzen 9": "AMD Ryzen 7", "AMD Ryzen 7": "AMD Ryzen 7",
    "AMD Ryzen 5": "AMD Ryzen 5", "Other Intel Processor": "Other Intel Processor",
    "Other AMD Processor": "Other AMD Processor"
}

# 4. UI Layout
st.title("💻 Laptop Price & Deal Expert")

c1, c2 = st.columns([1, 2])
with c1:
    try:
        r = requests.get("https://lottie.host/85a1936c-2f96-4191-bc10-097587841c62/An2Bv8K763.json")
        st_lottie(r.json(), height=250)
    except: st.write("💻")

with c2:
    st.subheader("Specs Select Karein")
    col_a, col_b = st.columns(2)
    with col_a:
        purpose = st.selectbox("Purpose", ["Gaming", "Editing", "Office", "Student"])
        ram = st.selectbox("RAM (GB)", [4, 8, 16, 32, 64], index=1)
    with col_b:
        cpu_choice = st.selectbox("Processor", list(cpu_safe_map.keys()))
        gpu = st.selectbox("GPU", encoder_gpu.classes_)
    
    refresh = st.select_slider("Refresh Rate", options=["60Hz", "120Hz", "144Hz"])

# 5. Prediction Logic
if st.button("Check Realistic Price"):
    # Fix: Zero-Delay Sound Logic
    st.components.v1.html("""
        <audio autoplay>
            <source src="https://www.soundjay.com/buttons/sounds/button-20.mp3" type="audio/mp3">
        </audio>
    """, height=0)
    
    # ML Prediction
    cpu_safe = cpu_safe_map[cpu_choice]
    cpu_enc = encoder_cpu.transform([cpu_safe])[0]
    gpu_enc = encoder_gpu.transform([gpu])[0]
    
    # Base Price Calculation
    # Note: Weight default 1.6kg for realistic average
    pred = model.predict(np.array([[ram, 1.6, cpu_enc, gpu_enc]]))[0]
    
    # --- REALISTIC CALIBRATION (Price Fix) ---
    # Inflation kam kar di hai (sirf 4% for currency adjustment)
    final_price = pred * 1.04 
    
    # Gap matching for i9/Ryzen 9
    if "i9" in cpu_choice or "Ryzen 9" in cpu_choice:
        final_price += 18000 # Realistic premium
    
    final_price = int(final_price)
    
    st.balloons()
    st.markdown(f"<h2 style='text-align: center;'>Market Price: ₹{final_price:,}</h2>", unsafe_allow_html=True)

    # 6. Recommendations
    st.markdown("---")
    df['diff'] = abs(df['Price'] - (final_price/1.04))
    suggestions = df.sort_values('diff').head(4)
    
    cols = st.columns(4)
    img_list = [
        "https://images.unsplash.com/photo-1517336714731-489689fd1ca8?w=400",
        "https://images.unsplash.com/photo-1588872657578-7efd1f1555ed?w=400",
        "https://images.unsplash.com/photo-1593642632823-8f785ba67e45?w=400",
        "https://images.unsplash.com/photo-1611078489935-0cb964de46d6?w=400"
    ]
    
    for i, (idx, row) in enumerate(suggestions.iterrows()):
        with cols[i]:
            st.markdown(f"""
                <div class="card">
                    <img src="{img_list[i]}">
                    <p><b>{row['Company']} {row['TypeName']}</b></p>
                    <p style="font-size:12px; color:gray;">{row['Cpu']}<br>RAM: {row['Ram']}</p>
                    <p class="price-tag">₹{int(row['Price'] * 1.04):,}</p>
                </div>
                """, unsafe_allow_html=True)
            st.link_button("Amazon Link", f"https://www.amazon.in/s?k={row['Company']}+laptop")
