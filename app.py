import streamlit as st
import joblib
import numpy as np
import pandas as pd
import requests
import random
import os
from streamlit_lottie import st_lottie

# 1. Page Configuration
st.set_page_config(page_title="Laptop Price Expert", layout="wide", page_icon="💻")

# 2. Advanced CSS for Amazon-style Product Cards
st.markdown("""
    <style>
    .main { background-color: #f1f3f6; }
    .stButton>button { background-color: #febd69; color: black; border-radius: 5px; border: 1px solid #a88734; font-weight: bold; width: 100%; height: 50px; }
    .stButton>button:hover { background-color: #f3a847; }
    .card {
        background-color: white; padding: 15px; border-radius: 12px;
        border: 1px solid #ddd; text-align: center; margin-bottom: 20px;
        height: 480px; transition: 0.3s; position: relative;
    }
    .card:hover { transform: translateY(-5px); box-shadow: 0 10px 20px rgba(0,0,0,0.1); }
    .card img { width: 100%; height: 220px; object-fit: cover; border-radius: 8px; margin-bottom: 10px; }
    .price-tag { color: #B12704; font-size: 24px; font-weight: bold; margin-top: 5px; }
    .brand-tag { position: absolute; top: 10px; left: 10px; background: #232f3e; color: white; padding: 2px 10px; border-radius: 4px; font-size: 12px; }
    </style>
    """, unsafe_allow_html=True)

# 3. Helper Functions
@st.cache_data
def load_data():
    return pd.read_csv("https://raw.githubusercontent.com/campusx-official/laptop-price-predictor-regression-project/main/laptop_data.csv")

def load_lottieurl(url):
    try:
        r = requests.get(url, timeout=5)
        return r.json() if r.status_code == 200 else None
    except: return None

# 4. Loading Resources
df = load_data()
model = joblib.load("laptop_price_prediction.pkl")
encoder_cpu = joblib.load("cpu_encoder.pkl")
encoder_gpu = joblib.load("gpu_encoder.pkl")

# Realistic High-Quality Image Mapping
brand_images = {
    "Apple": "https://images.unsplash.com/photo-1517336714731-489689fd1ca8?auto=format&fit=crop&w=400&q=80",
    "Dell": "https://images.unsplash.com/photo-1588872657578-7efd1f1555ed?auto=format&fit=crop&w=400&q=80",
    "HP": "https://images.unsplash.com/photo-1589561084283-930aa7b1ce50?auto=format&fit=crop&w=400&q=80",
    "Lenovo": "https://images.unsplash.com/photo-1611078489935-0cb964de46d6?auto=format&fit=crop&w=400&q=80",
    "Asus": "https://images.unsplash.com/photo-1544117518-3baf352aa202?auto=format&fit=crop&w=400&q=80",
    "MSI": "https://images.unsplash.com/photo-1593642702821-c8da6771f0c6?auto=format&fit=crop&w=400&q=80",
    "Acer": "https://images.unsplash.com/photo-1525547719571-a2d4ac8945e2?auto=format&fit=crop&w=400&q=80"
}
default_img = "https://images.unsplash.com/photo-1496181133206-80ce9b88a853?auto=format&fit=crop&w=400&q=80"

# Processor Safe Mapping
cpu_map = {
    "Intel Core i9": "Intel Core i7", "Intel Core i7": "Intel Core i7",
    "Intel Core i5": "Intel Core i5", "Intel Core i3": "Intel Core i3",
    "AMD Ryzen 9": "AMD Ryzen 7", "AMD Ryzen 7": "AMD Ryzen 7",
    "AMD Ryzen 5": "AMD Ryzen 5", "Other Intel Processor": "Other Intel Processor",
    "Other AMD Processor": "Other AMD Processor"
}

# --- MAIN UI ---
st.title("💻 2026 Ultimate Laptop Price Predictor")

c_main_1, c_main_2 = st.columns([1, 2])
with c_main_1:
    ani = load_lottieurl("https://lottie.host/85a1936c-2f96-4191-bc10-097587841c62/An2Bv8K763.json")
    if ani: st_lottie(ani, height=280)

with c_main_2:
    st.subheader("Select Specifications")
    i1, i2 = st.columns(2)
    with i1:
        purpose = st.selectbox("Primary Use", ["Gaming", "Editing", "Corporate", "Multi-tasking"])
        ram = st.selectbox("RAM (GB)", [8, 16, 32, 64, 128], index=1)
    with i2:
        cpu_choice = st.selectbox("Processor", list(cpu_map.keys()))
        gpu = st.selectbox("Graphics Card", encoder_gpu.classes_)
    
    refresh = st.select_slider("Display Refresh Rate", options=["60Hz", "90Hz", "120Hz", "144Hz", "165Hz", "240Hz"])

# PREDICTION
if st.button("Predict Price & Show Best Deals"):
    # Professional Click Sound
    st.components.v1.html("""<audio autoplay><source src="https://www.soundjay.com/communication/sounds/selection-sounds-01.mp3"></audio>""", height=0)
    
    # ML Logic (Weight default set to 1.8 for stability)
    cpu_safe = cpu_map[cpu_choice]
    cpu_enc = encoder_cpu.transform([cpu_safe])[0]
    gpu_enc = encoder_gpu.transform([gpu])[0]
    
    # Predict
    raw_pred = model.predict(np.array([[ram, 1.8, cpu_enc, gpu_enc]]))[0]
    
    # Calibration
    bonus = 0
    if "i9" in cpu_choice or "Ryzen 9" in cpu_choice: bonus += 35000
    if "Gaming" in purpose: bonus += 12000
    if "144Hz" in refresh: bonus += 7000
    
    final_price = int((raw_pred * 1.10) + bonus)
    
    st.balloons()
    st.markdown(f"<h2 style='text-align: center; color: #232f3e;'>Estimated 2026 Price: ₹{final_price:,}</h2>", unsafe_allow_html=True)
    
    st.markdown("---")
    st.subheader("🛒 Personalized Recommendations")
    
    # Recommendation Filtering
    df['diff'] = abs(df['Price'] - (final_price/1.10))
    suggestions = df.sort_values('diff').head(4)
    
    cols = st.columns(4)
    for i, (idx, row) in enumerate(suggestions.iterrows()):
        brand = row['Company']
        img_url = brand_images.get(brand, default_img)
        
        with cols[i]:
            st.markdown(f"""
                <div class="card">
                    <div class="brand-tag">{brand}</div>
                    <img src="{img_url}">
                    <div style="font-weight:bold; height:45px; overflow:hidden;">{brand} {row['TypeName']}</div>
                    <div style="font-size:12px; color:#565959; margin-top:5px;">{row['Cpu']}<br>RAM: {row['Ram']}</div>
                    <div class="price-tag">₹{int(row['Price'] * 1.10):,}/-</div>
                </div>
                """, unsafe_allow_html=True)
            search_url = f"https://www.amazon.in/s?k={brand}+{row['TypeName']}+{purpose}".replace(" ", "+")
            st.link_button("View on Amazon", search_url, use_container_width=True)
