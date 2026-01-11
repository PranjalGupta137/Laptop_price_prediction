import streamlit as st
import joblib
import numpy as np
import pandas as pd
import requests
import os
from streamlit_lottie import st_lottie

# 1. Page Config
st.set_page_config(page_title="Laptop Price Expert", layout="wide", page_icon="💻")

# 2. Custom CSS
st.markdown("""
    <style>
    .main { background-color: #f1f3f6; }
    .card {
        background-color: white; padding: 20px; border-radius: 12px;
        border: 1px solid #ddd; text-align: center; height: 520px;
        transition: 0.3s; position: relative;
    }
    .brand-tag {
        position: absolute; top: 10px; left: 10px;
        background: #232f3e; color: white; padding: 4px 12px;
        border-radius: 5px; font-size: 11px; font-weight: bold;
    }
    .price-tag { color: #B12704; font-size: 24px; font-weight: bold; }
    .card img { width: 100%; height: 180px; object-fit: contain; margin-bottom: 10px; }
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

# Processor Map
cpu_display_map = {
    "Intel Core i9": "Intel Core i9 (13th/14th Gen)",
    "Intel Core i7": "Intel Core i7 (Pro Performance)",
    "Intel Core i5": "Intel Core i5 (Mainstream)",
    "Intel Core i3": "Intel Core i3 (Budget)",
    "AMD Ryzen 9": "AMD Ryzen 9 (Gaming)",
    "AMD Ryzen 7": "AMD Ryzen 7 (Performance)",
    "AMD Ryzen 5": "AMD Ryzen 5 (All-Rounder)",
    "Other Intel Processor": "Intel Celeron/Pentium",
    "Other AMD Processor": "AMD Athlon/A-Series"
}

# 4. UI START
st.title("💻 Ultimate Laptop Advisor")

c1, c2 = st.columns([1, 2])
with c1:
    # Lottie Fix: Using a more stable way to load
    lottie_url = "https://assets5.lottiefiles.com/packages/lf20_iv4dsx3q.json" # Alternative URL
    try:
        r = requests.get(lottie_url)
        if r.status_code == 200:
            st_lottie(r.json(), height=250)
        else: st.write("💻")
    except: st.write("💻 Laptop Predictor")

with c2:
    st.subheader("Configure Specs")
    ca, cb = st.columns(2)
    with ca:
        purpose = st.selectbox("Purpose", ["Gaming", "Editing", "Work", "Student"])
        ram = st.selectbox("RAM (GB)", [4, 8, 16, 32, 64], index=2)
    with cb:
        cpu_choice = st.selectbox("Processor", list(cpu_display_map.values()))
        cpu_orig = [k for k, v in cpu_display_map.items() if v == cpu_choice][0]
        gpu = st.selectbox("GPU", encoder_gpu.classes_)
        weight = st.slider("Weight (kg)", 1.0, 4.0, 1.8)

# 5. Prediction
if st.button("Predict Price", use_container_width=True):
    # ML Transform
    cpu_enc = encoder_cpu.transform([cpu_orig])[0]
    gpu_enc = encoder_gpu.transform([gpu])[0]
    base_pred = model.predict(np.array([[ram, weight, cpu_enc, gpu_enc]]))[0]
    
    # --- REALISTIC PRICE LOGIC (Fixing High Price) ---
    # Sirf 5-10% inflation add karenge, purana wala 15-20% zyada tha
    final_price = base_pred * 1.08 
    
    # Specific adjustment for High-end CPUs
    if "i9" in cpu_orig or "Ryzen 9" in cpu_orig:
        final_price += 25000 # Realistic i9 gap
        
    final_price = int(final_price)

    st.balloons()
    st.markdown(f"<h2 style='text-align: center;'>Estimated Price: ₹{final_price:,}</h2>", unsafe_allow_html=True)

    st.markdown("---")
    st.subheader("🛒 Market Recommendations")
    
    # Recommendations Fix
    df['diff'] = abs(df['Price'] - (final_price/1.08))
    matches = df.sort_values('diff').head(4)
    
    cols = st.columns(4)
    for i, (idx, row) in enumerate(matches.iterrows()):
        brand = row['Company']
        # Fixed Image URL
        img = f"https://images.unsplash.com/photo-1588872657578-7efd1f1555ed?w=400&q=80"
        
        with cols[i]:
            st.markdown(f"""
                <div class="card">
                    <div class="brand-tag">{brand.upper()}</div>
                    <img src="{img}">
                    <div style="font-weight:bold; height:40px;">{brand} {row['TypeName']}</div>
                    <p style="color:gray; font-size:12px;">{row['Cpu']}</p>
                    <p class="price-tag">₹{int(row['Price'] * 1.08):,}</p>
                </div>
                """, unsafe_allow_html=True)
            st.link_button("View Deal", f"https://www.amazon.in/s?k={brand}+laptop")
