import streamlit as st
import joblib
import numpy as np
import pandas as pd
import requests
import os
from streamlit_lottie import st_lottie

# 1. Page Config
st.set_page_config(page_title="2026 Laptop Advisor", layout="wide", page_icon="💻")

# 2. Custom CSS for Premium Amazon Look
st.markdown("""
    <style>
    .main { background-color: #f1f3f6; }
    .card {
        background-color: white; padding: 20px; border-radius: 12px;
        border: 1px solid #ddd; text-align: center; height: 560px;
        transition: 0.3s; position: relative;
    }
    .brand-tag {
        position: absolute; top: 10px; left: 10px;
        background: #232f3e; color: white; padding: 4px 12px;
        border-radius: 5px; font-size: 11px; font-weight: bold;
    }
    .price-tag { color: #B12704; font-size: 26px; font-weight: bold; }
    .stButton>button { height: 50px; font-size: 18px; }
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

# --- HARDWARE MAPPING (All Generations) ---
cpu_display_map = {
    "Intel Core i9": "Intel Core i9 (13th/14th Gen - Extreme)",
    "Intel Core i7": "Intel Core i7 (All Generations - Pro)",
    "Intel Core i5": "Intel Core i5 (All Generations - Mid)",
    "Intel Core i3": "Intel Core i3 (Entry Level)",
    "Intel Core Ultra 7": "Intel Core Ultra 7/9 (2025-26 AI)",
    "AMD Ryzen 9": "AMD Ryzen 9 (High End Gaming)",
    "AMD Ryzen 7": "AMD Ryzen 7 (Performance)",
    "AMD Ryzen 5": "AMD Ryzen 5 (All-Rounder)",
    "AMD Ryzen 3": "AMD Ryzen 3 (Budget)",
    "Other Intel Processor": "Intel Celeron / Pentium / Atom",
    "Other AMD Processor": "AMD Athlon / A-Series"
}

# 4. UI START
st.title("🚀 Smart Laptop Price & Purpose Advisor (2026)")

c1, c2 = st.columns([1, 2])
with c1:
    # Lottie Error Handling: App won't crash even if JSON fails
    try:
        r = requests.get("https://lottie.host/85a1936c-2f96-4191-bc10-097587841c62/An2Bv8K763.json", timeout=5)
        if r.status_code == 200:
            st_lottie(r.json(), height=300)
        else: st.image("https://cdn-icons-png.flaticon.com/512/4213/4213511.png", width=200)
    except:
        st.write("💻 [Animation Loading...]")

with c2:
    st.subheader("Enter Specifications")
    col_a, col_b = st.columns(2)
    with col_a:
        purpose = st.selectbox("Select Purpose", ["Hardcore Gaming", "Professional Editing", "Corporate/Work", "Students/Multi-tasking"])
        ram = st.selectbox("RAM (GB)", [4, 8, 16, 32, 64, 128], index=2)
        refresh = st.selectbox("Refresh Rate", ["60Hz", "90Hz", "120Hz", "144Hz", "165Hz", "240Hz+"])
    with col_b:
        cpu_choice = st.selectbox("Select Processor", list(cpu_display_map.values()))
        # Mapping back to encoder labels
        cpu_orig = [k for k, v in cpu_display_map.items() if v == cpu_choice][0]
        # Fallback logic for new processors like "Ultra"
        if "Ultra" in cpu_orig: cpu_safe = "Intel Core i7"
        else: cpu_safe = cpu_orig
        
        gpu = st.selectbox("Graphics (GPU)", encoder_gpu.classes_)
        weight = st.slider("Laptop Weight (kg)", 0.8, 4.5, 1.8)

# 5. Prediction Logic
if st.button("Calculate Market Value & Deals", use_container_width=True):
    # Instant Sound
    st.components.v1.html("""<audio autoplay><source src="https://www.soundjay.com/buttons/sounds/button-3.mp3"></audio>""", height=0)
    
    # ML Transform
    cpu_enc = encoder_cpu.transform([cpu_safe])[0]
    gpu_enc = encoder_gpu.transform([gpu])[0]
    base_pred = model.predict(np.array([[ram, weight, cpu_enc, gpu_enc]]))[0]
    
    # --- ACCURACY CALIBRATION ---
    final_price = base_pred
    
    # CPU Tier Bonus
    if "i9" in cpu_orig or "Ryzen 9" in cpu_orig: final_price += 40000
    if "Ultra" in cpu_orig: final_price += 25000
    
    # Purpose & Refresh Rate Bonus
    if purpose == "Hardcore Gaming": final_price += 15000
    if purpose == "Professional Editing": final_price += 12000
    
    hz_val = int(refresh.split('H')[0].replace('+', ''))
    if hz_val > 60: final_price += (hz_val - 60) * 150
    
    # Market Adjustment (2026)
    final_price = int(final_price * 1.15)

    st.balloons()
    st.markdown(f"<h2 style='text-align: center; color:#232f3e;'>Estimated 2026 Price: ₹{final_price:,}</h2>", unsafe_allow_html=True)
    st.write(f"**Target Audience:** {purpose} | **Display:** {refresh}")

    # Recommendations
    st.markdown("---")
    st.subheader("🛒 Recommended Models in this Price Range")
    df['diff'] = abs(df['Price'] - (final_price/1.15))
    matches = df.sort_values('diff').head(4)
    
    cols = st.columns(4)
    for i, (idx, row) in enumerate(matches.iterrows()):
        with cols[i]:
            img = f"https://source.unsplash.com/400x300/?laptop,{row['Company'].lower()}"
            st.markdown(f"""
                <div class="card">
                    <div class="brand-tag">{row['Company'].upper()}</div>
                    <img src="{img}">
                    <p style="font-weight:bold; height:50px;">{row['Company']} {row['TypeName']}</p>
                    <p style="color:gray; font-size:12px;">{row['Cpu']}<br>RAM: {row['Ram']}</p>
                    <p style="color:#007185; font-size:13px;"><b>Best for {purpose}</b></p>
                    <p class="price-tag">₹{int(row['Price'] * 1.15):,}</p>
                </div>
                """, unsafe_allow_html=True)
            st.link_button("View on Amazon", f"https://www.amazon.in/s?k={row['Company']}+laptop+{purpose}")
