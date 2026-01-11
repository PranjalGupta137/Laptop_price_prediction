import streamlit as st
import joblib
import numpy as np
import pandas as pd
import requests
import os
from streamlit_lottie import st_lottie

# 1. Page Config
st.set_page_config(page_title="2026 Laptop AI", layout="wide", page_icon="💻")

# 2. Custom CSS
st.markdown("""
    <style>
    .main { background-color: #f1f3f6; }
    .card {
        background-color: white; padding: 20px; border-radius: 12px;
        border: 1px solid #ddd; text-align: center; height: 540px;
        transition: 0.3s; position: relative;
    }
    .brand-tag {
        position: absolute; top: 10px; left: 10px;
        background: #232f3e; color: white; padding: 4px 12px;
        border-radius: 5px; font-size: 11px; font-weight: bold;
    }
    .price-tag { color: #B12704; font-size: 26px; font-weight: bold; }
    .purpose-tag { color: #007185; font-size: 14px; font-weight: bold; }
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

# 4. SAFE MAPPING (Sirf wahi keys use karein jo .pkl mein hain)
# Format: "Dropdown Name": "Original Name for Encoder"
cpu_display_map = {
    "Intel Core i9-14980HX / Ultra 9 (2025 Extreme)": "Intel Core i7", # Map to i7 for model, add bonus in logic
    "Intel Core i7-13700H / Ultra 7 (Pro Editing)": "Intel Core i7",
    "Intel Core i5-13500H / Ultra 5 (Multitasking)": "Intel Core i5",
    "AMD Ryzen 9 8945HS / 7945HX (God Mode)": "AMD Ryzen 7", # Map to Ryzen 7, add bonus
    "AMD Ryzen 7 8845HS / 7840HS (Creator)": "AMD Ryzen 7",
    "AMD Ryzen 5 7640HS / 5600H (Value)": "AMD Ryzen 5",
    "Intel Core i3-1315U (Basic Work)": "Intel Core i3",
    "Intel Celeron / Pentium / Core Ultra 3": "Other Intel Processor",
    "AMD Ryzen 3 / Athlon / Budget AMD": "Other AMD Processor"
}

# --- MAIN UI ---
st.title("🚀 Next-Gen Laptop AI Advisor 2026")

col_l, col_r = st.columns([1, 2])
with col_l:
    try:
        res = requests.get("https://lottie.host/85a1936c-2f96-4191-bc10-097587841c62/An2Bv8K763.json")
        st_lottie(res.json(), height=300)
    except: st.write("💻")

with col_r:
    st.subheader("Select Your Requirements")
    c1, c2 = st.columns(2)
    with c1:
        purpose = st.selectbox("Primary Use Case", ["Professional Editing", "Hardcore Gaming", "Corporate/Business", "Multi-tasking Mix"])
        ram = st.selectbox("RAM Size", [8, 16, 32, 64, 128], index=1)
        refresh_rate = st.selectbox("Display Refresh Rate", ["60Hz", "90Hz", "120Hz", "144Hz", "165Hz", "240Hz+"])
    with c2:
        cpu_choice = st.selectbox("Processor (All New 2025-26)", list(cpu_display_map.keys()))
        cpu_orig = cpu_display_map[cpu_choice] # Get the SAFE name for encoder
        gpu = st.selectbox("Graphics Card", encoder_gpu.classes_)
        weight = st.slider("Laptop Weight (kg)", 0.9, 4.0, 1.8)

# 5. LOGIC & PREDICTION
if st.button("Generate AI Price Quote & Recommendations"):
    # Sound Script
    st.components.v1.html("""<audio autoplay><source src="https://www.soundjay.com/buttons/sounds/button-3.mp3"></audio>""", height=0)
    
    # ML Prediction with Safe CPU mapping
    cpu_enc = encoder_cpu.transform([cpu_orig])[0]
    gpu_enc = encoder_gpu.transform([gpu])[0]
    base_pred = model.predict(np.array([[ram, weight, cpu_enc, gpu_enc]]))[0]
    
    # --- CALIBRATION LOGIC ---
    # 1. Refresh Rate Bonus
    refresh_val = int(refresh_rate.replace("Hz", "").replace("+", ""))
    hz_bonus = (refresh_val - 60) * 180 
    
    # 2. Purpose Bonus
    purpose_bonus = 0
    if purpose == "Professional Editing": purpose_bonus = 18000
    elif purpose == "Hardcore Gaming": purpose_bonus = 15000
    
    # 3. CPU Generation Bonus (Agar i9 ya Ryzen 9 select kiya hai)
    cpu_bonus = 0
    if "i9" in cpu_choice or "Ryzen 9" in cpu_choice: cpu_bonus = 30000
    elif "Ultra 7" in cpu_choice: cpu_bonus = 10000

    # Final Price with 2026 inflation
    final_price = int((base_pred * 1.15) + hz_bonus + purpose_bonus + cpu_bonus)

    st.balloons()
    st.markdown(f"<h2 style='text-align: center;'>Estimated 2026 Price: ₹{final_price:,}</h2>", unsafe_allow_html=True)
    st.markdown(f"<p style='text-align: center;' class='purpose-tag'>Optimized for {purpose} at {refresh_rate}</p>", unsafe_allow_html=True)

    st.markdown("---")
    st.subheader(f"🛒 Top Recommended {purpose} Laptops:")

    # Recommendation Logic
    df['diff'] = abs(df['Price'] - final_price)
    matches = df.sort_values('diff').head(4)
    
    cols = st.columns(4)
    for i, (idx, row) in enumerate(matches.iterrows()):
        brand = row['Company']
        img = f"https://source.unsplash.com/400x300/?laptop,{brand.lower()}"
        with cols[i]:
            st.markdown(f"""
                <div class="card">
                    <div class="brand-tag">{brand.upper()}</div>
                    <img src="{img}">
                    <div style="font-weight:bold; height:50px;">{brand} {row['TypeName']}</div>
                    <p style="font-size:12px; color:gray;">{row['Cpu']}<br>RAM: {row['Ram']}</p>
                    <p class="purpose-tag">Best for {purpose}</p>
                    <div class="price-tag">₹{int(row['Price'] * 1.18):,}</div>
                </div>
                """, unsafe_allow_html=True)
            st.link_button("View on Amazon", f"https://www.amazon.in/s?k={brand}+laptop+{purpose}")
