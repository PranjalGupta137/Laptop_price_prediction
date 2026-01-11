import streamlit as st
import joblib
import numpy as np
import pandas as pd
import requests
import os
import base64
from streamlit_lottie import st_lottie

# 1. Page Config
st.set_page_config(page_title="Laptop Predictor", layout="centered", page_icon="💻")

# 2. CSS for Amazon Style Cards (Only for Recommendation Section)
st.markdown("""
    <style>
    .card {
        background: white; border-radius: 10px; padding: 15px;
        border: 1px solid #ddd; text-align: center; height: 420px;
        box-shadow: 2px 2px 10px rgba(0,0,0,0.05);
    }
    .brand-tag {
        background: #232f3e; color: white; padding: 2px 8px;
        border-radius: 5px; font-size: 10px; font-weight: bold;
    }
    .price-tag { color: #B12704; font-size: 20px; font-weight: bold; }
    </style>
    """, unsafe_allow_html=True)

# 3. Load Resources with Error Handling
@st.cache_resource
def load_all():
    try:
        df = pd.read_csv("https://raw.githubusercontent.com/campusx-official/laptop-price-predictor-regression-project/main/laptop_data.csv")
        model = joblib.load("laptop_price_prediction.pkl")
        enc_cpu = joblib.load("cpu_encoder.pkl")
        enc_gpu = joblib.load("gpu_encoder.pkl")
        return df, model, enc_cpu, enc_gpu
    except Exception as e:
        st.error("Model files load nahi ho payi. GitHub check karein!")
        st.stop()

df, model, encoder_cpu, encoder_gpu = load_all()

# 4. Processor Mapping (Full Names for 2025-26)
cpu_map = {
    "Intel Core i5": "Intel Core i5-13500H (13th Gen)",
    "Intel Core i7": "Intel Core i7-13700H (High End)",
    "AMD Ryzen 5": "AMD Ryzen 5 5600H / 7535HS",
    "AMD Ryzen 7": "AMD Ryzen 7 5800H / 7735HS",
    "Intel Core i3": "Intel Core i3-1215U (Budget)",
    "Intel Core i9": "Intel Core i9-13980HX (Extreme)",
    "Other Intel Processor": "Intel Core Ultra / Celeron",
    "Other AMD Processor": "AMD Ryzen 3 / Athlon"
}

# --- UI START (Purana Layout: Everything Centered) ---

# Animation (Center mein)
try:
    res = requests.get("https://lottie.host/85a1936c-2f96-4191-bc10-097587841c62/An2Bv8K763.json")
    if res.status_code == 200:
        st_lottie(res.json(), height=200, key="main_ani")
except:
    st.title("💻")

st.title("Advanced Laptop Price Predictor")
st.write("---")

# Input Fields
ram = st.selectbox("RAM (GB)", [4, 8, 16, 32, 64], index=1)
weight = st.number_input("Weight (kg)", 0.5, 4.0, 1.6)

# Processor with Full Names
cpu_display = st.selectbox("Processor", list(cpu_map.values()))
cpu_orig = [k for k, v in cpu_map.items() if v == cpu_display][0]

gpu = st.selectbox("GPU Brand", encoder_gpu.classes_)

# Prediction Button
if st.button("Predict Price", use_container_width=True):
    # Click sound effect
    st.components.v1.html("<audio autoplay><source src='https://www.soundjay.com/buttons/sounds/button-4.mp3' type='audio/mp3'></audio>", height=0)
    
    # ML Logic
    cpu_enc = encoder_cpu.transform([cpu_orig])[0]
    gpu_enc = encoder_gpu.transform([gpu])[0]
    pred = int(model.predict(np.array([[ram, weight, cpu_enc, gpu_enc]]))[0])
    
    st.balloons()
    st.success(f"### Estimated Market Price: ₹{pred:,}")
    
    st.write("---")
    st.subheader("🛒 Recommended Laptops in this Range")
    
    # Recommendations (Amazon Style)
    df['diff'] = abs(df['Price'] - pred)
    matches = df.sort_values('diff').head(3) # Top 3 laptops
    
    cols = st.columns(3)
    for i, (idx, row) in enumerate(matches.iterrows()):
        brand = row['Company']
        img = f"https://source.unsplash.com/400x300/?laptop,{brand.lower()}"
        
        with cols[i]:
            st.markdown(f"""
                <div class="card">
                    <span class="brand-tag">{brand.upper()}</span>
                    <img src="{img}" style="width:100%; height:150px; object-fit:contain; margin-top:10px;">
                    <p style="font-weight:bold; margin-top:10px; height:40px; overflow:hidden;">{brand} {row['TypeName']}</p>
                    <p style="font-size:12px; color:gray;">{row['Cpu']}</p>
                    <p class="price-tag">₹{int(row['Price']):,}</p>
                </div>
                """, unsafe_allow_html=True)
            search_query = f"https://www.amazon.in/s?k={brand}+laptop+{row['Ram']}".replace(" ", "+")
            st.link_button(f"View on Amazon", search_query, use_container_width=True)
