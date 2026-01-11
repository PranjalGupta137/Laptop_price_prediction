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

# 2. Advanced CSS for Amazon-style Cards & Animations
st.markdown("""
    <style>
    .main { background-color: #f1f3f6; }
    .stButton>button { 
        background-color: #febd69; color: black; border-radius: 5px; 
        border: 1px solid #a88734; font-weight: bold; width: 100%;
        transition: 0.2s;
    }
    .stButton>button:hover { background-color: #f3a847; transform: scale(1.01); }
    .card {
        background-color: white; padding: 15px; border-radius: 12px;
        border: 1px solid #ddd; text-align: center; margin-bottom: 20px;
        height: 480px; transition: 0.3s; position: relative;
    }
    .card:hover { box-shadow: 0 10px 20px rgba(0,0,0,0.12); transform: translateY(-5px); }
    .brand-tag {
        position: absolute; top: 10px; left: 10px;
        background: #232f3e; color: white; padding: 3px 10px;
        border-radius: 4px; font-size: 11px; font-weight: bold;
    }
    .card img { width: 100%; height: 200px; object-fit: contain; margin-bottom: 10px; }
    .price-tag { color: #B12704; font-size: 24px; font-weight: bold; margin-top: 10px; }
    .laptop-name { font-size: 16px; font-weight: bold; height: 45px; overflow: hidden; }
    .specs-text { font-size: 13px; color: #565959; height: 50px; overflow: hidden; margin-top: 5px; }
    </style>
    """, unsafe_allow_html=True)

# 3. Helper Functions
@st.cache_data
def load_data():
    url = "https://raw.githubusercontent.com/campusx-official/laptop-price-predictor-regression-project/main/laptop_data.csv"
    return pd.read_csv(url)

def load_lottieurl(url):
    try:
        r = requests.get(url)
        return r.json() if r.status_code == 200 else None
    except: return None

def load_model_files(file_name):
    if os.path.exists(file_name): return joblib.load(file_name)
    else:
        st.error(f"File '{file_name}' nahi mili!"); st.stop()

# 4. Loading Resources
df = load_data()
model = load_model_files("laptop_price_prediction.pkl")
encoder_cpu = load_model_files("cpu_encoder.pkl")
encoder_gpu = load_model_files("gpu_encoder.pkl")

# Mapping Full Processor Names for 2025-26
cpu_display_map = {
    "Intel Core i5": "Intel Core i5 (13th Gen / 13500H)",
    "Intel Core i7": "Intel Core i7 (High Perf / 13700H)",
    "AMD Ryzen 5": "AMD Ryzen 5 (5600H / 7535HS)",
    "AMD Ryzen 7": "AMD Ryzen 7 (5800H / 7735HS)",
    "Intel Core i3": "Intel Core i3 (12th Gen / Budget)",
    "Intel Core i9": "Intel Core i9 (13980HX / Gaming)",
    "Other Intel Processor": "Intel Core Ultra / Celeron",
    "Other AMD Processor": "AMD Ryzen 3 / Athlon"
}

# 5. UI Layout
st.title("💻 Ultimate Laptop Price Predictor")
st.write("AI-powered market estimation with real-time Amazon suggestions.")

col_main_1, col_main_2 = st.columns([1, 2])

with col_main_1:
    lottie_url = "https://lottie.host/85a1936c-2f96-4191-bc10-097587841c62/An2Bv8K763.json"
    selected_ani = load_lottieurl(lottie_url)
    if selected_ani:
        st_lottie(selected_ani, height=280, key="main_ani")

with col_main_2:
    st.subheader("Select Specifications")
    c1, c2 = st.columns(2)
    with c1:
        ram = st.selectbox("RAM (GB)", [4, 8, 16, 32, 64], index=1)
        weight = st.number_input("Weight (kg)", 0.5, 4.0, 1.6, step=0.1)
    with c2:
        # Full Names in Selectbox
        cpu_choice = st.selectbox("Processor", list(cpu_display_map.values()))
        cpu_orig = [k for k, v in cpu_display_map.items() if v == cpu_choice][0]
        gpu = st.selectbox("Graphics Card", encoder_gpu.classes_)

# 6. Prediction Logic with Instant Sound
if st.button("Predict Price & Show Deals"):
    # Instant Professional Click Sound (JS Injection)
    st.components.v1.html("""
        <audio id="clickSound" src="https://www.soundjay.com/communication/sounds/selection-sounds-01.mp3" preload="auto"></audio>
        <script>document.getElementById('clickSound').play();</script>
    """, height=0)
    
    cpu_enc = encoder_cpu.transform([cpu_orig])[0]
    gpu_enc = encoder_gpu.transform([gpu])[0]
    pred_price = int(model.predict(np.array([[ram, weight, cpu_enc, gpu_enc]]))[0])
    
    st.balloons()
    st.markdown(f"<h2 style='text-align: center; color: #232f3e;'>Estimated Market Price: ₹{pred_price:,}</h2>", unsafe_allow_html=True)
    
    st.markdown("---")
    st.subheader("🛒 Recommended Laptops (Amazon Style)")
    
    # Advanced Filtering (Closest 4 laptops)
    df['diff'] = abs(df['Price'] - pred_price)
    suggestions = df.sort_values('diff').head(4)
    
    # Brand Images Mapping
    brand_images = {
        "Apple": "https://images.unsplash.com/photo-1517336714731-489689fd1ca8?w=400",
        "Dell": "https://images.unsplash.com/photo-1588872657578-7efd1f1555ed?w=400",
        "HP": "https://images.unsplash.com/photo-1589561084283-930aa7b1ce50?w=400",
        "Lenovo": "https://images.unsplash.com/photo-1611078489935-0cb964de46d6?w=400",
        "Asus": "https://images.unsplash.com/photo-1541807084-5c52b6b3adef?w=400"
    }
    default_img = "https://images.unsplash.com/photo-1496181133206-80ce9b88a853?w=400"

    cols = st.columns(4)
    for i, (idx, row) in enumerate(suggestions.iterrows()):
        brand = row['Company']
        img = brand_images.get(brand, default_img)
        
        with cols[i]:
            st.markdown(f"""
                <div class="card">
                    <div class="brand-tag">{brand.upper()}</div>
                    <img src="{img}">
                    <div class="laptop-name">{brand} {row['TypeName']}</div>
                    <div class="specs-text">{row['Cpu']}<br><b>{row['Ram']} RAM</b></div>
                    <div class="price-tag">₹{int(row['Price']):,}</div>
                </div>
                """, unsafe_allow_html=True)
            search_query = f"https://www.amazon.in/s?k={brand}+{row['TypeName']}".replace(" ", "+")
            st.link_button(f"View Deal", search_query, use_container_width=True)

st.markdown("---")
st.caption("Updated for 2025-26 Market Trends | Powered by AI")
