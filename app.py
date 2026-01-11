import streamlit as st
import joblib
import numpy as np
import pandas as pd
import requests
import random
import os
from streamlit_lottie import st_lottie

# 1. Page Configuration (Wide layout for Amazon look)
st.set_page_config(page_title="Laptop Price Expert", layout="wide", page_icon="💻")

# 2. Custom CSS for Amazon-style Product Cards
st.markdown("""
    <style>
    .main { background-color: #f1f3f6; }
    .stButton>button { background-color: #febd69; color: black; border-radius: 5px; border: 1px solid #a88734; font-weight: bold; }
    .stButton>button:hover { background-color: #f3a847; border: 1px solid #846a29; }
    .card {
        background-color: white;
        padding: 15px;
        border-radius: 8px;
        border: 1px solid #ddd;
        text-align: center;
        margin-bottom: 20px;
        height: 450px;
        transition: transform 0.2s;
    }
    .card:hover { transform: scale(1.02); box-shadow: 0 10px 20px rgba(0,0,0,0.1); }
    .card img {
        width: 100%;
        height: 200px;
        object-fit: contain;
        margin-bottom: 10px;
    }
    .price-tag { color: #B12704; font-size: 22px; font-weight: bold; margin-top: 10px; }
    .laptop-name { font-size: 16px; font-weight: bold; height: 45px; overflow: hidden; margin-bottom: 5px; }
    .specs-text { font-size: 13px; color: #565959; height: 40px; overflow: hidden; }
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
    except:
        return None

def load_model_files(file_name):
    if os.path.exists(file_name):
        return joblib.load(file_name)
    else:
        st.error(f"File '{file_name}' not found. Please check GitHub!")
        st.stop()

# 4. Loading Resources
df = load_data()
model = load_model_files("laptop_price_prediction.pkl")
encoder_cpu = load_model_files("cpu_encoder.pkl")
encoder_gpu = load_model_files("gpu_encoder.pkl")

# Random Animations List
lottie_urls = [
    "https://lottie.host/85a1936c-2f96-4191-bc10-097587841c62/An2Bv8K763.json",
    "https://lottie.host/5db43163-4819-487e-977a-a4869894e637/7KOnzS4I5X.json",
    "https://lottie.host/869c9b14-8f4d-4581-9b7e-96696775677d/9u0qG1w3yK.json"
]
selected_ani = load_lottieurl(random.choice(lottie_urls))

# Realistic Image Mapping for Brands
brand_images = {
    "Apple": "https://images.unsplash.com/photo-1517336714731-489689fd1ca8?w=400",
    "Dell": "https://images.unsplash.com/photo-1588872657578-7efd1f1555ed?w=400",
    "HP": "https://images.unsplash.com/photo-1589561084283-930aa7b1ce50?w=400",
    "Lenovo": "https://images.unsplash.com/photo-1611078489935-0cb964de46d6?w=400",
    "Asus": "https://images.unsplash.com/photo-1541807084-5c52b6b3adef?w=400",
    "MSI": "https://images.unsplash.com/photo-1593642702821-c8da6771f0c6?w=400",
    "Acer": "https://images.unsplash.com/photo-1525547719571-a2d4ac8945e2?w=400"
}
default_img = "https://images.unsplash.com/photo-1496181133206-80ce9b88a853?w=400"

# --- MAIN UI ---
st.title("💻 Ultimate Laptop Price Predictor")
st.write("Get AI-powered price estimates and shop recommendations instantly.")

col_main_1, col_main_2 = st.columns([1, 2])

with col_main_1:
    if selected_ani:
        st_lottie(selected_ani, height=250, key="main_ani")

with col_main_2:
    st.subheader("Select Specifications")
    c1, c2 = st.columns(2)
    with c1:
        ram = st.selectbox("RAM (GB)", [2, 4, 8, 16, 32, 64], index=2)
        weight = st.number_input("Weight (kg)", 0.5, 5.0, 1.5, step=0.1)
    with c2:
        cpu = st.selectbox("Processor", encoder_cpu.classes_)
        gpu = st.selectbox("Graphics Card", encoder_gpu.classes_)

# PREDICTION BUTTON
if st.button("Predict Price & Show Deals", use_container_width=True):
    # ML Logic
    cpu_enc = encoder_cpu.transform([cpu])[0]
    gpu_enc = encoder_gpu.transform([gpu])[0]
    input_data = np.array([[ram, weight, cpu_enc, gpu_enc]])
    pred_price = int(model.predict(input_data)[0])
    
    st.balloons()
    st.markdown(f"<h2 style='text-align: center; color: #232f3e;'>Estimated Market Price: ₹{pred_price:,}</h2>", unsafe_allow_html=True)
    
    st.markdown("---")
    st.subheader("🛒 Recommended Laptops in this Range")
    
    # Filter Similar Laptops (+/- 10000 range)
    suggestions = df[(df['Price'] >= pred_price - 10000) & (df['Price'] <= pred_price + 10000)].sample(min(4, len(df)))
    
    cols = st.columns(4)
    for i, (idx, row) in enumerate(suggestions.iterrows()):
        brand = row['Company']
        img = brand_images.get(brand, default_img)
        
        with cols[i]:
            st.markdown(f"""
                <div class="card">
                    <img src="{img}">
                    <div class="laptop-name">{brand} {row['TypeName']}</div>
                    <div class="specs-text">{row['Cpu']}<br>RAM: {row['Ram']}</div>
                    <div class="price-tag">₹{int(row['Price']):,}</div>
                </div>
                """, unsafe_allow_html=True)
            # Amazon direct search link
            search_query = f"https://www.amazon.in/s?k={brand}+{row['TypeName']}".replace(" ", "+")
            st.link_button(f"View on Amazon", search_query, use_container_width=True)

st.markdown("---")
st.caption("Data source: Laptop Price Dataset | Model: Random Forest Regressor")
