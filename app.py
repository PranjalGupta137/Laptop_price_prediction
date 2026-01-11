import streamlit as st
import joblib
import numpy as np
import pandas as pd
import requests
import base64
import os  # <--- Ye miss tha, ab fix ho gaya

# 1. Page Config
st.set_page_config(page_title="Laptop Price Expert", layout="wide", page_icon="💻")

# 2. CSS - Text Bada, Bold aur Premium UI
st.markdown("""
    <style>
    .main { background-color: #f1f3f6; }
    .card {
        background-color: white; padding: 25px; border-radius: 15px;
        border: 1px solid #ddd; text-align: center; height: 580px;
        box-shadow: 0 4px 8px rgba(0,0,0,0.05);
    }
    .card img { width: 100%; height: 230px; object-fit: contain; }
    
    /* Laptop Name: BADA AUR BOLD */
    .laptop-name { 
        font-size: 22px !important; 
        font-weight: 800 !important; 
        color: #111; 
        margin-top: 15px;
        line-height: 1.2;
    }
    /* Specs Text: Medium Bold */
    .specs-text { 
        font-size: 17px !important; 
        font-weight: 600 !important; 
        color: #444; 
        margin: 10px 0;
    }
    /* Price: EXTRA BOLD RED */
    .price-tag { 
        color: #B12704; 
        font-size: 28px !important; 
        font-weight: 900 !important; 
    }
    .stButton>button { 
        background-color: #febd69; color: black; font-weight: bold; 
        height: 55px; font-size: 20px; border-radius: 8px;
    }
    </style>
    """, unsafe_allow_html=True)

# 3. Background Music (Supporting MPEG/MP3)
# File name should be 'background_music.mp3' or 'background_music.mpeg'
music_file = "background_music.mp3" 
if not os.path.exists(music_file):
    music_file = "background_music.mpeg"

if os.path.exists(music_file):
    with open(music_file, "rb") as f:
        data = f.read()
        b64 = base64.b64encode(data).decode()
        st.markdown(f"""
            <audio autoplay loop>
                <source src="data:audio/mpeg;base64,{b64}" type="audio/mpeg">
            </audio>
            """, unsafe_allow_html=True)

# 4. Load Data & Model
@st.cache_resource
def load_all():
    df = pd.read_csv("https://raw.githubusercontent.com/campusx-official/laptop-price-predictor-regression-project/main/laptop_data.csv")
    model = joblib.load("laptop_price_prediction.pkl")
    enc_cpu = joblib.load("cpu_encoder.pkl")
    enc_gpu = joblib.load("gpu_encoder.pkl")
    return df, model, enc_cpu, enc_gpu

df, model, encoder_cpu, encoder_gpu = load_all()

# Safe Mapping
cpu_safe_map = {"Intel Core i9":"Intel Core i7", "Intel Core i7":"Intel Core i7", "Intel Core i5":"Intel Core i5", "Intel Core i3":"Intel Core i3", "AMD Ryzen 9":"AMD Ryzen 7", "AMD Ryzen 7":"AMD Ryzen 7", "AMD Ryzen 5":"AMD Ryzen 5", "Other Intel Processor":"Other Intel Processor", "Other AMD Processor":"Other AMD Processor"}

# --- UI ---
st.title("💻 Premium Laptop Price Predictor")
c1, c2 = st.columns([1, 2])
with c1:
    st.image("https://cdn-icons-png.flaticon.com/512/4213/4213511.png", width=280)

with c2:
    st.subheader("Select Specifications")
    ca, cb = st.columns(2)
    with ca:
        ram = st.selectbox("RAM (GB)", [4, 8, 16, 32, 64], index=1)
        purpose = st.selectbox("Purpose", ["Gaming", "Editing", "Office", "Student"])
    with cb:
        cpu_choice = st.selectbox("Processor", list(cpu_safe_map.keys()))
        gpu = st.selectbox("GPU", encoder_gpu.classes_)

# Predict Button
if st.button("Predict Accurate Price"):
    # Instant Sound Effect
    st.components.v1.html("""<audio autoplay><source src="https://www.soundjay.com/buttons/sounds/button-20.mp3"></audio>""", height=0)
    
    cpu_safe = cpu_safe_map[cpu_choice]
    cpu_enc = encoder_cpu.transform([cpu_safe])[0]
    gpu_enc = encoder_gpu.transform([gpu])[0]
    
    # ML Prediction (1.6kg Weight)
    pred = model.predict(np.array([[ram, 1.6, cpu_enc, gpu_enc]]))[0]
    final_price = int(pred * 1.04)
    if "i9" in cpu_choice or "Ryzen 9" in cpu_choice: final_price += 15000

    st.markdown(f"<h1 style='text-align: center; color: #B12704; font-size: 45px;'>Estimated Price: ₹{final_price:,}</h1>", unsafe_allow_html=True)

    # Recommendations
    st.markdown("---")
    st.subheader("🛒 Recommended Models")
    df['diff'] = abs(df['Price'] - (final_price/1.04))
    suggestions = df.sort_values('diff').head(4)
    cols = st.columns(4)
    
    img_list = ["https://images.unsplash.com/photo-1517336714731-489689fd1ca8?w=400", "https://images.unsplash.com/photo-1588872657578-7efd1f1555ed?w=400", "https://images.unsplash.com/photo-1593642632823-8f785ba67e45?w=400", "https://images.unsplash.com/photo-1611078489935-0cb964de46d6?w=400"]

    for i, (idx, row) in enumerate(suggestions.iterrows()):
        with cols[i]:
            st.markdown(f"""
                <div class="card">
                    <img src="{img_list[i]}">
                    <p class="laptop-name">{row['Company']} {row['TypeName']}</p>
                    <p class="specs-text">{row['Cpu']}<br>RAM: {row['Ram']}</p>
                    <p class="price-tag">₹{int(row['Price'] * 1.04):,}</p>
                </div>
                """, unsafe_allow_html=True)
            st.link_button("View on Amazon", f"https://www.amazon.in/s?k={row['Company']}+laptop")
