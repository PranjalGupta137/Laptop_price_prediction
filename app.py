import streamlit as st
import joblib
import numpy as np
import pandas as pd
import requests
import base64
import os

# 1. Page Config
st.set_page_config(page_title="Laptop Price Expert", layout="wide", page_icon="💻")

# 2. CSS - Ultra Bold Text & Premium Look
st.markdown("""
    <style>
    .main { background-color: #f1f3f6; }
    .card {
        background-color: white; padding: 25px; border-radius: 15px;
        border: 1px solid #ddd; text-align: center; height: 600px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
    }
    .card img { width: 100%; height: 240px; object-fit: contain; }
    
    /* Laptop Name: ULTRA BOLD */
    .laptop-name { 
        font-size: 24px !important; 
        font-weight: 900 !important; 
        color: #111; 
        margin-top: 15px;
        text-transform: uppercase;
    }
    /* Specs Text: BOLD */
    .specs-text { 
        font-size: 18px !important; 
        font-weight: 700 !important; 
        color: #333; 
        margin: 12px 0;
    }
    /* Price: BIG RED BOLD */
    .price-tag { 
        color: #B12704; 
        font-size: 30px !important; 
        font-weight: 900 !important; 
    }
    .stButton>button { 
        background-color: #febd69; color: black; font-weight: bold; 
        height: 60px; font-size: 22px; border-radius: 10px;
        border: 2px solid #a88734;
    }
    </style>
    """, unsafe_allow_html=True)

# 3. Background Music Function (Base64 Embedding)
def get_audio_html(file_path):
    if os.path.exists(file_path):
        with open(file_path, "rb") as f:
            data = f.read()
            b64 = base64.b64encode(data).decode()
            # Loop and Autoplay enabled
            return f"""
                <audio id="bg-audio" loop autoplay>
                    <source src="data:audio/mpeg;base64,{b64}" type="audio/mpeg">
                </audio>
                <script>
                    var audio = document.getElementById("bg-audio");
                    audio.volume = 0.4; // Setting volume to 40%
                    document.body.addEventListener('click', function() {{
                        audio.play();
                    }}, {{ once: true }});
                </script>
            """
    return ""

# Add Background Music (Check for .mp3 or .mpeg)
music_html = get_audio_html("background_music.mp3") or get_audio_html("background_music.mpeg")
st.components.v1.html(music_html, height=0)

# 4. Resources Loading
@st.cache_resource
def load_all():
    df = pd.read_csv("https://raw.githubusercontent.com/campusx-official/laptop-price-predictor-regression-project/main/laptop_data.csv")
    model = joblib.load("laptop_price_prediction.pkl")
    enc_cpu = joblib.load("cpu_encoder.pkl")
    enc_gpu = joblib.load("gpu_encoder.pkl")
    return df, model, enc_cpu, enc_gpu

df, model, encoder_cpu, encoder_gpu = load_all()

# Safe Mapping Logic
cpu_safe_map = {"Intel Core i9":"Intel Core i7", "Intel Core i7":"Intel Core i7", "Intel Core i5":"Intel Core i5", "Intel Core i3":"Intel Core i3", "AMD Ryzen 9":"AMD Ryzen 7", "AMD Ryzen 7":"AMD Ryzen 7", "AMD Ryzen 5":"AMD Ryzen 5", "Other Intel Processor":"Other Intel Processor", "Other AMD Processor":"Other AMD Processor"}

# --- MAIN UI ---
st.title("🚀 Premium Laptop Price Predictor 2026")

col1, col2 = st.columns([1, 2])
with col1:
    st.image("https://cdn-icons-png.flaticon.com/512/4213/4213511.png", width=300)
    st.info("💡 Note: Click anywhere on the page to start background music!")

with col2:
    st.subheader("Personalize Your Search")
    ca, cb = st.columns(2)
    with ca:
        ram = st.selectbox("RAM (GB)", [4, 8, 16, 32, 64], index=1)
        purpose = st.selectbox("Usage Purpose", ["Gaming", "Editing", "Office", "Mixed"])
    with cb:
        cpu_choice = st.selectbox("Processor Tier", list(cpu_safe_map.keys()))
        gpu = st.selectbox("Graphics Model", encoder_gpu.classes_)

# 5. Prediction
if st.button("🔥 Calculate Market Price"):
    # Instant Sound Effect for Interaction
    st.components.v1.html("""<audio autoplay><source src="https://www.soundjay.com/buttons/sounds/button-20.mp3"></audio>""", height=0)
    
    cpu_safe = cpu_safe_map[cpu_choice]
    cpu_enc = encoder_cpu.transform([cpu_safe])[0]
    gpu_enc = encoder_gpu.transform([gpu])[0]
    
    # Predict (1.6kg reference)
    pred = model.predict(np.array([[ram, 1.6, cpu_enc, gpu_enc]]))[0]
    final_val = int(pred * 1.04)
    if "i9" in cpu_choice or "Ryzen 9" in cpu_choice: final_val += 15000

    st.markdown(f"<h1 style='text-align: center; color: #B12704; font-size: 50px;'>ESTIMATED: ₹{final_val:,}</h1>", unsafe_allow_html=True)

    # Recommendations
    st.markdown("---")
    df['diff'] = abs(df['Price'] - (final_val/1.04))
    suggestions = df.sort_values('diff').head(4)
    cols = st.columns(4)
    
    # High-quality realistic laptop images
    img_list = [
        "https://images.unsplash.com/photo-1517336714731-489689fd1ca8?w=500", 
        "https://images.unsplash.com/photo-1588872657578-7efd1f1555ed?w=500", 
        "https://images.unsplash.com/photo-1593642632823-8f785ba67e45?w=500", 
        "https://images.unsplash.com/photo-1611078489935-0cb964de46d6?w=500"
    ]

    for i, (idx, row) in enumerate(suggestions.iterrows()):
        with cols[i]:
            st.markdown(f"""
                <div class="card">
                    <img src="{img_list[i]}">
                    <p class="laptop-name">{row['Company']}</p>
                    <p class="specs-text">{row['Cpu']}<br>RAM: {row['Ram']}</p>
                    <p class="price-tag">₹{int(row['Price'] * 1.04):,}</p>
                </div>
                """, unsafe_allow_html=True)
            st.link_button("Check Deals", f"https://www.amazon.in/s?k={row['Company']}+laptop")
