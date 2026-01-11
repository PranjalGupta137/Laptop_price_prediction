import streamlit as st
import joblib
import numpy as np
import pandas as pd
import base64
import os

# 1. Page Config
st.set_page_config(page_title="Laptop Price AI", layout="wide", page_icon="💻")

# 2. Luxury Dark UI CSS
st.markdown("""
    <style>
    .stApp { background-color: #0f1111; } /* Amazon Dark Theme */
    
    /* Welcome Screen - Ultra Stylish */
    .welcome-container {
        background: linear-gradient(45deg, #232f3e, #000000);
        padding: 60px;
        border-radius: 25px;
        text-align: center;
        border: 2px solid #febd69;
        margin-top: 50px;
    }
    .welcome-container h1 { color: #febd69 !important; font-size: 60px !important; font-weight: 900; }
    .welcome-container p { color: #ffffff !important; font-size: 24px; font-weight: 500; }

    /* Card Styling - Realistic Look */
    .card {
        background-color: #ffffff; 
        padding: 20px; 
        border-radius: 15px;
        text-align: center; 
        height: 580px;
        border: 4px solid #f3f3f3;
        transition: 0.3s;
    }
    .card:hover { transform: translateY(-10px); border-color: #febd69; }
    .card img { width: 100%; height: 230px; object-fit: contain; }
    
    /* Text Inside Cards */
    .laptop-name { font-size: 26px !important; font-weight: 900 !important; color: #111; margin-top: 10px; }
    .specs-text { font-size: 18px !important; font-weight: 700 !important; color: #444; margin: 10px 0; }
    .price-tag { color: #B12704; font-size: 32px !important; font-weight: 900 !important; }
    
    /* Button Styling */
    .stButton>button { 
        background-color: #febd69 !important; 
        color: black !important; 
        font-weight: 900 !important; 
        font-size: 22px !important; 
        height: 60px;
        border-radius: 12px;
    }
    </style>
    """, unsafe_allow_html=True)

# 3. Sound Logic
def get_audio_html(file_path):
    if os.path.exists(file_path):
        with open(file_path, "rb") as f:
            data = f.read()
            b64 = base64.b64encode(data).decode()
            return f"""
                <audio id="bg-audio" loop autoplay>
                    <source src="data:audio/mpeg;base64,{b64}" type="audio/mpeg">
                </audio>
                <script>
                    var audio = document.getElementById("bg-audio");
                    audio.volume = 0.4;
                    audio.play();
                </script>
            """
    return ""

# --- APP FLOW ---
if 'enter' not in st.session_state:
    st.session_state.enter = False

if not st.session_state.enter:
    # WELCOME SCREEN (Stylish & Readable)
    st.markdown("""
        <div class="welcome-container">
            <h1>💻 LAPTOP AI 2026</h1>
            <p>Experience the Premium Price Predictor</p>
            <p style='font-size: 16px; color: #febd69 !important;'>Click the button below to start immersive audio</p>
        </div>
    """, unsafe_allow_html=True)
    
    st.write(" ")
    if st.button("🚀 UNLOCK EXPERIENCE", use_container_width=True):
        st.session_state.enter = True
        st.rerun()
else:
    # MAIN APP (Sound Starts Now)
    music_html = get_audio_html("background_music.mpeg") or get_audio_html("background_music.mp3")
    st.components.v1.html(music_html, height=0)

    @st.cache_resource
    def load_all():
        df = pd.read_csv("https://raw.githubusercontent.com/campusx-official/laptop-price-predictor-regression-project/main/laptop_data.csv")
        model = joblib.load("laptop_price_prediction.pkl")
        enc_cpu = joblib.load("cpu_encoder.pkl")
        enc_gpu = joblib.load("gpu_encoder.pkl")
        return df, model, enc_cpu, enc_gpu

    df, model, encoder_cpu, encoder_gpu = load_all()

    # Crash Fix Mapping
    available_cpus = list(encoder_cpu.classes_)
    cpu_safe_map = {
        "Intel Core i9": "Intel Core i7", "Intel Core i7": "Intel Core i7",
        "Intel Core i5": "Intel Core i5", "Intel Core i3": "Intel Core i3",
        "AMD Ryzen 9": "AMD Ryzen 7", "AMD Ryzen 7": "AMD Ryzen 7",
        "AMD Ryzen 5": "AMD Ryzen 5"
    }

    st.markdown("<h1 style='text-align: center; color: #febd69;'>🚀 Laptop Price Predictor</h1>", unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 2])
    with col1:
        st.image("https://cdn-icons-png.flaticon.com/512/4213/4213511.png", width=300)
    
    with col2:
        st.markdown("<h3 style='color: white;'>Select Specifications</h3>", unsafe_allow_html=True)
        ca, cb = st.columns(2)
        with ca:
            ram = st.selectbox("RAM (GB)", [4, 8, 16, 32, 64], index=2)
            cpu_choice = st.selectbox("Processor", list(cpu_safe_map.keys()))
        with cb:
            gpu = st.selectbox("Graphics Card", encoder_gpu.classes_)
            purpose = st.selectbox("Primary Usage", ["Gaming", "Editing", "Office"])

    if st.button("🔥 CALCULATE PRICE"):
        # Instant Button Sound
        st.components.v1.html("""<audio autoplay><source src="https://www.soundjay.com/buttons/sounds/button-20.mp3"></audio>""", height=0)
        
        # ML Logic
        cpu_val = cpu_safe_map.get(cpu_choice, available_cpus[0])
        cpu_enc = encoder_cpu.transform([cpu_val])[0]
        gpu_enc = encoder_gpu.transform([gpu])[0]
        
        pred = model.predict(np.array([[ram, 1.6, cpu_enc, gpu_enc]]))[0]
        final_price = int(pred * 1.04)
        if "i9" in cpu_choice or "Ryzen 9" in cpu_choice: final_price += 15000

        st.markdown(f"<h1 style='text-align: center; color: #febd69; font-size: 60px;'>₹{final_price:,}</h1>", unsafe_allow_html=True)

        st.markdown("---")
        st.subheader("🛒 Recommended Models")
        df['diff'] = abs(df['Price'] - (final_price/1.04))
        suggestions = df.sort_values('diff').head(4)
        cols = st.columns(4)
        
        # Stable Real Laptop Images
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
                st.link_button(f"Buy {row['Company']}", f"https://www.amazon.in/s?k={row['Company']}+laptop")
