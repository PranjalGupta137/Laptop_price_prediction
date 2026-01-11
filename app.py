import streamlit as st
import joblib
import numpy as np
import pandas as pd
import base64
import os

# 1. Page Config
st.set_page_config(page_title="Laptop Price Expert", layout="wide", page_icon="💻")

# 2. Sound Loading Function
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
                    audio.volume = 0.3;
                    audio.play();
                </script>
            """
    return ""

# 3. CSS for Premium UI & Bold Text
st.markdown("""
    <style>
    .card {
        background-color: white; padding: 25px; border-radius: 15px;
        border: 1px solid #ddd; text-align: center; height: 620px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
    }
    .card img { width: 100%; height: 250px; object-fit: contain; }
    .laptop-name { font-size: 26px !important; font-weight: 900 !important; color: #111; }
    .price-tag { color: #B12704; font-size: 32px !important; font-weight: 900 !important; }
    .welcome-box {
        text-align: center; padding: 100px; background: white; border-radius: 20px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.2); margin-top: 50px;
    }
    </style>
    """, unsafe_allow_html=True)

# --- SESSION STATE FOR WELCOME SCREEN ---
if 'enter' not in st.session_state:
    st.session_state.enter = False

if not st.session_state.enter:
    # WELCOME SCREEN (This enables sound)
    st.markdown("""
        <div class="welcome-box">
            <h1 style='font-size: 50px;'>💻 Welcome to Laptop AI 2026</h1>
            <p style='font-size: 20px;'>Experience the future of laptop shopping with immersive sound.</p>
        </div>
    """, unsafe_allow_html=True)
    
    if st.button("🚀 ENTER APP & START MUSIC", use_container_width=True):
        st.session_state.enter = True
        st.rerun()
else:
    # --- MAIN APP CODE ---
    
    # Trigger Sound Immediately on Enter
    music_html = get_audio_html("background_music.mpeg") or get_audio_html("background_music.mp3")
    st.components.v1.html(music_html, height=0)

    # Load Resources
    @st.cache_resource
    def load_all():
        df = pd.read_csv("https://raw.githubusercontent.com/campusx-official/laptop-price-predictor-regression-project/main/laptop_data.csv")
        model = joblib.load("laptop_price_prediction.pkl")
        enc_cpu = joblib.load("cpu_encoder.pkl")
        enc_gpu = joblib.load("gpu_encoder.pkl")
        return df, model, enc_cpu, enc_gpu

    df, model, encoder_cpu, encoder_gpu = load_all()
    cpu_safe_map = {"Intel Core i9":"Intel Core i7", "Intel Core i7":"Intel Core i7", "Intel Core i5":"Intel Core i5", "Intel Core i3":"Intel Core i3", "AMD Ryzen 9":"AMD Ryzen 7", "AMD Ryzen 7":"AMD Ryzen 7", "AMD Ryzen 5":"AMD Ryzen 5", "Other Intel Processor":"Other Intel Processor", "Other AMD Processor":"Other AMD Processor"}

    st.title("🚀 Premium Laptop Price Predictor")
    
    col1, col2 = st.columns([1, 2])
    with col1:
        st.image("https://cdn-icons-png.flaticon.com/512/4213/4213511.png", width=300)
    
    with col2:
        st.subheader("Personalize Your Search")
        ca, cb = st.columns(2)
        with ca:
            ram = st.selectbox("RAM (GB)", [4, 8, 16, 32, 64], index=1)
            cpu_choice = st.selectbox("Processor Tier", list(cpu_safe_map.keys()))
        with cb:
            gpu = st.selectbox("Graphics Model", encoder_gpu.classes_)
            purpose = st.selectbox("Usage", ["Gaming", "Office", "Student"])

    if st.button("🔥 Calculate Market Price"):
        # Prediction Logic
        cpu_safe = cpu_safe_map[cpu_choice]
        cpu_enc = encoder_cpu.transform([cpu_safe])[0]
        gpu_enc = encoder_gpu.transform([gpu])[0]
        pred = model.predict(np.array([[ram, 1.6, cpu_enc, gpu_enc]]))[0]
        final_val = int(pred * 1.04)
        if "i9" in cpu_choice or "Ryzen 9" in cpu_choice: final_val += 15000

        st.markdown(f"<h1 style='text-align: center; color: #B12704; font-size: 50px;'>₹{final_val:,}</h1>", unsafe_allow_html=True)

        # Recommendations
        st.markdown("---")
        df['diff'] = abs(df['Price'] - (final_val/1.04))
        suggestions = df.sort_values('diff').head(4)
        cols = st.columns(4)
        img_list = ["https://images.unsplash.com/photo-1517336714731-489689fd1ca8?w=500", "https://images.unsplash.com/photo-1588872657578-7efd1f1555ed?w=500", "https://images.unsplash.com/photo-1593642632823-8f785ba67e45?w=500", "https://images.unsplash.com/photo-1611078489935-0cb964de46d6?w=500"]

        for i, (idx, row) in enumerate(suggestions.iterrows()):
            with cols[i]:
                st.markdown(f"""
                    <div class="card">
                        <img src="{img_list[i]}">
                        <p class="laptop-name">{row['Company']}</p>
                        <p style="font-size:18px; font-weight:700;">{row['Cpu']}<br>RAM: {row['Ram']}</p>
                        <p class="price-tag">₹{int(row['Price'] * 1.04):,}</p>
                    </div>
                    """, unsafe_allow_html=True)
