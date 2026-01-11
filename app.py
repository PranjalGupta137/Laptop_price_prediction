import streamlit as st
import joblib
import numpy as np
import pandas as pd
import base64
import os

# 1. Page Config
st.set_page_config(page_title="Laptop AI Expert", layout="wide", page_icon="💻")

# 2. CSS - Premium Dark Theme & Layout Fix
st.markdown("""
    <style>
    .stApp { background-color: #0f1111; color: white; }
    
    /* Welcome Screen */
    .welcome-container {
        background: linear-gradient(45deg, #232f3e, #000000);
        padding: 50px; border-radius: 20px; text-align: center;
        border: 2px solid #febd69; margin-top: 30px;
    }
    
    /* Sidebar/Left Side Stats */
    .stats-box {
        background-color: #1e293b; padding: 20px; border-radius: 15px;
        border-left: 5px solid #febd69; margin-bottom: 20px;
    }

    /* Product Cards */
    .card {
        background-color: white; padding: 20px; border-radius: 15px;
        text-align: center; height: 560px; border: 3px solid #f3f3f3;
    }
    .laptop-name { font-size: 24px !important; font-weight: 900 !important; color: #111; }
    .price-tag { color: #B12704; font-size: 28px !important; font-weight: 900 !important; }
    
    /* Buttons */
    .stButton>button { 
        background-color: #febd69 !important; color: black !important; 
        font-weight: 900 !important; font-size: 20px !important; width: 100%;
    }
    </style>
    """, unsafe_allow_html=True)

# 3. Sound Logic
def get_audio_html(file_path):
    if os.path.exists(file_path):
        with open(file_path, "rb") as f:
            data = f.read()
            b64 = base64.b64encode(data).decode()
            return f"""<audio id="bg-audio" loop autoplay><source src="data:audio/mpeg;base64,{b64}" type="audio/mpeg"></audio>
                       <script>var audio = document.getElementById("bg-audio"); audio.volume = 0.3; audio.play();</script>"""
    return ""

# --- APP FLOW ---
if 'enter' not in st.session_state:
    st.session_state.enter = False

if not st.session_state.enter:
    st.markdown("""<div class="welcome-container">
                    <h1 style='color:#febd69;'>💻 LAPTOP PRICE AI 2026</h1>
                    <p style='color:white;'>Precision Pricing • Real-time Deals • Immersive Sound</p>
                   </div>""", unsafe_allow_html=True)
    if st.button("🚀 ENTER DASHBOARD"):
        st.session_state.enter = True
        st.rerun()
else:
    # Play Audio
    st.components.v1.html(get_audio_html("background_music.mpeg") or get_audio_html("background_music.mp3"), height=0)

    # Load Data
    @st.cache_resource
    def load_all():
        df = pd.read_csv("https://raw.githubusercontent.com/campusx-official/laptop-price-predictor-regression-project/main/laptop_data.csv")
        model = joblib.load("laptop_price_prediction.pkl")
        enc_cpu = joblib.load("cpu_encoder.pkl")
        enc_gpu = joblib.load("gpu_encoder.pkl")
        return df, model, enc_cpu, enc_gpu

    df, model, encoder_cpu, encoder_gpu = load_all()

    # Safe Mapping
    cpu_safe_map = {
        "Intel Core i9": "Intel Core i7", "Intel Core i7": "Intel Core i7",
        "Intel Core i5": "Intel Core i5", "Intel Core i3": "Intel Core i3",
        "AMD Ryzen 9": "AMD Ryzen 7", "AMD Ryzen 7": "AMD Ryzen 7", "AMD Ryzen 5": "AMD Ryzen 5"
    }

    # --- MAIN LAYOUT ---
    col_left, col_right = st.columns([1, 2.5])

    with col_left:
        st.markdown("### 📊 Live Specs Summary")
        ram = st.selectbox("RAM (GB)", [4, 8, 16, 32, 64], index=1)
        cpu_choice = st.selectbox("Processor", list(cpu_safe_map.keys()))
        gpu = st.selectbox("Graphics Card", encoder_gpu.classes_)
        purpose = st.selectbox("Usage", ["Gaming", "Editing", "Work"])
        
        st.markdown(f"""<div class="stats-box">
            <p><b>Current Config:</b></p>
            <p>⚡ {cpu_choice}</p>
            <p>💾 {ram}GB DDR4/5</p>
            <p>🎮 {gpu}</p>
        </div>""", unsafe_allow_html=True)

    with col_right:
        st.markdown("<h2 style='color:#febd69;'>AI Price Analysis</h2>", unsafe_allow_html=True)
        
        if st.button("🔥 PREDICT REALISTIC PRICE"):
            # Button Sound
            st.components.v1.html("""<audio autoplay><source src="https://www.soundjay.com/buttons/sounds/button-20.mp3"></audio>""", height=0)
            
            # ML Logic
            cpu_val = cpu_safe_map.get(cpu_choice, "Intel Core i7")
            cpu_enc = encoder_cpu.transform([cpu_val])[0]
            gpu_enc = encoder_gpu.transform([gpu])[0]
            
            # Pure Model Prediction (No high multipliers)
            raw_pred = model.predict(np.array([[ram, 1.6, cpu_enc, gpu_enc]]))[0]
            
            # Minimal adjustment for 2026 (Sirf 2% inflation)
            final_price = int(raw_pred * 1.02)
            
            st.markdown(f"<div style='text-align:center; background:#1e293b; padding:20px; border-radius:10px; border:2px solid #febd69;'>"
                        f"<h1 style='color:#febd69; font-size:60px; margin:0;'>₹{final_price:,}</h1>"
                        f"<p style='color:white; margin:0;'>Estimated Market Value</p></div>", unsafe_allow_html=True)

            st.markdown("---")
            st.subheader("🛒 Recommended for You")
            
            # Filter matches
            df['diff'] = abs(df['Price'] - raw_pred)
            suggestions = df.sort_values('diff').head(4)
            
            cols = st.columns(4)
            img_list = ["https://images.unsplash.com/photo-1517336714731-489689fd1ca8?w=400", "https://images.unsplash.com/photo-1588872657578-7efd1f1555ed?w=400", "https://images.unsplash.com/photo-1593642632823-8f785ba67e45?w=400", "https://images.unsplash.com/photo-1611078489935-0cb964de46d6?w=400"]

            for i, (idx, row) in enumerate(suggestions.iterrows()):
                with cols[i]:
                    st.markdown(f"""<div class="card">
                        <img src="{img_list[i]}">
                        <p class="laptop-name">{row['Company']}</p>
                        <p style="color:#444; font-weight:700;">{row['Cpu']}<br>RAM: {row['Ram']}</p>
                        <p class="price-tag">₹{int(row['Price']):,}</p>
                    </div>""", unsafe_allow_html=True)
                    st.link_button(f"Amazon {row['Company']}", f"https://www.amazon.in/s?k={row['Company']}+laptop")
