import streamlit as st
import joblib
import numpy as np
import pandas as pd
import requests
import base64

# 1. Page Config
st.set_page_config(page_title="Laptop Price Expert", layout="wide", page_icon="💻")

# 2. CSS - Text Bada aur Bold karne ke liye
st.markdown("""
    <style>
    .main { background-color: #f1f3f6; }
    .card {
        background-color: white; padding: 20px; border-radius: 12px;
        border: 1px solid #ddd; text-align: center; height: 550px;
    }
    .card img { width: 100%; height: 220px; object-fit: contain; }
    
    /* Image ke niche ka text BADA aur BOLD */
    .laptop-name { 
        font-size: 20px !important; 
        font-weight: 800 !important; 
        color: #232f3e; 
        margin-top: 10px;
    }
    .specs-text { 
        font-size: 16px !important; 
        font-weight: 600 !important; 
        color: #565959; 
    }
    .price-tag { 
        color: #B12704; 
        font-size: 26px !important; 
        font-weight: 900 !important; 
        margin-top: 10px;
    }
    .stButton>button { background-color: #febd69; color: black; font-weight: bold; height: 50px; font-size: 18px; }
    </style>
    """, unsafe_allow_html=True)

# 3. Sound Function (Instant Play)
def play_sound():
    # Instant click sound (No delay)
    sound_html = """
        <audio autoplay>
            <source src="https://www.soundjay.com/buttons/sounds/button-20.mp3" type="audio/mp3">
        </audio>
    """
    st.components.v1.html(sound_html, height=0)

# 4. Background Music Logic
if os.path.exists("background_music.mp3"):
    with open("background_music.mp3", "rb") as f:
        data = f.read()
        b64 = base64.b64encode(data).decode()
        st.markdown(f"""
            <audio autoplay loop>
                <source src="data:audio/mp3;base64,{b64}" type="audio/mp3">
            </audio>
            """, unsafe_allow_html=True)

# 5. Load Data & Model
@st.cache_resource
def load_all():
    df = pd.read_csv("https://raw.githubusercontent.com/campusx-official/laptop-price-predictor-regression-project/main/laptop_data.csv")
    model = joblib.load("laptop_price_prediction.pkl")
    enc_cpu = joblib.load("cpu_encoder.pkl")
    enc_gpu = joblib.load("gpu_encoder.pkl")
    return df, model, enc_cpu, enc_gpu

df, model, encoder_cpu, encoder_gpu = load_all()

# Mapping
cpu_safe_map = {"Intel Core i9":"Intel Core i7", "Intel Core i7":"Intel Core i7", "Intel Core i5":"Intel Core i5", "Intel Core i3":"Intel Core i3", "AMD Ryzen 9":"AMD Ryzen 7", "AMD Ryzen 7":"AMD Ryzen 7", "AMD Ryzen 5":"AMD Ryzen 5", "Other Intel Processor":"Other Intel Processor", "Other AMD Processor":"Other AMD Processor"}

# --- UI ---
st.title("💻 Premium Laptop Advisor")
c1, c2 = st.columns([1, 2])
with c1:
    st.image("https://cdn-icons-png.flaticon.com/512/4213/4213511.png", width=250)

with c2:
    st.subheader("Configuration")
    ca, cb = st.columns(2)
    with ca:
        ram = st.selectbox("RAM (GB)", [4, 8, 16, 32, 64], index=1)
        purpose = st.selectbox("Purpose", ["Gaming", "Editing", "Office"])
    with cb:
        cpu_choice = st.selectbox("Processor", list(cpu_safe_map.keys()))
        gpu = st.selectbox("GPU", encoder_gpu.classes_)

# Predict Button
if st.button("Predict Price"):
    play_sound() # Instant Sound
    
    cpu_safe = cpu_safe_map[cpu_choice]
    cpu_enc = encoder_cpu.transform([cpu_safe])[0]
    gpu_enc = encoder_gpu.transform([gpu])[0]
    
    # ML Prediction
    pred = model.predict(np.array([[ram, 1.6, cpu_enc, gpu_enc]]))[0]
    final_price = int(pred * 1.04)
    if "i9" in cpu_choice or "Ryzen 9" in cpu_choice: final_price += 15000

    st.markdown(f"<h1 style='text-align: center; color: #B12704;'>Price: ₹{final_price:,}</h1>", unsafe_allow_html=True)

    # Recommendations
    st.markdown("---")
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
            st.link_button("Amazon View", f"https://www.amazon.in/s?k={row['Company']}+laptop")
