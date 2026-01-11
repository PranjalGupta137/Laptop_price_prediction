import streamlit as st
import joblib
import numpy as np
import pandas as pd
import base64
import os

# 1. Page Configuration
st.set_page_config(page_title="Laptop Price Prediction AI | Pranjal Gupta", layout="wide", page_icon="💻")

# 2. Custom CSS for Spacing & UI
st.markdown("""
    <style>
    .stApp { background-color: #0f1111; color: white; }
    
    /* Welcome Screen Spacing */
    .welcome-container {
        text-align: center; padding: 60px; 
        background: linear-gradient(45deg, #232f3e, #000000);
        border-radius: 25px; border: 2px solid #febd69; 
        margin: 5% auto; /* Thoda upar kar diya */
        max-width: 850px;
    }

    /* Main Header Spacing */
    .main-header {
        text-align: center; 
        color: #febd69; 
        margin-top: 40px; /* Chipakne se bachane ke liye gap */
        margin-bottom: 30px;
        font-size: 45px;
        font-weight: bold;
    }

    /* Card Styling */
    .card {
        background-color: white; padding: 15px; border-radius: 15px;
        text-align: center; height: 500px; border: 1px solid #ddd;
    }
    .card img {
        width: 100%; height: 180px; 
        object-fit: contain; background-color: #f9f9f9;
        border-radius: 10px; margin-bottom: 15px;
    }
    .laptop-name { font-size: 20px !important; font-weight: 900 !important; color: #111; }
    .price-tag { color: #B12704; font-size: 24px !important; font-weight: 900 !important; }

    /* Info Box */
    .info-box {
        background-color: #1e293b; padding: 25px; border-radius: 15px;
        border: 1px solid #334155; margin-top: 10px;
    }
    </style>
    """, unsafe_allow_html=True)

# 3. Audio Logic
def get_audio_html(file_path):
    if os.path.exists(file_path):
        with open(file_path, "rb") as f:
            data = f.read()
            b64 = base64.b64encode(data).decode()
            return f"""<audio id="bg-audio" loop autoplay><source src="data:audio/mpeg;base64,{b64}" type="audio/mpeg"></audio>
                       <script>var audio = document.getElementById("bg-audio"); audio.volume = 0.3; audio.play();</script>"""
    return ""

if 'enter' not in st.session_state:
    st.session_state.enter = False

# --- SCREEN 1: WELCOME ---
if not st.session_state.enter:
    col1, col2, col3 = st.columns([0.5, 2, 0.5])
    with col2:
        st.markdown("""
            <div class="welcome-container">
                <h1 style='color:#febd69; font-size: 55px; margin-bottom:10px;'>LAPTOP PRICE PREDICTION AI</h1>
                <p style='font-size: 22px;'>Smart Market Valuations by Pranjal Gupta</p>
            </div>
        """, unsafe_allow_html=True)
        if st.button("🚀 OPEN DASHBOARD", use_container_width=True):
            st.session_state.enter = True
            st.rerun()

# --- SCREEN 2: MAIN DASHBOARD ---
else:
    st.components.v1.html(get_audio_html("background_music.mpeg") or get_audio_html("background_music.mp3"), height=0)

    @st.cache_resource
    def load_all():
        df = pd.read_csv("https://raw.githubusercontent.com/campusx-official/laptop-price-predictor-regression-project/main/laptop_data.csv")
        model = joblib.load("laptop_price_prediction.pkl")
        enc_cpu = joblib.load("cpu_encoder.pkl")
        enc_gpu = joblib.load("gpu_encoder.pkl")
        return df, model, enc_cpu, enc_gpu

    df, model, encoder_cpu, encoder_gpu = load_all()
    cpu_safe_map = {"Intel Core i9":"Intel Core i7", "Intel Core i7":"Intel Core i7", "Intel Core i5":"Intel Core i5", "Intel Core i3":"Intel Core i3", "AMD Ryzen 9":"AMD Ryzen 7", "AMD Ryzen 7":"AMD Ryzen 7", "AMD Ryzen 5":"AMD Ryzen 5"}

    # Diamond Hata diya aur class add ki for spacing
    st.markdown("<div class='main-header'>Laptop Price Prediction Dashboard</div>", unsafe_allow_html=True)
    
    c1, c2 = st.columns([1, 2.5])
    with c1:
        st.markdown("### ⚙️ Hardware")
        ram = st.selectbox("RAM (GB)", [4, 8, 16, 32, 64], index=1)
        cpu_choice = st.selectbox("Processor", list(cpu_safe_map.keys()))
        gpu = st.selectbox("GPU", list(encoder_gpu.classes_))
        predict_click = st.button("🔥 PREDICT PRICE", use_container_width=True)

    with c2:
        if predict_click:
            cpu_val = cpu_safe_map.get(cpu_choice, "Intel Core i7")
            cpu_enc = encoder_cpu.transform([cpu_val])[0]
            gpu_enc = encoder_gpu.transform([gpu])[0]
            raw_p = model.predict(np.array([[ram, 1.6, cpu_enc, gpu_enc]]))[0]
            final_p = int(raw_p * 1.02)
            
            st.markdown(f"<div style='text-align:center; background:#1e293b; padding:25px; border-radius:15px; border:2px solid #febd69;'>"
                        f"<h1 style='color:#febd69; font-size:60px; margin:0;'>₹{final_p:,}</h1>"
                        f"<p style='color:white;'>Predicted Fair Market Price</p></div>", unsafe_allow_html=True)
            
            st.markdown("---")
            df['diff'] = abs(df['Price'] - raw_p)
            sugg = df.sort_values('diff').head(4)
            cols = st.columns(4)
            img_list = ["https://images.unsplash.com/photo-1496181133206-80ce9b88a853?w=400", "https://images.unsplash.com/photo-1517336714731-489689fd1ca8?w=400", "https://images.unsplash.com/photo-1593642632823-8f785ba67e45?w=400", "https://images.unsplash.com/photo-1611078489935-0cb964de46d6?w=400"]
            
            for i, (idx, row) in enumerate(sugg.iterrows()):
                with cols[i]:
                    st.markdown(f'<div class="card"><img src="{img_list[i]}"><p class="laptop-name">{row["Company"]}</p><p class="price-tag">₹{int(row["Price"]):,}</p></div>', unsafe_allow_html=True)
                    st.link_button("Buy Link", f"https://www.amazon.in/s?k={row['Company']}+laptop")
        else:
            st.markdown("""
                <div class="info-box">
                    <h3>🔍 Analysis Overview</h3>
                    <p>This AI model has been engineered to process multiple hardware parameters and provide an unbiased valuation.</p>
                    <hr style="border-color: #334155;">
                    <h3>📈 Market Tips</h3>
                    <p>• 16GB RAM is highly recommended for 2026 multitasking.<br>• SSD storage is 10x faster for boot speeds.</p>
                </div>
            """, unsafe_allow_html=True)

    # --- FOOTER ---
    st.markdown("<br><hr>", unsafe_allow_html=True)
    f1, f2 = st.columns(2)
    with f1:
        st.markdown('<p style="color:#febd69; font-size:22px; font-weight:bold;">👨‍💻 About Developer</p>', unsafe_allow_html=True)
        st.markdown("<p style='color:#ccc;'><b>Pranjal Gupta</b> is a Data Scientist and AI Architect. He builds smart tools to help people make better tech decisions.</p>", unsafe_allow_html=True)
    with f2:
        st.markdown('<p style="color:#febd69; font-size:22px; font-weight:bold;">📧 Feedback</p>', unsafe_allow_html=True)
        with st.form("feedback_form", clear_on_submit=True):
            u_name = st.text_input("Name")
            u_email = st.text_input("Email")
            u_msg = st.text_area("Message")
            if st.form_submit_button("Submit"):
                st.success("Feedback sent!")

    st.markdown('<p style="text-align:center; color:#444; margin-top:30px;">© 2026 | Built by Pranjal Gupta</p>', unsafe_allow_html=True)
