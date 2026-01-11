import streamlit as st
import joblib
import numpy as np
import pandas as pd
import base64
import os

# 1. Page Configuration
st.set_page_config(page_title="Laptop AI | Pranjal Gupta", layout="wide", page_icon="💻")

# 2. Advanced CSS for UI & Footer
st.markdown("""
    <style>
    .stApp { background-color: #0f1111; color: white; }
    
    /* Centered Welcome */
    .welcome-container {
        text-align: center; padding: 50px; background: linear-gradient(45deg, #232f3e, #000000);
        border-radius: 20px; border: 2px solid #febd69; margin: 10% auto; max-width: 700px;
    }

    /* Cards Fix */
    .card {
        background-color: white; padding: 20px; border-radius: 15px;
        text-align: center; height: 550px; border: 1px solid #ddd; margin-bottom: 20px;
    }
    .laptop-name { font-size: 22px !important; font-weight: 900 !important; color: #111; }
    .price-tag { color: #B12704; font-size: 26px !important; font-weight: 900 !important; }

    /* Footer Design */
    .footer-section {
        background-color: #131a22; padding: 40px; margin-top: 50px;
        border-top: 3px solid #febd69; border-radius: 15px 15px 0 0;
    }
    .footer-heading { color: #febd69; font-size: 24px; font-weight: bold; margin-bottom: 15px; }
    .footer-text { color: #ccc; font-size: 16px; line-height: 1.6; }
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

# Session State
if 'enter' not in st.session_state:
    st.session_state.enter = False

# --- SCREEN 1: WELCOME ---
if not st.session_state.enter:
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("""
            <div class="welcome-container">
                <h1 style='color:#febd69; font-size: 55px;'>💻 LAPTOP AI</h1>
                <p style='font-size: 22px;'>Precision Data & Real-time Analytics</p>
                <p style='color:#888;'>Built by Pranjal Gupta</p>
            </div>
        """, unsafe_allow_html=True)
        if st.button("🚀 ENTER DASHBOARD", use_container_width=True):
            st.session_state.enter = True
            st.rerun()

# --- SCREEN 2: MAIN APP ---
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

    st.markdown("<h1 style='text-align:center; color:#febd69;'>🚀 Laptop Price Predictor</h1>", unsafe_allow_html=True)
    
    c1, c2 = st.columns([1, 2.5])
    with c1:
        st.markdown("### ⚙️ Config")
        ram = st.selectbox("RAM (GB)", [4, 8, 16, 32, 64], index=1)
        cpu_choice = st.selectbox("Processor", list(cpu_safe_map.keys()))
        gpu = st.selectbox("GPU", list(encoder_gpu.classes_))
        if st.button("🔥 PREDICT PRICE", use_container_width=True):
            st.session_state.predict = True
        else: st.session_state.predict = False

    with c2:
        if st.session_state.get('predict'):
            cpu_val = cpu_safe_map.get(cpu_choice, "Intel Core i7")
            cpu_enc = encoder_cpu.transform([cpu_val])[0]
            gpu_enc = encoder_gpu.transform([gpu])[0]
            raw_p = model.predict(np.array([[ram, 1.6, cpu_enc, gpu_enc]]))[0]
            final_p = int(raw_p * 1.02)
            
            st.markdown(f"<div style='text-align:center; background:#1e293b; padding:30px; border-radius:15px; border:2px solid #febd69;'>"
                        f"<h1 style='color:#febd69; font-size:65px; margin:0;'>₹{final_p:,}</h1></div>", unsafe_allow_html=True)
            
            st.markdown("---")
            df['diff'] = abs(df['Price'] - raw_p)
            sugg = df.sort_values('diff').head(4)
            cols = st.columns(4)
            img_list = ["https://images.unsplash.com/photo-1517336714731-489689fd1ca8?w=400", "https://images.unsplash.com/photo-1588872657578-7efd1f1555ed?w=400", "https://images.unsplash.com/photo-1593642632823-8f785ba67e45?w=400", "https://images.unsplash.com/photo-1611078489935-0cb964de46d6?w=400"]
            for i, (idx, row) in enumerate(sugg.iterrows()):
                with cols[i]:
                    st.markdown(f'<div class="card"><img src="{img_list[i]}"><p class="laptop-name">{row["Company"]}</p><p class="price-tag">₹{int(row["Price"]):,}</p></div>', unsafe_allow_html=True)
        else:
            st.info("👈 Set your configuration and click Predict!")

    # --- FINAL WORKING FOOTER ---
    st.markdown("---")
    f_col1, f_col2 = st.columns(2)
    
    with f_col1:
        st.markdown('<p class="footer-heading">👨‍💻 About the Developer</p>', unsafe_allow_html=True)
        st.markdown("""<p class="footer-text">
            <b>Pranjal Gupta</b> is a visionary Data Scientist and AI Developer. 
            He has a specialized knack for building high-performance Machine Learning models that solve real-world problems. 
            With expertise in Python, Data Analytics, and UI/UX design, Pranjal ensures every project is not just functional, but a premium experience.
            <br><br>📫 <b>Email:</b> pranjal12345786@gmail.com
        </p>""", unsafe_allow_html=True)

    with f_col2:
        st.markdown('<p class="footer-heading">📧 Send Feedback</p>', unsafe_allow_html=True)
        with st.form("feedback_form", clear_on_submit=True):
            name = st.text_input("Your Name")
            email = st.text_input("Your Email")
            msg = st.text_area("Your Feedback")
            submit = st.form_submit_button("Send to Pranjal")
            if submit:
                # Direct Email via FormSubmit logic
                st.success("Feedback recorded! (FormSubmit will handle the email)")
                # This invisible bit triggers the email
                form_html = f"""
                <form action="https://formsubmit.co/pranjal12345786@gmail.com" method="POST" style="display:none;">
                    <input type="text" name="name" value="{name}">
                    <input type="email" name="email" value="{email}">
                    <textarea name="message">{msg}</textarea>
                </form>
                """
                st.components.v1.html(form_html + '<script>document.forms[0].submit();</script>', height=0)

    st.markdown('<p style="text-align:center; color:#555;">© 2026 | Developed by Pranjal Gupta</p>', unsafe_allow_html=True)
