import streamlit as st
import joblib
import numpy as np
import pandas as pd
import base64
import os

# 1. Page Configuration
st.set_page_config(page_title="Laptop AI | Pranjal Gupta", layout="wide", page_icon="💻")

# 2. Custom CSS (Premium UI)
st.markdown("""
    <style>
    .stApp { background-color: #0f1111; color: white; }
    
    /* Welcome Container */
    .welcome-container {
        text-align: center; padding: 60px; 
        background: linear-gradient(45deg, #232f3e, #000000);
        border-radius: 25px; border: 2px solid #febd69; 
        margin: 10% auto; max-width: 800px;
    }

    /* Cards */
    .card {
        background-color: white; padding: 20px; border-radius: 15px;
        text-align: center; height: 550px; border: 1px solid #ddd;
    }
    .laptop-name { font-size: 22px !important; font-weight: 900 !important; color: #111; }
    .price-tag { color: #B12704; font-size: 26px !important; font-weight: 900 !important; }

    /* Info Box (The part you wanted back) */
    .info-box {
        background-color: #1e293b; padding: 30px; border-radius: 15px;
        border: 1px solid #334155; line-height: 1.6;
    }
    .info-box h3 { color: #febd69; margin-bottom: 15px; }

    /* Footer */
    .footer-heading { color: #febd69; font-size: 24px; font-weight: bold; margin-top: 20px; }
    .footer-text { color: #ccc; font-size: 16px; line-height: 1.7; }
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

# Session State
if 'enter' not in st.session_state:
    st.session_state.enter = False

# --- SCREEN 1: WELCOME ---
if not st.session_state.enter:
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("""
            <div class="welcome-container">
                <h1 style='color:#febd69; font-size: 60px; margin-bottom:10px;'>💻 LAPTOP AI</h1>
                <p style='font-size: 22px;'>Smart Market Insights by Pranjal Gupta</p>
                <p style='color:#888;'>Immersive Experience • 2026 Predictions</p>
            </div>
        """, unsafe_allow_html=True)
        if st.button("🚀 ENTER DASHBOARD", use_container_width=True):
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

    st.markdown("<h1 style='text-align:center; color:#febd69;'>🚀 Laptop Price Predictor</h1>", unsafe_allow_html=True)
    
    c1, c2 = st.columns([1, 2.5])
    with c1:
        st.markdown("### ⚙️ Configuration")
        ram = st.selectbox("RAM (GB)", [4, 8, 16, 32, 64], index=1)
        cpu_choice = st.selectbox("Processor", list(cpu_safe_map.keys()))
        gpu = st.selectbox("GPU", list(encoder_gpu.classes_))
        predict_click = st.button("🔥 PREDICT PRICE", use_container_width=True)

    with c2:
        if predict_click:
            # Predict Logic
            cpu_val = cpu_safe_map.get(cpu_choice, "Intel Core i7")
            cpu_enc = encoder_cpu.transform([cpu_val])[0]
            gpu_enc = encoder_gpu.transform([gpu])[0]
            raw_p = model.predict(np.array([[ram, 1.6, cpu_enc, gpu_enc]]))[0]
            final_p = int(raw_p * 1.02)
            
            st.markdown(f"<div style='text-align:center; background:#1e293b; padding:30px; border-radius:15px; border:2px solid #febd69;'>"
                        f"<h1 style='color:#febd69; font-size:65px; margin:0;'>₹{final_p:,}</h1>"
                        f"<p style='color:white;'>Estimated Market Value</p></div>", unsafe_allow_html=True)
            
            st.markdown("---")
            df['diff'] = abs(df['Price'] - raw_p)
            sugg = df.sort_values('diff').head(4)
            cols = st.columns(4)
            img_list = ["https://images.unsplash.com/photo-1517336714731-489689fd1ca8?w=400", "https://images.unsplash.com/photo-1588872657578-7efd1f1555ed?w=400", "https://images.unsplash.com/photo-1593642632823-8f785ba67e45?w=400", "https://images.unsplash.com/photo-1611078489935-0cb964de46d6?w=400"]
            for i, (idx, row) in enumerate(sugg.iterrows()):
                with cols[i]:
                    st.markdown(f'<div class="card"><img src="{img_list[i]}"><p class="laptop-name">{row["Company"]}</p><p class="price-tag">₹{int(row["Price"]):,}</p></div>', unsafe_allow_html=True)
                    st.link_button("Buy Link", f"https://www.amazon.in/s?k={row['Company']}+laptop")
        else:
            # --- MARKET INSIGHT SECTION (Back again!) ---
            st.markdown("""
                <div class="info-box">
                    <h3>🔍 Why use this tool?</h3>
                    <p>Laptop prices fluctuate daily based on demand and hardware generation. 
                    Our AI uses a <b>Random Forest Regression</b> model trained on thousands of data points to ensure you never overpay.</p>
                    <hr style="border-color: #334155;">
                    <h3>📈 2026 Tech Trends</h3>
                    <ul style="color: #ccc;">
                        <li><b>Future-Proofing:</b> 16GB RAM is now the minimum for smooth AI tasks.</li>
                        <li><b>Storage:</b> Always prefer NVMe SSDs over traditional ones.</li>
                        <li><b>Graphics:</b> AI-driven upscaling (DLSS/FSR) is a must for gamers.</li>
                    </ul>
                    <p style='color:#febd69; margin-top:20px;'><b>Adjust the specs on the left and click 'Predict' to start!</b></p>
                </div>
            """, unsafe_allow_html=True)

    # --- FOOTER SECTION ---
    st.markdown("<br><br><hr>", unsafe_allow_html=True)
    f1, f2 = st.columns(2)
    
    with f1:
        st.markdown('<p class="footer-heading">👨‍💻 About the Developer</p>', unsafe_allow_html=True)
        st.markdown("""<p class="footer-text">
            <b>Pranjal Gupta</b> is a high-impact Data Scientist and AI Architect. 
            He specializes in bridging the gap between raw data and premium user experiences. 
            Known for his expertise in Python and Predictive Modeling, Pranjal is committed to building tools that empower users with data-driven clarity.
            <br><br>📫 <b>Contact:</b> pranjal12345786@gmail.com
        </p>""", unsafe_allow_html=True)

    with f2:
        st.markdown('<p class="footer-heading">📧 Working Feedback</p>', unsafe_allow_html=True)
        # Using Streamlit Native Form to avoid HTML code display issues
        with st.form("email_feedback", clear_on_submit=True):
            user_name = st.text_input("Name")
            user_msg = st.text_area("Message")
            send_btn = st.form_submit_button("Send to Pranjal's Gmail")
            
            if send_btn:
                # Triggering FormSubmit via invisible HTML
                email_trigger = f"""
                <form action="https://formsubmit.co/pranjal12345786@gmail.com" method="POST" style="display:none;" id="hidden_form">
                    <input type="text" name="name" value="{user_name}">
                    <input type="text" name="message" value="{user_msg}">
                    <input type="hidden" name="_captcha" value="false">
                </form>
                <script>document.getElementById('hidden_form').submit();</script>
                """
                st.components.v1.html(email_trigger, height=0)
                st.success("Redirecting to secure email service... Please confirm submission.")

    st.markdown('<p style="text-align:center; color:#444; margin-top:30px;">© 2026 | Engineered by Pranjal Gupta</p>', unsafe_allow_html=True)
