import streamlit as st
import joblib
import numpy as np
import pandas as pd
import base64
import os

# 1. Page Config
st.set_page_config(page_title="Laptop AI Expert | Pranjal Gupta", layout="wide", page_icon="💻")

# 2. Advanced CSS
st.markdown("""
    <style>
    .stApp { background-color: #0f1111; color: white; }
    
    /* Welcome Screen */
    .welcome-wrapper { display: flex; justify-content: center; align-items: center; height: 70vh; text-align: center; }
    .welcome-container { background: linear-gradient(45deg, #232f3e, #000000); padding: 60px; border-radius: 25px; border: 2px solid #febd69; max-width: 800px; }
    
    /* Footer */
    .footer { background-color: #131a22; color: white; padding: 50px 20px; margin-top: 60px; border-top: 4px solid #febd69; }
    .footer h3 { color: #febd69; margin-bottom: 15px; font-size: 22px; }
    .footer p { color: #ccc; font-size: 15px; line-height: 1.6; }
    
    /* Feedback Form Styling */
    .feedback-input { width: 100%; padding: 10px; margin-bottom: 10px; border-radius: 5px; border: none; }
    .feedback-btn { background-color: #febd69; color: black; font-weight: bold; padding: 10px 20px; border: none; border-radius: 5px; cursor: pointer; width: 100%; }
    
    /* Cards */
    .card { background-color: white; padding: 20px; border-radius: 15px; text-align: center; height: 560px; border: 3px solid #f3f3f3; }
    .laptop-name { font-size: 24px !important; font-weight: 900 !important; color: #111; }
    .price-tag { color: #B12704; font-size: 28px !important; font-weight: 900 !important; }
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
    st.markdown('<div class="welcome-wrapper">', unsafe_allow_html=True)
    empty_l, welcome_col, empty_r = st.columns([0.5, 2, 0.5])
    with welcome_col:
        st.markdown("""<div class="welcome-container">
                        <h1 style='color:#febd69; font-size: 60px;'>💻 LAPTOP PRICE AI</h1>
                        <p style='color:white; font-size: 22px;'>Precision Data • Expert Developer • Immersive Audio</p>
                       </div>""", unsafe_allow_html=True)
        if st.button("🚀 ENTER DASHBOARD", use_container_width=True):
            st.session_state.enter = True
            st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)
else:
    # Play Audio
    st.components.v1.html(get_audio_html("background_music.mpeg") or get_audio_html("background_music.mp3"), height=0)

    # Load Data & Model
    @st.cache_resource
    def load_all():
        df = pd.read_csv("https://raw.githubusercontent.com/campusx-official/laptop-price-predictor-regression-project/main/laptop_data.csv")
        model = joblib.load("laptop_price_prediction.pkl")
        enc_cpu = joblib.load("cpu_encoder.pkl")
        enc_gpu = joblib.load("gpu_encoder.pkl")
        return df, model, enc_cpu, enc_gpu

    df, model, encoder_cpu, encoder_gpu = load_all()
    cpu_safe_map = {"Intel Core i9":"Intel Core i7", "Intel Core i7":"Intel Core i7", "Intel Core i5":"Intel Core i5", "Intel Core i3":"Intel Core i3", "AMD Ryzen 9":"AMD Ryzen 7", "AMD Ryzen 7":"AMD Ryzen 7", "AMD Ryzen 5":"AMD Ryzen 5"}

    # --- MAIN UI ---
    col_left, col_right = st.columns([1, 2.5])

    with col_left:
        st.markdown("### 📊 Configuration")
        ram = st.selectbox("RAM (GB)", [4, 8, 16, 32, 64], index=1)
        cpu_choice = st.selectbox("Processor", list(cpu_safe_map.keys()))
        gpu = st.selectbox("Graphics Card", encoder_gpu.classes_)
        predict_btn = st.button("🔥 PREDICT PRICE", use_container_width=True)

    with col_right:
        if predict_btn:
            cpu_val = cpu_safe_map.get(cpu_choice, "Intel Core i7")
            cpu_enc = encoder_cpu.transform([cpu_val])[0]
            gpu_enc = encoder_gpu.transform([gpu])[0]
            raw_pred = model.predict(np.array([[ram, 1.6, cpu_enc, gpu_enc]]))[0]
            final_price = int(raw_pred * 1.02)
            
            st.markdown(f"<div style='text-align:center; background:#1e293b; padding:20px; border-radius:10px; border:2px solid #febd69;'>"
                        f"<h1 style='color:#febd69; font-size:60px; margin:0;'>₹{final_price:,}</h1>"
                        f"<p style='color:white;'>Estimated Market Price</p></div>", unsafe_allow_html=True)

            st.markdown("---")
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
                    st.link_button(f"Amazon Link", f"https://www.amazon.in/s?k={row['Company']}+laptop")
        else:
            st.markdown("<h2 style='color:#febd69;'>Analysis Dashboard</h2>", unsafe_allow_html=True)
            st.markdown("""<div style="background:#1e293b; padding:20px; border-radius:10px;">
                <h4>Welcome!</h4><p>Adjust specifications on the left to see real-time price predictions.</p>
            </div>""", unsafe_allow_html=True)

    # --- FOOTER & FEEDBACK SECTION ---
    st.markdown("---")
    
    # HTML for Footer and Feedback Form (Direct Email to You)
    footer_html = f"""
    <div class="footer">
        <div style="display: flex; flex-wrap: wrap; justify-content: space-around; max-width: 1200px; margin: 0 auto; text-align: left;">
            
            <div style="flex: 1; min-width: 300px; margin: 10px;">
                <h3>👨‍💻 About the Developer</h3>
                <p><b>Pranjal Gupta</b> is a passionate Data Scientist and AI Developer specializing in Machine Learning solutions. 
                With a deep understanding of market analytics, he built this tool to simplify tech-buying decisions. 
                Known for writing clean, efficient code and creating user-centric designs, Pranjal is dedicated to pushing the boundaries of AI.</p>
                <p>📫 <b>Contact:</b> pranjal12345786@gmail.com</p>
            </div>

            <div style="flex: 1; min-width: 300px; margin: 10px;">
                <h3>📧 Send Feedback</h3>
                <form action="https://formsubmit.co/pranjal12345786@gmail.com" method="POST">
                    <input type="text" name="name" placeholder="Your Name" class="feedback-input" required>
                    <input type="email" name="email" placeholder="Your Email" class="feedback-input" required>
                    <textarea name="message" placeholder="Your Feedback/Query" class="feedback-input" rows="3" required></textarea>
                    <button type="submit" class="feedback-btn">Send to Pranjal</button>
                    <input type="hidden" name="_next" value="https://laptoppriceprediction-6utiwg3vbdci8tqtz6anry.streamlit.app/">
                    <input type="hidden" name="_subject" value="New Laptop AI Feedback!">
                </form>
            </div>

        </div>
        <hr style="border-color: #334155; margin: 30px 0;">
        <p style="text-align: center;">© 2026 | Built with ❤️ by Pranjal Gupta | Powered by Streamlit</p>
    </div>
    """
    st.markdown(footer_html, unsafe_allow_html=True)
