import streamlit as st
import joblib
import numpy as np
import pandas as pd
import requests
import os
from streamlit_lottie import st_lottie

# 1. Page Config
st.set_page_config(page_title="2026 Laptop AI", layout="wide", page_icon="💻")

# 2. Custom CSS
st.markdown("""
    <style>
    .card {
        background-color: white; padding: 20px; border-radius: 12px;
        border: 1px solid #ddd; text-align: center; height: 550px;
        transition: 0.3s; position: relative;
    }
    .brand-tag {
        position: absolute; top: 10px; left: 10px;
        background: #232f3e; color: white; padding: 4px 12px;
        border-radius: 5px; font-size: 11px; font-weight: bold;
    }
    .price-tag { color: #B12704; font-size: 26px; font-weight: bold; }
    </style>
    """, unsafe_allow_html=True)

# 3. Load Resources
@st.cache_resource
def load_all():
    df = pd.read_csv("https://raw.githubusercontent.com/campusx-official/laptop-price-predictor-regression-project/main/laptop_data.csv")
    model = joblib.load("laptop_price_prediction.pkl")
    enc_cpu = joblib.load("cpu_encoder.pkl")
    enc_gpu = joblib.load("gpu_encoder.pkl")
    return df, model, enc_cpu, enc_gpu

df, model, encoder_cpu, encoder_gpu = load_all()

# --- ACCURACY FIX: Encoder Overriding ---
# Hum encoder mein naye processors ko add kar rahe hain taaki crash na ho
new_labels = ["Intel Core i9", "AMD Ryzen 9", "Intel Core Ultra 7", "Intel Core Ultra 5"]
encoder_cpu.classes_ = np.unique(np.concatenate((encoder_cpu.classes_, new_labels)))

# 4. Processor Mapping (Full Names)
cpu_display_map = {
    "Intel Core i9": "Intel Core i9-14980HX (Extreme Gaming)",
    "Intel Core i7": "Intel Core i7-13700H (High Performance)",
    "Intel Core i5": "Intel Core i5-13500H (Mainstream)",
    "AMD Ryzen 9": "AMD Ryzen 9 8945HS (Professional)",
    "AMD Ryzen 7": "AMD Ryzen 7 7840HS (Creator)",
    "AMD Ryzen 5": "AMD Ryzen 5 5600H (Budget Performance)",
    "Intel Core i3": "Intel Core i3-1215U (Basic)",
    "Other Intel Processor": "Intel Celeron / Pentium",
    "Other AMD Processor": "AMD Ryzen 3 / Athlon"
}

# --- MAIN UI ---
st.title("🚀 Next-Gen Laptop AI Advisor 2026")

c1, c2 = st.columns([1, 2])
with c1:
    st_lottie(requests.get("https://lottie.host/85a1936c-2f96-4191-bc10-097587841c62/An2Bv8K763.json").json(), height=300)

with c2:
    st.subheader("Select Specifications")
    col_a, col_b = st.columns(2)
    with col_a:
        purpose = st.selectbox("Purpose", ["Gaming", "Editing", "Corporate", "Multitasking"])
        ram = st.selectbox("RAM (GB)", [8, 16, 32, 64], index=1)
        refresh = st.selectbox("Refresh Rate", ["60Hz", "120Hz", "144Hz", "240Hz"])
    with col_b:
        cpu_choice = st.selectbox("Processor", list(cpu_display_map.values()))
        cpu_orig = [k for k, v in cpu_display_map.items() if v == cpu_choice][0]
        gpu = st.selectbox("GPU", encoder_gpu.classes_)
        weight = st.slider("Weight (kg)", 1.0, 4.0, 1.8)

# 5. Prediction Logic
if st.button("Predict Accurate Price"):
    # Click Sound
    st.components.v1.html("""<audio autoplay><source src="https://www.soundjay.com/buttons/sounds/button-3.mp3"></audio>""", height=0)
    
    # ML Prediction
    cpu_enc = encoder_cpu.transform([cpu_orig])[0]
    gpu_enc = encoder_gpu.transform([gpu])[0]
    base_pred = model.predict(np.array([[ram, weight, cpu_enc, gpu_enc]]))[0]
    
    # --- REAL-TIME PRICE CALIBRATION ---
    # Naye processors (i9/Ryzen 9) ke liye model ko accurate banane ka logic
    final_price = base_pred
    if cpu_orig == "Intel Core i9" or cpu_orig == "AMD Ryzen 9":
        final_price += 45000  # i9 ki real market value add karna
    
    # Purpose based cost (Gaming/Editing laptops have premium builds)
    if purpose == "Gaming": final_price += 15000
    if purpose == "Editing": final_price += 10000
    
    # Refresh Rate cost
    if "144Hz" in refresh: final_price += 8000
    if "240Hz" in refresh: final_price += 15000

    # 2026 Inflation (12%)
    final_price = int(final_price * 1.12)

    st.balloons()
    st.markdown(f"<h2 style='text-align: center;'>Market Price: ₹{final_price:,}</h2>", unsafe_allow_html=True)
    
    # Suggestions
    st.markdown("---")
    df['diff'] = abs(df['Price'] - final_price)
    matches = df.sort_values('diff').head(4)
    
    cols = st.columns(4)
    for i, (idx, row) in enumerate(matches.iterrows()):
        with cols[i]:
            img = f"https://source.unsplash.com/400x300/?laptop,{row['Company'].lower()}"
            st.markdown(f"""
                <div class="card">
                    <div class="brand-tag">{row['Company']}</div>
                    <img src="{img}">
                    <p><b>{row['Company']} {row['TypeName']}</b></p>
                    <p style='color:gray; font-size:12px;'>{row['Cpu']}</p>
                    <p class="price-tag">₹{int(row['Price'] * 1.12):,}</p>
                </div>
                """, unsafe_allow_html=True)
            st.link_button("Amazon Link", f"https://www.amazon.in/s?k={row['Company']}+laptop")
