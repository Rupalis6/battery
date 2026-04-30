import streamlit as st
import pandas as pd
import numpy as np
import joblib
import time
from tensorflow.keras.models import load_model

# --- Initialization ---
# Load model and scaler[cite: 2, 3]
model = load_model("ann_model.h5", compile=False)
scaler = joblib.load("scaler.pkl")

st.set_page_config(page_title="Battery SOC Monitor", layout="wide")
st.title("🔋 Battery State of Charge (SOC) Dashboard")

# Navigation Sidebar
option = st.sidebar.selectbox("Choose Input Method", ("Manual Prediction", "CSV Simulation"))

# --- Option 1: Manual Entry ---
if option == "Manual Prediction":
    st.header("1️⃣ Manual Entry Mode")
    
    col1, col2 = st.columns(2)
    with col1:
        voltage = st.number_input("Battery Voltage (V)", min_value=0.0, max_value=20.0, value=13.5, step=0.1)
    with col2:
        current = st.number_input("Battery Current (A)", min_value=-50.0, max_value=50.0, value=0.0, step=0.1)

    if st.button("Predict SOC"):
        # Scale and Predict[cite: 2, 3]
        input_scaled = scaler.transform([[current, voltage]])
        prediction = model.predict(input_scaled)[0][0]
        soc = np.clip(prediction, 0, 100)
        
        # Load Detection Logic
        if current > 0.5:
            st.error(f"🔴 LOAD DETECTED: Battery is Discharging ({current} A)")
        elif current < -0.5:
            st.success(f"🟢 CHARGING: Power Source Connected ({abs(current)} A)")
        else:
            st.info("🔵 IDLE: No significant current flow.")

        st.metric(label="Predicted State of Charge", value=f"{soc:.2f}%")
        st.progress(float(soc) / 100)

# --- Option 2: CSV Simulation ---
else:
    st.header("2️⃣ CSV Simulation Mode")
    uploaded_file = st.file_uploader("Upload your Battery Data (CSV)", type=["csv"])

    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        
        if st.button("Start Simulation"):
            # UI Placeholders for real-time updates
            status_box = st.empty()
            metric_row = st.columns(3)
            chart_placeholder = st.empty()
            
            history = []
            
            for i in range(len(df)):
                row = df.iloc[i]
                v = row['battery_voltage']
                c = row['battery_current']
                
                # Predict[cite: 1, 2]
                scaled = scaler.transform([[c, v]])
                p_soc = np.clip(model.predict(scaled)[0][0], 0, 100)
                history.append(p_soc)
                
                # Dynamic Load Detection
                if c > 0.5:
                    status_box.error(f"🔴 LOAD ACTIVE: Discharging @ {c} A")
                elif c < -0.5:
                    status_box.success(f"🟢 CHARGING: Receiving @ {abs(c)} A")
                else:
                    status_box.info("🔵 IDLE")

                # Metrics and Chart
                metric_row[0].metric("Voltage", f"{v:.2f} V")
                metric_row[1].metric("Current", f"{c:.2f} A")
                metric_row[2].metric("SOC %", f"{p_soc:.2f}%")
                chart_placeholder.line_chart(history)
                
                # 10-second delay per your request
                time.sleep(10)