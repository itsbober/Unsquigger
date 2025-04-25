import streamlit as st
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

st.set_page_config(page_title="Graph Trace Tool", layout="wide")

st.title("📈 Measurement Graph Trace Tool")

# --- Upload image ---
uploaded_file = st.file_uploader("Upload Graph Image", type=["png", "jpg", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file)
    image_np = np.array(image)
    st.image(image_np, caption="Uploaded Graph", use_column_width=True)

    # Calibration input
    st.markdown("### Step 1: Axis Calibration")
    with st.expander("Click Calibration Points (Order Matters)"):
        st.markdown("1. Click point at **20 Hz**\n2. Click point at **20 kHz**\n3. Click point at **lowest dB**\n4. Click point at **highest dB**")
        canvas_result = st.image(image_np)
        st.info("Calibration by click input will be added using streamlit-drawable-canvas or frontend JS hook.")

    # Manual calibration (mockup fallback)
    st.markdown("### Or enter calibration manually")
    col1, col2 = st.columns(2)
    with col1:
        x1 = st.number_input("X (pixels) at 20 Hz", value=100)
        x2 = st.number_input("X (pixels) at 20 kHz", value=1000)
    with col2:
        y1 = st.number_input("Y (pixels) at lowest dB (e.g. 40 dB)", value=400)
        y2 = st.number_input("Y (pixels) at highest dB (e.g. 80 dB)", value=100)

    freq_log_scale = lambda px: 10 ** (np.log10(20) + (np.log10(20000 / 20) * ((px - x1) / (x2 - x1))))
    db_scale = lambda py: 80 - (40 * (py - y2) / (y1 - y2))

    st.markdown("### Step 2: Auto-Detect Trace")
    hue_low, hue_high = 40, 90  # Green hue range
    img_hsv = cv2.cvtColor(image_np, cv2.COLOR_RGB2HSV)
    mask = cv2.inRange(img_hsv, (hue_low, 40, 40), (hue_high, 255, 255))
    mask_clean = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))

    # Extract trace
    trace_x = []
    trace_y = []
    for col in range(mask_clean.shape[1]):
        column = mask_clean[:, col]
        y_indices = np.where(column > 0)[0]
        if len(y_indices):
            y_mean = int(np.mean(y_indices))
            trace_x.append(col)
            trace_y.append(y_mean)

    if trace_x:
        freqs = [freq_log_scale(x) for x in trace_x]
        dbs = [db_scale(y) for y in trace_y]

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.semilogx(freqs, dbs, label="Traced Curve", color="green")
        ax.set_xlim(20, 20000)
        ax.set_ylim(30, 90)
        ax.grid(True, which="both", linestyle=":")
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("dB SPL")
        ax.set_title("Extracted Trace from Image")
        ax.legend()
        st.pyplot(fig)

        # Export
        df = pd.DataFrame({"Frequency (Hz)": freqs, "SPL (dB)": dbs})
        csv = df.to_csv(index=False, sep='\t')
        st.download_button("Download Traced Curve", csv, file_name="traced_response.txt", mime="text/tab-separated-values")
    else:
        st.warning("Could not detect a trace with the given parameters. Try adjusting calibration or color filtering.")
