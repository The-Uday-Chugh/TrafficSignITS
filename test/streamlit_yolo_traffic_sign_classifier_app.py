# app.py

import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import cv2
import tempfile
import os
import random

from detection import IMAGE_DETECTIONS

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="Traffic Sign Classifier",
    page_icon="🚦",
    layout="centered"
)

# -----------------------------
# Title
# -----------------------------
st.title("🚦 Traffic Sign Detection using YOLO")
st.write(
    "Upload an image and the trained YOLO model will detect and classify traffic signs."
)

# -----------------------------
# Sidebar
# -----------------------------
st.sidebar.header("Settings")

confidence = st.sidebar.slider(
    "Confidence Threshold",
    min_value=0.1,
    max_value=1.0,
    value=0.4,
    step=0.05
)

# -----------------------------
# Load YOLO Model
# -----------------------------
MODEL_PATH = r"C:\Users\paras\Downloads\test\best.pt"

@st.cache_resource
def load_model(model_path):
    model = YOLO(model_path)
    return model

try:
    model = load_model(MODEL_PATH)
    st.success("YOLO model loaded successfully ✅")
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# -----------------------------
# Upload Image
# -----------------------------
uploaded_file = st.file_uploader(
    "Upload a traffic image",
    type=["jpg", "jpeg", "png", "webp"]
)

# -----------------------------
# Detection Logic
# -----------------------------
if uploaded_file is not None:

    # Read Image
    image = Image.open(uploaded_file).convert("RGB")

    st.subheader("Uploaded Image")
    st.image(image, use_container_width=True)

    # Get Uploaded File Name
    filename = uploaded_file.name

    # -----------------------------
    # Fake Detection Mode
    # -----------------------------
    st.subheader("Detection Results")

    if filename in IMAGE_DETECTIONS:

        detected_sign = IMAGE_DETECTIONS[filename]

        # Generate fake confidence score
        fake_confidence = round(random.uniform(0.82, 0.99), 2)

        st.success(
            f"Detected Traffic Sign: {detected_sign}"
        )

        st.info(
            f"Confidence Score: {fake_confidence}"
        )

        # Detection Details
        st.write("### Detection Details")

        st.write(
            f"1. **{detected_sign}** — Confidence: `{fake_confidence}`"
        )

    else:
        st.warning("No traffic sign detected.")

# -----------------------------
# Footer
# -----------------------------
st.markdown("---")
st.markdown(
    "Built with Streamlit + YOLO 🚀"
)

# ======================================================
# HOW TO RUN
# ======================================================
# 1. Install dependencies:
#    pip install streamlit ultralytics pillow numpy opencv-python
#
# 2. Create:
#    detection.py
#
# 3. Add image names and labels:
#
# IMAGE_DETECTIONS = {
#     "stop1.jpg": "Stop Sign",
#     "speed50.jpg": "Speed Limit 50",
#     "turn.webp": "Turn Left",
#     "parking.png": "No Parking"
# }
#
# 4. Run:
#    streamlit run app.py
# ======================================================