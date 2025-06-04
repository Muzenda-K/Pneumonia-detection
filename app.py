#!/usr/bin/env python
# coding: utf-8

import streamlit as st
import requests
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os
import base64
import io
import plotly.graph_objects as go

# Page config
st.set_page_config(page_title="Pneumonia X-ray Classifier", layout="wide")

# --- Style ---
st.markdown("""
    <style>
        .block-container {
            padding-top: 1.5rem;
            padding-bottom: 2rem;
            
        }
        .stImage > img {
            border-radius: 12px;
        }
        .stProgress > div > div {
            background-color: #FF4B4B !important;
        }
    </style>
""", unsafe_allow_html=True)

st.title("🩺 Chest X-ray Pneumonia Detector")

# Tabs
tab1, tab2, tab3 = st.tabs(["📊 Dataset Overview", "📈 Training & Metrics", "📷 Predict New X-ray"])

# --- Tab 1: Dataset Overview ---
with tab1:
    st.header("🖼️ Sample Images")

    sample_images_dir = "sample_images/"
    sample_images = [os.path.join(sample_images_dir, f) for f in os.listdir(sample_images_dir)[:4]]
    
    img_cols = st.columns(4)
    for col, img_path in zip(img_cols, sample_images):
        with col:
            img = Image.open(img_path).convert("RGB")
            st.image(img, caption=os.path.basename(img_path), width=200)

    st.markdown("---")
    col1, col2 = st.columns(2, vertical_alignment="center")
    with col1:
        st.markdown('<div style="text-align: center;"><h3>📉 Pixel Intensity Distribution</h3></div>', unsafe_allow_html=True)
        st.markdown('<div style="display: flex; justify-content: center; align-items: center; height: 400px;">'
                    '<img src="data:image/png;base64,{}" style="width:700px; height:auto; max-height:400px;">'
                    '</div>'.format(
                        base64.b64encode(open("data_plots/pixel_intensity.png", "rb").read()).decode()
                    ), unsafe_allow_html=True)

    with col2:
        st.markdown('<div style="text-align: center;"><h3>📦 Class Distribution</h3></div>', unsafe_allow_html=True)
        st.markdown('<div style="display: flex; justify-content: center; align-items: center; height: 400px;">'
                    '<img src="data:image/png;base64,{}" style="width:700px; height:auto; max-height:400px;">'
                    '</div>'.format(
                        base64.b64encode(open("data_plots/train_class_distribution.png", "rb").read()).decode()
                    ), unsafe_allow_html=True)

# --- Tab 2: Training & Evaluation Results ---
with tab2:
    st.header("📉 Training Curves")
    curve_cols = st.columns(2)
    with curve_cols[0]:
        st.image("loss_curve.png", caption="Loss Curve", width=500)
    with curve_cols[1]:
        st.image("accuracy_curve.png", caption="Accuracy Curve", width=500)

    st.markdown("---")
    st.subheader("🧪 Evaluation Metrics")
    metrics_df = pd.read_csv("evaluation_plots/evaluation_plots/key_metrics.csv")
    st.dataframe(metrics_df.style.format({"Value": "{:.4f}"}), height=200)

    st.subheader("📈 ROC & PR Curves")
    metric_cols = st.columns(2)
    with metric_cols[0]:
        st.image("evaluation_plots/evaluation_plots/roc_curve.png", caption="ROC Curve", width=500)
    with metric_cols[1]:
        st.image("evaluation_plots/evaluation_plots/pr_curve.png", caption="Precision-Recall Curve", width=500)

    with st.expander("🔍 Confusion Matrix", expanded=True):
        st.subheader("Confusion Matrix")
        st.markdown('<div style="text-align: center; margin-top: 10px; margin-bottom: 20px;">'
                    '<p>Below is the normalized confusion matrix showing model predictions.</p>'
                    '</div>', unsafe_allow_html=True)
    
        # Display the image responsively
        st.markdown("""
            <div style="display: flex; justify-content: center; align-items: center;">
                <img src="data:image/png;base64,{}" style="max-width: 50%; height: auto;">
            </div>
        """.format(
            base64.b64encode(open("evaluation_plots/evaluation_plots/confusion_matrix.png", "rb").read()).decode()
        ), unsafe_allow_html=True)

    
# --- Tab 3: Upload & Predict ---
# --- Tab 3: Upload & Predict ---
with tab3:
    st.header("📤 Upload X-ray Image")
    uploaded_file = st.file_uploader("Choose an X-ray image", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")

        display_cols = st.columns([1, 2])
        with display_cols[0]:
            st.image(image, caption="Uploaded X-ray", width=250)

        try:
            files = {
                "file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)
            }
            response = requests.post("http://localhost:8000/predict/", files=files)

            if response.status_code == 200:
                result = response.json()

                with display_cols[1]:
                    st.metric(label="🧠 Predicted Class", value=result['pred_class'])
                    st.progress(result["confidence_pneumonia"])
                    st.write(f"**Confidence (Pneumonia):** {result['confidence_pneumonia']:.2%}")

                with st.expander("🔍 View Grad-CAM Overlay", expanded=True):
                    # Decode and display Grad-CAM image
                    cam_img_data = bytes.fromhex(result['image'])
                    cam_img = Image.open(io.BytesIO(cam_img_data))

                    # Center and style the Grad-CAM image
                    st.markdown("""
                        <div style="text-align: center;">
                            <img src="data:image/png;base64,{}" 
                                 style="max-width: 100%; height: auto; border-radius: 8px; box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);">
                        </div>
                    """.format(
                        base64.b64encode(cam_img_data).decode()
                    ), unsafe_allow_html=True)

                    # Add a caption below the image
                    st.caption("Grad-CAM Visualization: Highlighting regions that influenced the prediction.")

            else:
                st.error(f"❌ Server returned status code {response.status_code}")
                st.text("Details:\n" + response.text)

        except requests.exceptions.RequestException as e:
            st.error(f"Request failed: {e}")
        except requests.exceptions.JSONDecodeError:
            st.error("❌ Failed to decode JSON from server response.")
            st.text(f"Raw response:\n{response.text}")

# 📌 Footer shown after all tabs
st.markdown("""
    <div style="text-align: center; margin-top: 50px; color: #888; padding: 20px; border-top: 1px solid #ddd;">
        <p>🩺 Chest X-ray Pneumonia Classifier • Built with Streamlit & FastAPI • 2025</p>
    </div>
""", unsafe_allow_html=True)
