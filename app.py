"""
Pneumonia Detection Web App
Author: Prajwal Mhaske
Description: Streamlit web application for real-time pneumonia detection
             from chest X-ray images using a trained CNN model.
"""

import streamlit as st
import numpy as np
import cv2
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import os

# ─────────────────────────────────────────────
# Page Configuration
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="PneumoScan AI | Prajwal Mhaske",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────
# Custom CSS Styling
# ─────────────────────────────────────────────
st.markdown("""
    <style>
    .main { background-color: #0f1117; color: #ffffff; }
    .stButton>button {
        background: linear-gradient(135deg, #667eea, #764ba2);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.5rem 2rem;
        font-size: 1rem;
        font-weight: bold;
    }
    .result-box {
        padding: 1.5rem;
        border-radius: 15px;
        text-align: center;
        font-size: 1.5rem;
        font-weight: bold;
        margin-top: 1rem;
    }
    .pneumonia { background: linear-gradient(135deg, #ff416c, #ff4b2b); color: white; }
    .normal    { background: linear-gradient(135deg, #11998e, #38ef7d); color: white; }
    .title-text { font-size: 2.5rem; font-weight: 800; color: #667eea; }
    </style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# Helper: Preprocess Image
# ─────────────────────────────────────────────
def preprocess_image(image: Image.Image, target_size=(150, 150)) -> np.ndarray:
    """
    Preprocess a PIL image for model inference.
    - Converts to RGB
    - Resizes to target size
    - Normalizes pixel values to [0, 1]
    - Expands dims to add batch dimension
    """
    img = image.convert("RGB")
    img = img.resize(target_size)
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)  # shape: (1, 150, 150, 3)
    return img_array


# ─────────────────────────────────────────────
# Helper: Generate Grad-CAM Heatmap
# ─────────────────────────────────────────────
def generate_gradcam(model, img_array, last_conv_layer_name="conv2d_2"):
    """
    Generate a Grad-CAM heatmap for model interpretability.
    Highlights the regions of the X-ray the model focused on.

    Args:
        model: Trained Keras model
        img_array: Preprocessed image array (1, H, W, 3)
        last_conv_layer_name: Name of the last convolutional layer

    Returns:
        heatmap: numpy array (H, W) with values in [0, 1]
    """
    # Build a model that outputs the last conv layer + final prediction
    grad_model = tf.keras.models.Model(
        inputs=model.inputs,
        outputs=[model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        loss = predictions[:, 0]

    # Gradients of prediction w.r.t. last conv layer outputs
    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)
    return heatmap.numpy()


def overlay_gradcam(original_img: Image.Image, heatmap: np.ndarray) -> np.ndarray:
    """
    Overlay Grad-CAM heatmap on the original image.

    Args:
        original_img: Original PIL image
        heatmap: Grad-CAM heatmap array

    Returns:
        superimposed_img: BGR numpy array with heatmap overlay
    """
    img = np.array(original_img.convert("RGB").resize((150, 150)))
    heatmap_resized = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap_colored = np.uint8(255 * heatmap_resized)
    heatmap_colored = cv2.applyColorMap(heatmap_colored, cv2.COLORMAP_JET)
    superimposed = cv2.addWeighted(img, 0.6, heatmap_colored, 0.4, 0)
    return superimposed


# ─────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🩺 PneumoScan AI")
    st.markdown("**Author:** Prajwal Mhaske")
    st.markdown("**Model:** Custom CNN + VGG16")
    st.markdown("**Dataset:** Kaggle Chest X-Ray (5,863 images)")
    st.markdown("---")
    st.markdown("### 📊 Model Performance")
    st.metric("CNN Accuracy", "~90%")
    st.metric("VGG16 Accuracy", "~92%")
    st.metric("Recall (Pneumonia)", "~95%")
    st.markdown("---")
    show_gradcam = st.checkbox("🗺️ Show Grad-CAM Heatmap", value=True)
    st.markdown("---")
    st.markdown("### ℹ️ About")
    st.markdown(
        "This AI model analyzes chest X-rays and classifies them "
        "as **Normal** or **Pneumonia** using deep learning."
    )


# ─────────────────────────────────────────────
# Main Page
# ─────────────────────────────────────────────
st.markdown('<p class="title-text">🩺 PneumoScan AI</p>', unsafe_allow_html=True)
st.markdown("### Pneumonia Detection from Chest X-Rays")
st.markdown("Upload a chest X-ray image and get an **instant AI-powered prediction** with visual explanation.")
st.markdown("---")

col1, col2 = st.columns([1, 1])

with col1:
    st.markdown("#### 📤 Upload Chest X-Ray")
    uploaded_file = st.file_uploader(
        "Choose an X-ray image",
        type=["jpg", "jpeg", "png"],
        help="Upload a chest X-ray in JPG or PNG format"
    )

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded X-Ray", use_column_width=True)

with col2:
    if uploaded_file is not None:
        st.markdown("#### 🔍 Analysis Result")

        # Check if model file exists
        model_path = "pneumonia_model.h5"
        if not os.path.exists(model_path):
            st.warning(
                "⚠️ **Model file not found!**\n\n"
                "Please train the model first by running `PNEUMONIA_DETECTION.ipynb` "
                "and save it as `pneumonia_model.h5` in this directory."
            )
            st.info(
                "**Demo Mode:** In a real deployment, the model would predict here. "
                "Train the notebook on the Kaggle dataset to enable live predictions."
            )
        else:
            with st.spinner("🔬 Analyzing X-ray..."):
                model = load_model(model_path)
                img_array = preprocess_image(image)
                prediction = model.predict(img_array)[0][0]

            confidence = prediction if prediction > 0.5 else 1 - prediction
            label = "PNEUMONIA DETECTED" if prediction > 0.5 else "NORMAL"
            css_class = "pneumonia" if prediction > 0.5 else "normal"
            icon = "🔴" if prediction > 0.5 else "🟢"

            st.markdown(
                f'<div class="result-box {css_class}">{icon} {label}<br>'
                f'<small>Confidence: {confidence * 100:.1f}%</small></div>',
                unsafe_allow_html=True
            )

            # Progress bar
            st.markdown("**Pneumonia Probability:**")
            st.progress(float(prediction))

            # Grad-CAM
            if show_gradcam:
                st.markdown("#### 🗺️ Grad-CAM Heatmap")
                st.caption("Red/yellow regions = areas the model focused on")
                try:
                    heatmap = generate_gradcam(model, img_array)
                    overlay = overlay_gradcam(image, heatmap)
                    st.image(overlay, caption="Grad-CAM Visualization", use_column_width=True)
                except Exception as e:
                    st.warning(f"Grad-CAM not available for this model architecture: {e}")

# ─────────────────────────────────────────────
# Footer
# ─────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<center>Built with ❤️ by <b>Prajwal Mhaske</b> | "
    "<a href='https://github.com/prajwalmhaske123-er'>GitHub</a></center>",
    unsafe_allow_html=True
)
