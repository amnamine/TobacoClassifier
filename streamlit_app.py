# ==============================================================================
# 1. FORCE 100% CPU & SILENCE WARNINGS (MUST BE AT THE VERY TOP)
# ==============================================================================
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Completely hides the GPU!
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"  # Silences the oneDNN rounding warning

import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# ==============================================================================
# 2. THE ULTIMATE FIX FOR THE "QUANTIZATION_CONFIG" BUG
# ==============================================================================
class SafeDense(tf.keras.layers.Dense):
    def __init__(self, *args, **kwargs):
        kwargs.pop('quantization_config', None)
        super().__init__(*args, **kwargs)

class SafeBatchNormalization(tf.keras.layers.BatchNormalization):
    def __init__(self, *args, **kwargs):
        kwargs.pop('quantization_config', None)
        super().__init__(*args, **kwargs)

class SafeDropout(tf.keras.layers.Dropout):
    def __init__(self, *args, **kwargs):
        kwargs.pop('quantization_config', None)
        super().__init__(*args, **kwargs)

# ==============================================================================
# 3. CONSTANTS & CONFIGURATION
# ==============================================================================
MODEL_PATH = "efficientnetv2s_tobacco3482.h5"
IMG_SIZE = (384, 384)
CLASS_NAMES = ['ADVE', 'Email', 'Form', 'Letter', 'Memo', 'News', 'Note', 'Report', 'Resume', 'Scientific']

# ==============================================================================
# 4. LOAD MODEL (CACHED SO IT ONLY LOADS ONCE)
# ==============================================================================
@st.cache_resource
def load_ai_model():
    """Loads the model once and keeps it in memory for fast predictions."""
    if not os.path.exists(MODEL_PATH):
        st.error(f"❌ Could not find '{MODEL_PATH}'. Please put it in the same folder as this script.")
        return None
        
    custom_objects = {
        'Dense': SafeDense,
        'BatchNormalization': SafeBatchNormalization,
        'Dropout': SafeDropout
    }
    
    with tf.keras.utils.custom_object_scope(custom_objects):
        model = tf.keras.models.load_model(MODEL_PATH, compile=False)
        
    return model

# ==============================================================================
# 5. STREAMLIT UI BUILDER
# ==============================================================================
# Page Config
st.set_page_config(page_title="Tobacco Document AI", page_icon="📄", layout="centered")

# Header
st.title("📄 Document Image Classifier")
st.markdown("**Powered by EfficientNetV2-S (Local CPU)**")
st.markdown("Upload a vintage tobacco industry document, and the AI will classify it into one of 10 categories.")

# Load the model secretly in the background
model = load_ai_model()

if model is not None:
    st.success("✅ AI Model Loaded Successfully!")

    st.markdown("---")

    # File Uploader
    uploaded_file = st.file_uploader("📂 Choose a document image...", type=["jpg", "jpeg", "png", "tif", "tiff", "bmp"])

    if uploaded_file is not None:
        # Display the uploaded image
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            image = Image.open(uploaded_file).convert("RGB")
            st.image(image, caption="Uploaded Document", use_column_width=True)

        # The Predict Button
        st.markdown("<br>", unsafe_allow_html=True) # Add some spacing
        
        # Center the button
        _, btn_col, _ = st.columns([1, 1, 1])
        with btn_col:
            predict_button = st.button("🔍 Run AI Prediction", use_container_width=True)

        if predict_button:
            with st.spinner("🧠 Analyzing document on CPU..."):
                try:
                    # 1. Resize the image exactly how the AI expects it
                    img_resized = image.resize(IMG_SIZE)
                    
                    # 2. Convert to Array & Float32
                    img_array = tf.keras.utils.img_to_array(img_resized)
                    img_array = tf.cast(img_array, tf.float32)
                    
                    # 3. Add Batch Dimension -> (1, 384, 384, 3)
                    img_batch = tf.expand_dims(img_array, 0)
                    
                    # 4. Predict
                    predictions = model.predict(img_batch, verbose=0)
                    
                    # 5. Extract Results
                    class_idx = np.argmax(predictions[0])
                    confidence = np.max(predictions[0]) * 100
                    predicted_class = CLASS_NAMES[class_idx]
                    
                    # Display Results
                    st.markdown("---")
                    st.subheader("🎯 Prediction Result:")
                    st.markdown(f"### **{predicted_class}**")
                    
                    # Progress bar for confidence
                    st.progress(int(confidence) / 100)
                    st.write(f"**Confidence:** {confidence:.2f}%")
                    
                    if confidence < 50:
                        st.warning("⚠️ The AI has low confidence. The document might be ambiguous or poor quality.")

                except Exception as e:
                    st.error(f"An error occurred during prediction: {e}")
                    
    else:
        st.info("👆 Please upload an image to begin.")