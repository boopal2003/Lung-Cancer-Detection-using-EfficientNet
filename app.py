import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import os
import requests

MODEL_PATH = "lung_model.h5"
FILE_ID = "10MSXkQzSOShPfPllGcMxT90aSTc9db54"  # Replace with your Google Drive file ID

def download_model():
    if not os.path.exists(MODEL_PATH):
        st.info("📥 Downloading model...")
        url = f"https://drive.google.com/uc?export=download&id={10MSXkQzSOShPfPllGcMxT90aSTc9db54}"
        response = requests.get(url)
        with open(MODEL_PATH, "wb") as f:
            f.write(response.content)
        st.success("✅ Model downloaded!")

# Download & load model
download_model()
model = load_model(MODEL_PATH)

class_labels = ['Lung Adenocarcinoma', 'Lung Squamous Cell Carcinoma', 'Lung Benign Tissue']

st.set_page_config(page_title="Lung Cancer Detection", layout="centered")
st.title("🫁 AI-Assisted Lung Cancer Detection")
st.markdown("Upload a **CT scan image** to detect lung cancer type.")

uploaded_file = st.file_uploader("Choose a CT-scan image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    st.image(uploaded_file, caption="Uploaded CT Scan", use_column_width=True)
    img = Image.open(uploaded_file).convert("RGB")
    img = img.resize((512, 512))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    prediction = model.predict(img_array)
    predicted_class = class_labels[np.argmax(prediction)]
    st.markdown(f"### 🧠 Prediction: **{predicted_class}**")
