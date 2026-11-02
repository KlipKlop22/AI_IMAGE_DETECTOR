import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

st.set_page_config(page_title="AI Image Detector", layout="centered")

DEVICE = torch.device("cpu")

# 1. LOAD MODEL DENGAN ARSITEKTUR YANG SAMA
@st.cache_resource
def load_trained_model():
    model = models.efficientnet_b0(weights=None)
    num_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(num_features, 2)
    
    # Load bobot hasil training
    model.load_state_dict(torch.load("ai_detector_model.pth", map_location=DEVICE))
    model.eval() # Mematikan Dropout/BatchNorm untuk evaluasi
    return model

model = load_trained_model()

# URUTAN SESUAI ALFABET FOLDER (Index 0: ai_generated, Index 1: real)
CLASS_NAMES = ['AI GENERATED (Buatan AI)', 'REAL (Asli)']

# 2. TRANSFORMASI HARUS SAMA PERSIS DENGAN TRAIN.PY
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# UI STREAMLIT
st.title("Sistem Deteksi Gambar AI vs Asli")
st.write("Unggah foto di bawah ini untuk mendeteksi apakah gambar tersebut Real (Asli) atau AI Generated (Buatan AI).")

uploaded_file = st.file_uploader("Pilih file gambar (.jpg, .jpeg, .png)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption="Gambar yang Diunggah", use_container_width=True)
    
    if st.button("Deteksi Gambar"):
        # Preprocessing gambar
        img_tensor = transform(image).unsqueeze(0).to(DEVICE)
        
        with torch.no_grad():
            outputs = model(img_tensor)
            # Gunakan Softmax untuk mendapatkan probabilitas 0-100%
            probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]
            confidence, predicted_class = torch.max(probabilities, 0)
            
            label = CLASS_NAMES[predicted_class.item()]
            score = confidence.item() * 100
            
            st.markdown("---")
            st.subheader("Hasil Analisis:")
            
            if predicted_class.item() == 1:
                st.success(f"Prediksi: **{label}** dengan tingkat keyakinan **{score:.2f}%**")
            else:
                st.error(f"Prediksi: **{label}** dengan tingkat keyakinan **{score:.2f}%**")