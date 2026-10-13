import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

st.set_page_config(page_title="AI Media Detector", page_icon="🤖", layout="centered")

st.title("AI Media Detector")
st.write("Unggah foto di bawah ini untuk mendeteksi apakah gambar tersebut **Real (Asli)** atau **AI Generated (Buatan AI)**.")

@st.cache_resource
def load_model():
    MODEL_PATH = "ai_detector_model.pth"
    # Sesuai urutan abjad ImageFolder PyTorch: 'ai_generated' (0) dan 'real' (1)
    CLASS_NAMES = ['ai_generated', 'real']
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = models.efficientnet_b0(weights=None)
    num_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(num_features, len(CLASS_NAMES))
    
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model = model.to(device)
    model.eval()
    return model, device, CLASS_NAMES

with st.spinner("Memuat model AI ke sistem..."):
    model, device, CLASS_NAMES = load_model()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

uploaded_file = st.file_uploader("Pilih file gambar (.jpg, .jpeg, .png)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption="Gambar yang Diunggah", use_container_width=True)
    
    if st.button("🔍 Deteksi Gambar"):
        with st.spinner("Sedang menganalisis gambar..."):
            input_tensor = transform(image).unsqueeze(0).to(device)
            
            with torch.no_grad():
                outputs = model(input_tensor)
                probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
                confidence, predicted_idx = torch.max(probabilities, 0)

            predicted_class = CLASS_NAMES[predicted_idx.item()]
            conf_percentage = confidence.item() * 100

            st.markdown("---")
            st.subheader("Hasil Analisis:")
            
            if predicted_class == 'real':
                st.success(f"**Prediksi: REAL (Asli)** dengan tingkat keyakinan **{conf_percentage:.2f}%**")
            else:
                st.error(f"**Prediksi: AI GENERATED (Buatan AI)** dengan tingkat keyakinan **{conf_percentage:.2f}%**")