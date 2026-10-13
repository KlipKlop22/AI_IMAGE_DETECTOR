import os
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image


MODEL_PATH = "ai_detector_model.pth"
TEST_DIR = "test_input/"  
IMG_SIZE = 224
CLASS_NAMES = ['ai_generated', 'real']  

def main():
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model '{MODEL_PATH}' belum ditemukan. Jalankan train.py terlebih dahulu!")
        return

    if not os.path.exists(TEST_DIR):
        print(f"Error: Folder '{TEST_DIR}' belum ada! Buat folder tersebut dan masukkan foto random di dalamnya.")
        return

    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    
    print("Memuat model AI...")
    model = models.efficientnet_b0(weights=None)
    num_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(num_features, len(CLASS_NAMES))
    
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model = model.to(device)
    model.eval()

    
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    
    valid_extensions = ('.jpg', '.jpeg', '.png')
    image_files = [f for f in os.listdir(TEST_DIR) if f.lower().endswith(valid_extensions)]

    if not image_files:
        print(f"Tidak ada file gambar (.jpg, .png) di dalam folder '{TEST_DIR}'.")
        return

    print("\n" + "="*50)
    print(f" HASIL DETEKSI FOTO RANDOM DI FOLDER '{TEST_DIR}' ")
    print("="*50)

    
    with torch.no_grad():
        for img_name in image_files:
            img_path = os.path.join(TEST_DIR, img_name)
            try:
                image = Image.open(img_path).convert('RGB')
                input_tensor = transform(image).unsqueeze(0).to(device)

                outputs = model(input_tensor)
                probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
                confidence, predicted_idx = torch.max(probabilities, 0)

                predicted_class = CLASS_NAMES[predicted_idx.item()]
                conf_percentage = confidence.item() * 100

                print(f" Nama File : {img_name}")
                print(f" Prediksi  : {predicted_class.upper()} ({conf_percentage:.2f}%)")
                print("-" * 50)
            except Exception as e:
                print(f" Gagal memproses {img_name}: {e}")

if __name__ == "__main__":
    main()