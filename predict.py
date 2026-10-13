import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import os


MODEL_PATH = "ai_detector_model.pth"
IMG_SIZE = 224

CLASS_NAMES = ['ai_generated', 'real'] 

def predict_image(image_path):
    if not os.path.exists(image_path):
        print(f"Error: File gambar '{image_path}' tidak ditemukan!")
        return

    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    
    model = models.efficientnet_b0(weights=None)
    num_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(num_features, len(CLASS_NAMES))

    
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model '{MODEL_PATH}' belum ada. Jalankan train.py terlebih dahulu!")
        return
        
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model = model.to(device)
    model.eval() 

    
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    image = Image.open(image_path).convert('RGB')
    input_tensor = transform(image).unsqueeze(0) 
    input_tensor = input_tensor.to(device)

    
    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
        confidence, predicted_idx = torch.max(probabilities, 0)

    predicted_class = CLASS_NAMES[predicted_idx.item()]
    conf_percentage = confidence.item() * 100

    print("\n" + "="*40)
    print(f" HASIL PREDIKSI AI DETECTOR ")
    print("="*40)
    print(f"File Gambar : {image_path}")
    print(f"Prediksi    : {predicted_class.upper()}")
    print(f"Kepercayaan : {conf_percentage:.2f}%")
    print("="*40)

if __name__ == "__main__":
    
    test_image = "dataset_processed/real/real_1.png" 
    
    predict_image(test_image)