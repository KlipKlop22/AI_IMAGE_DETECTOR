import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms

# 1. Konfigurasi Direktori & Hyperparameter
DATA_DIR = "dataset_processed/" # Folder utama dataset (harus berisi sub-folder 'real' dan 'ai_generated')
BATCH_SIZE = 16
EPOCHS = 5                      # Jumlah perulangan training (bisa ditambah sesuai kebutuhan)
LEARNING_RATE = 0.001
IMG_SIZE = 224

def main():
    # Cek ketersediaan GPU (CUDA) atau pakai CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Menggunakan perangkat: {device}")

    # 2. Transformasi & Augmentasi Data untuk PyTorch
    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) # Normalisasi standar ImageNet
        ]),
    }

    # 3. Memuat Dataset menggunakan ImageFolder
    # Catatan: Di dalam folder 'dataset_processed', harus ada sub-folder dengan nama kelas, misal: 'real' dan 'ai'
    if not os.path.exists(DATA_DIR):
        print(f"Error: Folder dataset '{DATA_DIR}' belum dibuat!")
        return

    image_datasets = datasets.ImageFolder(os.path.join(DATA_DIR), data_transforms['train'])
    dataloaders = DataLoader(image_datasets, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    
    class_names = image_datasets.classes
    print(f"Kelas yang terdeteksi: {class_names}")

    # 4. Memuat Model Pre-trained (EfficientNet-B0 / ResNet18)
    # Kita menggunakan EfficientNet-B0 karena ringan dan akurat
    print("Memuat model EfficientNet-B0...")
    model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)

    # Modifikasi layer terakhir (Classifier) agar sesuai dengan jumlah kelas kita (biasanya 2: Real vs AI)
    num_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(num_features, len(class_names))

    model = model.to(device)

    # 5. Definisikan Loss Function & Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # 6. Proses Training Loop Sederhana
    print("\nMemulai proses training model AI...")
    model.train()

    for epoch in range(EPOCHS):
        running_loss = 0.0
        correct_preds = 0
        total_preds = 0

        for inputs, labels in dataloaders:
            inputs = inputs.to(device)
            labels = labels.to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            # Backward pass & optimize
            loss.backward()
            optimizer.step()

            # Statistik
            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            correct_preds += (predicted == labels).sum().item()
            total_preds += labels.size(0)

        epoch_loss = running_loss / total_preds
        epoch_acc = correct_preds / total_preds

        print(f"Epoch [{epoch+1}/{EPOCHS}] - Loss: {epoch_loss:.4f} - Akurasi: {epoch_acc * 100:.2f}%")

    # 7. Simpan Model yang Sudah Dilatih
    torch.save(model.state_dict(), "ai_detector_model.pth")
    print("\nTraining selesai! Model berhasil disimpan sebagai 'ai_detector_model.pth'.")

if __name__ == "__main__":
    main()