import os
import cv2
import numpy as np

# Konfigurasi Direktori
VIDEO_DIR = "dataset_video/"      # Folder tempat video mentah (mp4, mkv, dll)
EXTRACTED_DIR = "frames_mentah/"  # Folder hasil potongan frame video
OUTPUT_DIR = "dataset_processed/" # Folder hasil preprocessing & augmentasi
IMG_SIZE = 224                    # Ukuran target (224x224 piksel)

# Buat direktori jika belum ada
os.makedirs(EXTRACTED_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------------------------------------------------------
# FUNGSI 1: EKSTRAKSI FRAME DARI VIDEO
# ---------------------------------------------------------
def extract_frames_from_video(video_filename, frame_interval=30):
    video_path = os.path.join(VIDEO_DIR, video_filename)
    if not os.path.exists(video_path):
        print(f"Video {video_filename} tidak ditemukan di {VIDEO_DIR}")
        return

    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    saved_count = 0

    print("mengekstrak frame dari {video_filename}...")
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Ambil frame setiap kelipatan 'frame_interval' (misal tiap 30 frame sekali)
        if frame_count % frame_interval == 0:
            frame_name = f"{os.path.splitext(video_filename)[0]}_frame_{saved_count}.jpg"
            save_path = os.path.join(EXTRACTED_DIR, frame_name)
            cv2.imwrite(save_path, frame)
            saved_count += 1

        frame_count += 1
    
    cap.release()
    print(f"Berhasil mengekstrak {saved_count} frame dari {video_filename}.")

# ---------------------------------------------------------
# FUNGSI 2: PREPROCESSING & AUGMENTASI GAMBAR
# ---------------------------------------------------------
def preprocess_and_augment():
    print("\nMemulai proses preprocessing dan augmentasi gambar...")
    if not os.path.exists(EXTRACTED_DIR):
        print(f"Folder {EXTRACTED_DIR} kosong.")
        return

    count = 0
    for filename in os.listdir(EXTRACTED_DIR):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(EXTRACTED_DIR, filename)
            img = cv2.imread(img_path)

            if img is None:
                continue

            # A. Standard Preprocessing: Resize & Normalisasi
            img_resized = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            
            # Simpan versi standar asli yang sudah di-resize
            base_output_path = os.path.join(OUTPUT_DIR, filename)
            cv2.imwrite(base_output_path, img_resized)

            # B. Augmentasi Sederhana: Flip Horizontal (Pencerminan)
            img_flipped = cv2.flip(img_resized, 1) # 1 artinya horizontal flip
            flip_filename = f"aug_flip_{filename}"
            cv2.imwrite(os.path.join(OUTPUT_DIR, flip_filename), img_flipped)

            count += 2 # Karena 1 gambar asli menghasilkan 2 (asli + augmentasi)

    print(f"Selesai! Total {count} gambar (termasuk augmentasi) siap digunakan di '{OUTPUT_DIR}'.")

# ---------------------------------------------------------
# EKSEKUSI UTAMA PROGRAM
# ---------------------------------------------------------
if __name__ == "__main__":
    # Contoh penggunaan:
    # 1. Pastikan Anda punya folder 'dataset_video' dan masukkan 1 file video (misal: 'sample.mp4')
    # extract_frames_from_video("sample.mp4", frame_interval=30)
    
    # 2. Menjalankan preprocessing dan augmentasi pada frame yang ada
    preprocess_and_augment()