# Tahap 1: Ekstraksi ROI Wajah (Micro-Expression Pipeline)

Notebook pre-processing skripsi untuk deteksi wajah dari video menggunakan **YOLOv5m-Face**.

## Struktur Folder

```
python test/
├── micro_expression_pipeline.ipynb   # Notebook utama
├── yolov5m-face.pt                   # Weights model face detection
├── yolov5-face/                      # Repo arsitektur (deepcam-cn/yolov5-face)
├── input_videos/                     # Video input
├── output_cropped_faces/             # Hasil crop wajah per frame
└── requirements.txt                  # Dependensi Python
```

## Cara Menjalankan

1. Install dependensi:
   ```bash
   pip install -r requirements.txt
   ```
2. Letakkan video uji di `input_videos/`
3. Buka `micro_expression_pipeline.ipynb` dan jalankan semua sel dari atas

## Output

- `output_cropped_faces/<nama_video>/frame_XXXX.jpg` — crop wajah 224×224
- `output_cropped_faces/<nama_video>/video_asli_with_bbox.mp4` — visualisasi bounding box
