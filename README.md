# Facial Micro-Expression Recognition

Repositori ini merupakan implementasi bertahap sistem **Facial Micro-Expression Recognition** untuk Tugas Akhir. Fokus pengembangan saat ini adalah membangun fondasi ekstraksi representasi spasial wajah melalui deteksi dan alignment wajah berbasis **YOLOv5-Face**, kemudian melakukan transfer learning menggunakan **ResNet-18 pre-trained ImageNet** pada dataset macro-expression FER2013.

Pre-training pada FER2013 digunakan untuk menginisialisasi model agar mampu mempelajari pola spasial dasar ekspresi wajah—seperti perubahan pada mata, alis, hidung, dan mulut—sebelum dikembangkan menjadi sistem spatiotemporal yang menangkap perubahan mikro-ekspresi antar-frame. Implementasi tersedia dalam bentuk Jupyter Notebook agar setiap tahap mudah direproduksi, diamati, dan dijelaskan untuk kebutuhan penelitian.

## Status Pengembangan

- [x] Deteksi wajah menggunakan YOLOv5m-Face
- [x] Pemetaan koordinat deteksi ke resolusi frame asli
- [x] Ekstraksi ROI wajah dengan safe padding dan boundary clamping
- [x] Visualisasi bounding box dan lima facial landmarks
- [x] Face alignment berdasarkan kemiringan kedua mata
- [x] Data preprocessing dan augmentasi FER2013
- [x] Baseline ResNet-18 dengan transfer learning ImageNet
- [x] Penyimpanan checkpoint berdasarkan validation accuracy terbaik
- [ ] Evaluasi lanjutan pada dataset mikro-ekspresi
- [ ] Pemodelan dinamika temporal antar-frame

## Arsitektur dan Pipeline Sistem

Pipeline yang telah diimplementasikan saat ini:

```text
Video / Gambar Wajah
        │
        ▼
YOLOv5-Face
  ├─ Face detection
  ├─ Bounding box
  └─ 5 facial landmarks
        │
        ▼
Face Pre-processing
  ├─ Coordinate rescaling (letterbox → frame asli)
  ├─ Safe padding dan square crop
  ├─ Boundary clamping
  └─ Face alignment berdasarkan posisi mata
        │
        ▼
ROI Wajah 224 × 224
        │
        ▼
ResNet-18 Pre-trained ImageNet
  ├─ Spatial feature extraction
  └─ Klasifikasi 7 emosi FER2013
        │
        ▼
Fondasi Feature Extractor Spasial
        │
        ▼
Tahap Selanjutnya: Pemodelan Spatiotemporal
```

### 1. Deteksi dan Alignment Wajah

Tahap preprocessing menggunakan **YOLOv5m-Face**, yaitu varian YOLOv5 yang dirancang khusus untuk deteksi wajah dan regresi lima titik landmark:

1. mata kiri,
2. mata kanan,
3. hidung,
4. sudut bibir kiri, dan
5. sudut bibir kanan.

Frame video diproses menggunakan letterbox pada resolusi inferensi, kemudian koordinat bounding box dipetakan kembali ke resolusi frame asli dengan memperhitungkan `gain` dan `pad`. ROI wajah diperluas menggunakan padding proporsional, dibentuk menjadi crop persegi, dan dibatasi agar tetap berada di dalam dimensi frame.

Face alignment dilakukan dengan menghitung sudut garis antara kedua mata. Crop wajah kemudian dirotasi hingga posisi mata sejajar secara horizontal. Kanvas rotasi diperluas dan diberi padding hitam agar bagian wajah tidak terpotong.

### 2. Pre-training Spasial dengan ResNet-18

Baseline spasial menggunakan **ResNet-18 pre-trained ImageNet**. Lapisan fully connected terakhir dimodifikasi agar menghasilkan tujuh kelas FER2013:

```text
anger, disgust, fear, happy, neutral, sad, surprised
```

FER2013 digunakan sebagai tahap pre-training karena menyediakan variasi ekspresi makro dalam jumlah besar. Model mempelajari representasi spasial ekspresi wajah terlebih dahulu sebelum bobot atau fitur tersebut ditransfer ke dataset mikro-ekspresi yang memiliki jumlah sampel lebih terbatas.

Konfigurasi baseline:

- **Input:** RGB 3-channel, `224 × 224`
- **Backbone:** ResNet-18
- **Pre-trained weights:** ImageNet-1K V1
- **Loss:** Cross-Entropy Loss
- **Optimizer:** Adam
- **Learning rate awal:** `0.001`
- **Weight decay:** `1e-4`
- **Scheduler:** ReduceLROnPlateau
- **Scheduler mode:** `max`
- **Scheduler factor:** `0.5`
- **Scheduler patience:** `2`
- **Batch size:** `32`
- **Checkpoint:** model dengan validation accuracy tertinggi

## Struktur Direktori

```text
Micro-Expression-Detector/
├── README.md
├── .gitignore
├── .gitattributes                     # Konfigurasi Git LFS untuk weights .pt
└── python test/
    ├── README.md                      # Dokumentasi singkat Tahap 1
    ├── requirements.txt               # Dependensi Python
    ├── micro_expression_pipeline.ipynb
    │                                  # Deteksi, crop, landmark, dan alignment wajah
    ├── fer2013_spatial_pretraining.ipynb
    │                                  # DataLoader FER2013 dan baseline ResNet-18
    ├── yolov5m-face.pt                # Weights YOLOv5-Face (Git LFS)
    ├── yolov5-face/                   # Source code YOLOv5-Face
    ├── input_videos/                  # Video input lokal (diabaikan Git)
    ├── output_cropped_faces/          # Hasil ROI dan video anotasi (diabaikan Git)
    ├── dataset 1/                     # Dataset lokal FER2013/CK+ (diabaikan Git)
    └── resnet18_best_baseline.pth     # Checkpoint training lokal (diabaikan Git)
```

Dataset dan checkpoint training tidak didistribusikan melalui repositori karena ukuran file, lisensi dataset, dan kebutuhan reproduksi eksperimen yang berbeda.

## Prasyarat

Lingkungan yang disarankan:

- Python 3.10 atau lebih baru
- Jupyter Notebook
- PyTorch
- torchvision
- OpenCV
- NumPy `< 2.0` untuk kompatibilitas dengan PyTorch 2.1.x
- SciPy
- pandas
- matplotlib
- Pillow
- tqdm
- PyYAML
- Git LFS

Perangkat komputasi yang didukung:

- **CUDA** untuk GPU NVIDIA
- **MPS** untuk Apple Silicon
- **CPU** sebagai fallback

> Pada pipeline YOLOv5-Face di Apple Silicon, inferensi deteksi dijalankan melalui CPU untuk menghindari pergeseran koordinat numerik yang ditemukan pada backend MPS. Training ResNet-18 tetap dapat menggunakan MPS.

## Instalasi

### 1. Clone repositori

Weights `yolov5m-face.pt` disimpan menggunakan Git LFS. Pastikan Git LFS sudah tersedia.

```bash
git lfs install
git clone https://github.com/stfnghr/Micro-Expression.git
cd Micro-Expression
```

### 2. Buat virtual environment

Menggunakan `venv`:

```bash
python -m venv .venv
source .venv/bin/activate
```

Atau menggunakan Conda:

```bash
conda create -n micro-expression python=3.10
conda activate micro-expression
```

### 3. Instal dependensi

```bash
pip install -r "python test/requirements.txt"
```

### 4. Jalankan Jupyter Notebook

```bash
cd "python test"
jupyter notebook
```

## Persiapan Dataset

Dataset tidak disertakan di dalam repositori. Untuk menjalankan pre-training, letakkan FER2013 dalam struktur folder yang kompatibel dengan `torchvision.datasets.ImageFolder`.

Struktur umum yang direkomendasikan:

```text
dataset/fer2013/
├── train/
│   ├── anger/
│   ├── disgust/
│   ├── fear/
│   ├── happy/
│   ├── neutral/
│   ├── sad/
│   └── surprised/
└── test/
    ├── anger/
    ├── disgust/
    ├── fear/
    ├── happy/
    ├── neutral/
    ├── sad/
    └── surprised/
```

Pada konfigurasi lokal saat pengembangan, dataset berada di:

```text
python test/dataset 1/fer2013/fer2013/Training/
```

Jika folder test terpisah tidak tersedia, notebook melakukan fallback split train/validation sebesar **80:20** menggunakan random seed tetap untuk menjaga reproduktibilitas. Path dataset dapat diubah melalui variabel `DATASET_ROOT`, `TRAIN_DIR`, dan `TEST_DIR` pada cell konfigurasi.

## Panduan Penggunaan

### Tahap 1 — Preprocessing dan Face Alignment

1. Letakkan video pada:

   ```text
   python test/input_videos/
   ```

2. Buka `micro_expression_pipeline.ipynb`.
3. Jalankan seluruh cell secara berurutan.
4. Notebook akan:
   - memuat YOLOv5m-Face,
   - mendeteksi wajah dan lima landmarks,
   - memetakan koordinat ke frame asli,
   - memilih kandidat wajah utama,
   - memberi safe padding,
   - melakukan face alignment,
   - mengubah ukuran ROI menjadi `224 × 224`, dan
   - menyimpan hasil ekstraksi.

Output:

```text
output_cropped_faces/<nama_video>/
├── frame_0000.jpg
├── frame_0001.jpg
├── ...
└── video_asli_with_bbox.mp4
```

### Tahap 2 — Data Pipeline dan Baseline ResNet-18

1. Pastikan dataset FER2013 sudah tersedia.
2. Buka `fer2013_spatial_pretraining.ipynb`.
3. Jalankan cell secara berurutan:
   - konfigurasi path dan hyperparameter,
   - transforms dan augmentasi,
   - pemuatan dataset dengan `ImageFolder`,
   - pembuatan DataLoader,
   - visualisasi sanity check,
   - inisialisasi ResNet-18, dan
   - training serta validasi.
4. Checkpoint terbaik disimpan sebagai:

   ```text
   resnet18_best_baseline.pth
   ```

Augmentasi training yang digunakan:

- resize ke `224 × 224`,
- random horizontal flip (`p=0.5`),
- random rotation (`±10°`),
- konversi grayscale menjadi RGB 3-channel,
- normalisasi mean dan standard deviation ImageNet.

Data validasi tidak menerima augmentasi acak agar evaluasi konsisten.

## Hasil Evaluasi Baseline

Baseline ResNet-18 mencapai:

| Model | Dataset | Pre-training | Validation Accuracy |
|---|---|---|---:|
| ResNet-18 | FER2013 | ImageNet-1K V1 | **±60,94%** |

Hasil tersebut digunakan sebagai **sanity check** bahwa pipeline preprocessing, augmentasi, DataLoader, transfer learning, dan training loop telah bekerja dengan benar. Nilai ini belum merupakan performa akhir sistem mikro-ekspresi karena evaluasi masih dilakukan pada dataset macro-expression FER2013.

## Roadmap

### Tahap berikutnya

- [ ] Mengintegrasikan dataset mikro-ekspresi **CASME II** dan/atau **SAMM**
- [ ] Menstandarkan pembagian data berbasis subjek untuk mencegah subject leakage
- [ ] Mengekstraksi perubahan gerak halus menggunakan **Optical Flow**
- [ ] Membangun representasi onset–apex–offset
- [ ] Melakukan transfer learning dari baseline FER2013 ke data mikro-ekspresi
- [ ] Mengevaluasi ketidakseimbangan kelas dengan weighted loss atau sampling strategy
- [ ] Menambahkan confusion matrix, precision, recall, macro-F1, dan UAR

### Kandidat arsitektur lanjutan

- **3D-CNN** untuk mempelajari fitur spasial dan temporal secara bersamaan
- **Vision Transformer (ViT)** untuk menangkap hubungan global antarwilayah wajah
- **Graph Convolutional Network (GCN)** untuk memodelkan relasi antar-landmark atau facial action regions
- Kombinasi **CNN + Optical Flow** untuk menonjolkan perubahan gerak mikro

## Keterbatasan Saat Ini

- Baseline baru dievaluasi pada ekspresi makro FER2013.
- ResNet-18 saat ini terutama mempelajari informasi spasial satu frame.
- Dinamika temporal mikro-ekspresi belum dimodelkan.
- Performa dapat dipengaruhi ketidakseimbangan kelas FER2013.
- Validasi akhir perlu menggunakan protokol evaluasi berbasis subjek pada dataset mikro-ekspresi.

## Acknowledgements

Implementasi deteksi wajah menggunakan arsitektur dari proyek open-source [deepcam-cn/yolov5-face](https://github.com/deepcam-cn/yolov5-face). Backbone klasifikasi menggunakan ResNet-18 dari `torchvision.models` dengan bobot pre-trained ImageNet.

## Lisensi dan Penggunaan Data

Kode penelitian mengikuti ketentuan lisensi dependensi dan source code pihak ketiga yang disertakan. Dataset FER2013, CK+, CASME II, dan SAMM harus diperoleh melalui sumber resmi serta digunakan sesuai lisensi masing-masing.

---

Repositori ini dikembangkan untuk kepentingan akademis dan penelitian Tugas Akhir mengenai **Facial Micro-Expression Recognition**.
