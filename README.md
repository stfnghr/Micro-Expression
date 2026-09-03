# Facial Micro-Expression Recognition

Proyek skripsi ini membandingkan **tiga arsitektur deep learning** untuk klasifikasi mikro-ekspresi wajah pada dataset **SMIC** (Spontaneous Micro-Expression Corpus), modalitas **HS** (High Speed).

Tujuannya bukan sekadar “melatih model”, melainkan menjawab pertanyaan metodologis:

> Representasi mana yang paling andal untuk mikro-ekspresi — **graf Face Mesh (ST-GCN)**, **volume video 3D (R3D-18)**, atau **peta Optical Flow (ViT)** — jika ketiganya dievaluasi dengan protokol yang **sama**, yaitu **Leave-One-Subject-Out (LOSO)**?

LOSO dipakai agar klip orang yang sama tidak pernah muncul di data latih dan data uji sekaligus (**Subject Leakage**). Split acak 80/20 tidak sah pada SMIC karena satu subjek (misalnya `s3`) dapat memiliki puluhan klip.

Workspace eksperimen yang aktif ada di direktori **`python test/`**. README root repositori yang lama (YOLOv5-Face + FER2013) **tidak lagi berlaku**.

---

## Daftar Isi

1. [Ringkasan eksperimen](#1-ringkasan-eksperimen)
2. [Struktur direktori](#2-struktur-direktori)
3. [Persiapan lingkungan (dari nol)](#3-persiapan-lingkungan-dari-nol)
4. [Dataset](#4-dataset)
5. [Protokol evaluasi LOSO](#5-protokol-evaluasi-loso)
6. [Pipeline per model](#6-pipeline-per-model)
7. [Urutan eksekusi notebook](#7-urutan-eksekusi-notebook)
8. [Hasil evaluasi saat ini](#8-hasil-evaluasi-saat-ini)
9. [Catatan teknis penting](#9-catatan-teknis-penting)
10. [Dependensi](#10-dependensi)

---

## 1. Ringkasan eksperimen

| Aspek | Ketentuan skripsi |
|---|---|
| Dataset uji | SMIC, modalitas **HS**, partisi **micro** saja |
| Jumlah sampel | **164 klip** dari **16 subjek** |
| Kelas | 3: **Negative**, **Positive**, **Surprise** |
| Protokol | **LOSO** — 16 fold, satu subjek penuh ditahan per fold |
| Metrik | Accuracy, **Macro F1**, **UAR** (Unweighted Average Recall) |
| Panjang temporal | **T = 16** frame (uniform sampling `np.linspace`) |
| Seed | `42` pada ketiga pipeline |

Tiga jalur representasi:

| Model | Representasi | Tensor input | Pre-training |
|---|---|---|---|
| **ST-GCN** | 468 node MediaPipe Face Mesh | `(N, C, T, V)` = `(8, 3, 16, 468)` | MMEW 7 kelas, lalu transfer ke SMIC |
| **R3D-18** | Klip RGB 16 frame | `(N, C, T, H, W)` = `(4, 3, 16, 112, 112)` | Kinetics-400 |
| **ViT-Base** | Satu gambar Optical Flow Farneback | `(N, C, H, W)` = `(16, 3, 224, 224)` | ImageNet |

Macro F1 dan UAR wajib dilaporkan karena kelas SMIC **tidak seimbang** (Negative 70, Positive 51, Surprise 43 pada HS/micro). Accuracy saja dapat tampak “baik” hanya dengan menebak kelas mayoritas.

---

## 2. Struktur direktori

Seluruh kode, checkpoint, dan data eksperimen berada di `python test/`. Notebook **harus dijalankan dari folder miliknya sendiri** karena path dataset bersifat relatif (`../dataset/...`).

```text
python test/
├── requirements.txt
├── .gitignore
│
├── discovery/                          # EDA — tidak ada training
│   ├── smic_data_discovery.ipynb       # peta SMIC: subjek, kelas, jumlah frame
│   └── mmew_data_discovery.ipynb       # peta MMEW: mikro (sekuens) vs makro (still)
│
├── stgcn/                              # model graf Face Mesh
│   ├── stgcn_mmew_extraction.ipynb     # MMEW JPG → .npy (16, 468, 3)
│   ├── stgcn_mmew_pretraining.ipynb    # pre-train 7 kelas MMEW
│   ├── stgcn_smic_extraction.ipynb     # SMIC BMP → .npy (16, 468, 3)
│   ├── stgcn_smic_training.ipynb       # LOSO SMIC + transfer backbone MMEW
│   ├── stgcn_mmew_pretrained.pth       # bobot pre-train MMEW
│   ├── stgcn_best_model.pth            # checkpoint fold terbaik (demo, bukan metrik skripsi)
│   └── stgcn_mmew_landmarks/           # 300 file .npy MMEW (7 folder emosi)
│
├── r3d18/                              # 3D-CNN pada klip video
│   ├── r3d18_smic_pipeline.ipynb       # dataset 3D + LOSO R3D-18
│   └── r3d18_best_model.pth
│
├── vit/                                # ViT pada gambar Optical Flow
│   ├── vit_optical_flow_pipeline.ipynb # Farneback + ImageFolder + LOSO ViT
│   └── vit_best_model.pth
│
└── dataset/                            # data mentah + hasil ekstraksi bersama
    ├── SMIC_all_cropped/               # frame BMP ter-crop & registrasi
    │   ├── HS/                         # ★ modalitas eksperimen
    │   ├── NIR/
    │   └── VIS/
    ├── MMEW/
    │   ├── Micro_Expression/           # 300 sekuens JPG (dipakai ST-GCN)
    │   └── Macro_Expression/           # still image (tidak dilatih)
    ├── stgcn_smic_landmarks/           # 164 .npy graf SMIC
    │   ├── Negative/
    │   ├── Positive/
    │   └── Surprise/
    └── vit_optical_flow_frames/        # 164 JPG flow untuk ImageFolder
        ├── Negative/
        ├── Positive/
        └── Surprise/
```

### Peran tiap folder

| Folder | Peran |
|---|---|
| `discovery/` | Memahami struktur data **sebelum** model dilatih. Output: DataFrame indeks, bukan `.pth`. |
| `stgcn/` | Satu arsitektur, dua dataset: ekstraksi + pre-train MMEW, lalu LOSO SMIC. |
| `r3d18/` | Pipeline 3D-CNN mandiri. Membaca frame BMP langsung dari `dataset/SMIC_all_cropped/`. |
| `vit/` | Pipeline Optical Flow + ViT mandiri. Menulis JPG ke `dataset/vit_optical_flow_frames/`. |
| `dataset/` | Satu atap untuk data mentah dan fitur hasil ekstraksi yang dipakai lintas notebook. |

Folder `dataset/` dan file `*.pth` diabaikan Git (lihat `.gitignore`) karena ukurannya besar. Setelah clone, dataset harus diletakkan manual ke `python test/dataset/`.

---

## 3. Persiapan lingkungan (dari nol)

Eksperimen ini dijalankan pada **Conda environment** bernama `microsense`, dengan kernel Jupyter yang sama. Langkah di bawah ini cukup untuk komputer baru (macOS, Linux, atau Windows + Anaconda).

### 3.1 Pasang Conda dan buat environment

```bash
conda create -n microsense python=3.11 -y
conda activate microsense
```

Python 3.11 dipilih agar kompatibel dengan PyTorch 2.x, MediaPipe, dan `timm`. Jangan memakai Python 3.13 untuk pipeline ini.

### 3.2 Masuk ke folder eksperimen dan pasang dependensi

```bash
cd "/path/ke/Micro-Expression-Detector/python test"
pip install -r requirements.txt
```

`requirements.txt` memaksa **`numpy>=1.23.5,<2.0`**. Numpy 2.x memecah kompatibilitas dengan PyTorch 2.1.x dan beberapa binary MediaPipe.

Paket kunci yang terpasang:

- `torch` / `torchvision` — model, DataLoader, checkpoint
- `timm` — `vit_base_patch16_224`
- `mediapipe` — Face Mesh 468 titik
- `opencv-python` — baca frame, Farneback, konversi warna
- `scikit-learn` — `LeaveOneGroupOut`, Macro F1, UAR
- `notebook` — menjalankan `.ipynb`

### 3.3 Daftarkan kernel Jupyter

```bash
python -m ipykernel install --user --name microsense --display-name "Python (microsense)"
```

Pada setiap notebook, pilih kernel **`Python (microsense)`** atau **`microsense`**. Jangan menjalankan sel di kernel `base` Anaconda.

### 3.4 Letakkan dataset

Salin data ke lokasi berikut (nama folder harus persis):

```text
python test/dataset/SMIC_all_cropped/
python test/dataset/MMEW/
```

Struktur SMIC yang diharapkan:

```text
SMIC_all_cropped/HS/{subjek}/micro/{negative|positive|surprise}/{id_klip}/reg_*.bmp
```

Contoh: `HS/s1/micro/negative/s1_ne_01/reg_00001.bmp`.

Struktur MMEW yang diharapkan:

```text
MMEW/Micro_Expression/{anger|disgust|fear|happiness|others|sadness|surprise}/{Sxx-yy-zzz}/*.jpg
```

Tanpa dua folder ini, cell pertama setiap notebook akan menaikkan `FileNotFoundError`.

### 3.5 Verifikasi perangkat

Notebook mendeteksi perangkat secara otomatis, urutan prioritas:

1. **CUDA** (NVIDIA)
2. **MPS** (Apple Silicon)
3. **CPU**

Tidak perlu mengubah kode. Pastikan hanya **satu notebook training** yang berjalan pada GPU/MPS pada satu waktu agar tidak kehabisan memori.

### 3.6 Buka Jupyter

```bash
conda activate microsense
cd "/path/ke/Micro-Expression-Detector/python test"
jupyter notebook
```

Kemudian buka notebook dari subfoldernya (`discovery/`, `stgcn/`, `r3d18/`, `vit/`). Jupyter menetapkan working directory ke folder notebook, sehingga path `../dataset/...` tetap benar.

---

## 4. Dataset

### 4.1 SMIC (dataset uji skripsi)

| Properti | Nilai |
|---|---|
| Modalitas yang dipakai | **HS** saja |
| Partisi | **micro** saja (`non_micro` dibuang) |
| Subjek | 16 (`s1` … `s16`) |
| Klip | 164 |
| Kelas | Negative, Positive, Surprise |
| Format lokal | Folder frame `.bmp` hasil crop + registrasi (`reg_*.bmp`) |
| Kode nama klip | `ne` → Negative, `po` → Positive, `sur` → Surprise |

NIR dan VIS **tidak dicampur** ke eksperimen. Keduanya dapat merekam kejadian yang sama (pasangan kamera). Mencampur modalitas tanpa grouping event akan menduplikasi sampel dan merusak evaluasi.

Unit sampel adalah **satu folder klip**, bukan satu file BMP. Menjadikan setiap frame sebagai sampel independen menghancurkan informasi temporal dan menggandakan label.

### 4.2 MMEW (hanya untuk pre-training ST-GCN)

| Properti | Nilai |
|---|---|
| Partisi yang dipakai | **Micro_Expression** (300 sekuens) |
| Partisi yang diabaikan | **Macro_Expression** (still image, tanpa sumbu waktu) |
| Kelas | 7: anger, disgust, fear, happiness, others, sadness, surprise |
| Resolusi tipikal | 231 × 231 (wajah sudah di-crop) |

Akurasi 43% pada pre-training MMEW adalah **akurasi training-set**, bukan metrik skripsi. Yang dilaporkan ke penguji adalah metrik **global LOSO SMIC**.

---

## 5. Protokol evaluasi LOSO

**Leave-One-Subject-Out** diimplementasikan dengan `sklearn.model_selection.LeaveOneGroupOut`.

Untuk setiap fold \(k = 1 \ldots 16\):

1. Seluruh klip milik subjek \(s_k\) menjadi **validasi**.
2. Klip 15 subjek lain menjadi **latihan**.
3. Model, optimizer, dan scheduler **dibuat ulang dari nol** (`create_fresh_model*`). Tanpa reset ini, bobot fold sebelumnya merembes (**weight leakage**) dan LOSO tidak sah.
4. Training berjalan **15 epoch**. Prediksi yang dikumpulkan adalah prediksi **epoch terakhir**, bukan epoch dengan Val F1 terbaik. Memilih epoch terbaik dari data validasi fold itu sendiri menghasilkan skor yang terlalu optimistis.
5. Prediksi 16 fold diakumulasi menjadi 164 pasangan \((y, \hat{y})\).
6. Accuracy, Macro F1, dan UAR dihitung **sekali** pada 164 prediksi itu (bukan rata-rata metrik per fold). Rata-rata per fold menyesatkan karena beberapa subjek hanya memiliki 1–2 kelas.

**Subject Leakage** = klip orang yang sama ada di train dan test. Model lalu “mengenali identitas”, bukan ekspresi. LOSO menutup celah itu.

Checkpoint `*_best_model.pth` menyimpan fold dengan akurasi validasi tertinggi **hanya untuk inferensi demo**. Sumber angka skripsi adalah akumulasi LOSO, bukan checkpoint tersebut.

---

## 6. Pipeline per model

### 6.1 ST-GCN (graf Face Mesh)

ST-GCN tidak melihat piksel. Setiap wajah menjadi **graf**: 468 titik MediaPipe Face Mesh sebagai **node**, koneksi `FACEMESH_TESSELATION` sebagai **edge**.

**Alur lengkap:**

1. **Ekstraksi MMEW** (`stgcn_mmew_extraction.ipynb`)
   - Ambil sekuens di `MMEW/Micro_Expression/`.
   - Uniform sampling `np.linspace` menjadi tepat **16 frame**.
   - MediaPipe Face Mesh: `static_image_mode=True`, `max_num_faces=1`, `refine_landmarks=False` (tetap 468 node, bukan 478 iris).
   - Fallback: jika satu frame gagal, salin koordinat frame valid terdekat. Tensor nol tidak disimpan.
   - Simpan `.npy` berbentuk `(T, V, C) = (16, 468, 3)`.

2. **Ekstraksi SMIC** (`stgcn_smic_extraction.ipynb`)
   - Proses identik pada BMP HS/micro.
   - `min_detection_confidence=0.3` karena crop SMIC sangat ketat (default 0.5 terlalu sering gagal).
   - Output: `dataset/stgcn_smic_landmarks/{Negative,Positive,Surprise}/`.

3. **Pre-training MMEW** (`stgcn_mmew_pretraining.ipynb`)
   - `.npy` di-permute `(T, V, C) → (C, T, V)` agar `Conv2d` melihat peta waktu × node.
   - Batch DataLoader: `(N, C, T, V)`.
   - Adjacency dinormalisasi: \(\hat{A} = D^{-1/2}(A+I)D^{-1/2}\). Self-loop \(I\) menjaga fitur node sendiri; \(D^{-1/2}\) mencegah node padat (mulut, mata) mendominasi.
   - 3 blok GraphConv + TemporalConv, kepala `fc` 7 kelas.
   - Adam `lr=1e-3`, `weight_decay=1e-4`, 25 epoch, batch 16.
   - Pose Normalization **dimatikan** (wajah MMEW/SMIC sudah ter-crop/registrasi).
   - Simpan `stgcn_mmew_pretrained.pth`.

4. **Fine-tuning LOSO SMIC** (`stgcn_smic_training.ipynb`)
   - Arsitektur identik, kepala `fc` **3 kelas**.
   - Transfer: buang kunci `fc.*` (ukuran `(7, 128)` vs `(3, 128)`). `strict=False` **saja tidak cukup** — PyTorch tetap error jika kunci yang sama berbeda ukuran.
   - Adam `lr=1e-3`, batch 8, 15 epoch per fold, 16 fold.
   - Metrik global dari 164 prediksi epoch terakhir.

**Notasi tensor (wajib sidang):** file `.npy` = `(T, V, C)`; model = `(C, T, V)`; batch = `(N, C, T, V)` dengan \(N\)=batch, \(C=3\) koordinat \((x,y,z)\), \(T=16\) waktu, \(V=468\) node.

### 6.2 R3D-18 (3D-CNN)

R3D-18 melihat **volume piksel** sepanjang ruang dan waktu. Backbone ResNet-18 3D diinisialisasi dari **Kinetics-400** (aksi manusia), bukan mikro-ekspresi — tetapi filter 3D sudah peka gerak.

**Alur lengkap:**

1. Bangun indeks klip `HS/{subjek}/micro/{emosi}/{klip}`.
2. Urutkan frame **numerik** (`reg_10.bmp` setelah `reg_2.bmp`, bukan urutan alfabet).
3. Uniform sampling 16 frame dengan `np.linspace`.
4. Resize spasial **112 × 112** (bukan 224: volume 5D `C×T×H×W` pada 224 memicu OOM di Apple Silicon).
5. Normalisasi ImageNet mean/std agar distribusi RGB sesuai pre-training Kinetics.
6. `torch.stack` → `(T, C, H, W)`, lalu `permute` → `(C, T, H, W)` sesuai `torchvision.models.video.r3d_18`.
7. Ganti `model.fc` dari 400 kelas Kinetics menjadi **3 kelas SMIC**.
8. Fine-tune Adam **`lr=1e-4`** (10× lebih kecil dari ST-GCN from-scratch agar filter Kinetics tidak rusak), `weight_decay=1e-4`, batch 4, 15 epoch/fold, LOSO.

### 6.3 ViT (Optical Flow + Vision Transformer)

ViT-Base menerima **satu gambar 2D**, bukan video. Gerak temporal dikompresi menjadi warna Optical Flow.

**Alur lengkap:**

1. Pada setiap klip SMIC, ambil **onset** = frame ke-0 dan **apex proksi** = frame tengah (`len // 2`). Anotasi apex resmi tidak tersedia di folder BMP.
2. Hitung Dense Optical Flow **Farneback** pada grayscale resolusi asli (gerak mikro tidak dihaluskan lebih dulu).
3. Ubah vektor \((dx, dy)\) ke HSV: **Hue** = arah, **Saturation** = penuh, **Value** = magnitudo. Lalu konversi ke BGR.
4. Resize **224 × 224** (ukuran native `vit_base_patch16_224`: patch 16×16 → 196 token + 1 CLS).
5. Simpan `{id_klip}_flow.jpg` di `dataset/vit_optical_flow_frames/{kelas}/` agar `ImageFolder` mengikat label ke nama folder.
6. Subject ID diparse dari nama file (`s1_ne_01_flow.jpg` → `s1`). LOSO **tidak** memakai `random_split`.
7. Fine-tune `timm.create_model('vit_base_patch16_224', pretrained=True)`, ganti `model.head` ke 3 kelas.
8. Adam **`lr=1e-4`**, batch 16, 15 epoch/fold, LOSO.

Farneback dipilih (bukan RAFT) karena tersedia di OpenCV tanpa bobot tambahan dan reprodusibel untuk skripsi.

---

## 7. Urutan eksekusi notebook

Jalankan **berurutan**. Jangan meloncat ke training sebelum ekstraksi selesai. Kernel: `microsense`.

### Tahap A — pahami data (wajib, sekali)

| Urutan | Notebook | Folder kerja |
|---:|---|---|
| 1 | `smic_data_discovery.ipynb` | `discovery/` |
| 2 | `mmew_data_discovery.ipynb` | `discovery/` |

Tidak menghasilkan model. Wajib untuk memastikan `subject_id` dan unit sampel benar sebelum LOSO.

### Tahap B — ST-GCN (empat notebook)

| Urutan | Notebook | Keluaran |
|---:|---|---|
| 3 | `stgcn_mmew_extraction.ipynb` | `stgcn/stgcn_mmew_landmarks/*.npy` (300 file) |
| 4 | `stgcn_smic_extraction.ipynb` | `dataset/stgcn_smic_landmarks/*.npy` (164 file) |
| 5 | `stgcn_mmew_pretraining.ipynb` | `stgcn/stgcn_mmew_pretrained.pth` |
| 6 | `stgcn_smic_training.ipynb` | metrik LOSO ST-GCN + `stgcn_best_model.pth` |

Notebook 6 **membutuhkan** file dari langkah 4 dan 5.

### Tahap C — R3D-18 (satu notebook, mandiri)

| Urutan | Notebook | Keluaran |
|---:|---|---|
| 7 | `r3d18_smic_pipeline.ipynb` | metrik LOSO R3D-18 + `r3d18_best_model.pth` |

Membaca BMP langsung. Tidak bergantung pada landmark atau Optical Flow.

### Tahap D — ViT (satu notebook, dua bagian)

| Urutan | Notebook | Keluaran |
|---:|---|---|
| 8 | `vit_optical_flow_pipeline.ipynb` | JPG flow (164) lalu metrik LOSO ViT + `vit_best_model.pth` |

Bagian atas notebook = ekstraksi Farneback. Bagian bawah = LOSO. Jika JPG flow sudah ada, sel ekstraksi boleh dilewati, tetapi sel ImageFolder/LOSO tetap wajib.

### Ringkasan dependensi

```text
discovery ─────────────────────────────────────────────┐
                                                       │  (pemahaman data)
stgcn_mmew_extraction ──► stgcn_mmew_pretraining ──┐   │
stgcn_smic_extraction ─────────────────────────────┼──► stgcn_smic_training (LOSO)
                                                   │
r3d18_smic_pipeline ───────────────────────────────┼──► LOSO R3D-18
                                                   │
vit_optical_flow_pipeline ─────────────────────────┴──► LOSO ViT
```

Ketiga jalur evaluasi memakai **LOSO yang sama** agar angka di tabel hasil dapat dibandingkan.

---

## 8. Hasil evaluasi saat ini

Angka di bawah ini diambil dari **output notebook terakhir** yang tersimpan di repositori (akumulasi 164 prediksi LOSO, epoch terakhir tiap fold).

| Model | Representasi | Accuracy | Macro F1 | UAR |
|---|---|---:|---:|---:|
| **ViT-Base** | Optical Flow Farneback 224×224 | **43.90%** | **41.68%** | **41.92%** |
| **ST-GCN** | Face Mesh 468 node, transfer MMEW | 38.41% | 34.20% | 35.31% |
| **R3D-18** | Klip RGB 16×112×112, Kinetics-400 | 31.10% | 29.04% | 31.63% |

**Cara membaca tabel**

- **ViT** unggul pada ketiga metrik. Kompresi gerak onset→apex ke satu peta warna ternyata paling informatif pada 164 sampel SMIC.
- **ST-GCN** di angka ini memakai backbone MMEW (7 kelas) dengan kepala `fc` SMIC yang diinisialisasi ulang. Pre-training MMEW sendiri mencapai Accuracy training **43.00%** (loss 1.4458) pada epoch ke-25 — angka itu **bukan** metrik uji SMIC.
- **R3D-18** paling rendah. Volume 3D pada 164 klip + resolusi 112 + batch 4 membuat fine-tune Kinetics sulit menyesuaikan ke gerak mikro yang sangat halus.

Karena kelas tidak seimbang, **Macro F1 dan UAR lebih representatif** daripada Accuracy. Chance-level 3 kelas ≈ 33.3%; R3D-18 berada di sekitar itu.

Hyperparameter ringkas yang menghasilkan tabel di atas:

| | ST-GCN | R3D-18 | ViT |
|---|---|---|---|
| Optimizer | Adam | Adam | Adam |
| Learning rate | \(1 \times 10^{-3}\) | \(1 \times 10^{-4}\) | \(1 \times 10^{-4}\) |
| Weight decay | \(1 \times 10^{-4}\) | \(1 \times 10^{-4}\) | \(1 \times 10^{-4}\) |
| Batch size | 8 | 4 | 16 |
| Epoch / fold | 15 | 15 | 15 |
| Scheduler | ReduceLROnPlateau (mode=max, factor=0.5, patience=3) | sama | sama |

---

## 9. Catatan teknis penting

- **`T = 16` wajib.** ST-GCN, R3D-18, dan sampling ViT memakai panjang temporal yang sama agar protokol sebanding. `np.linspace(0, L-1, 16)` menjamin onset dan offset ikut terwakili. Jika \(L < 16\), beberapa indeks berulang (padding implisit).
- **Pose Normalization dimatikan** (`USE_POSE_NORMALIZATION = False`). Ablasi pada SMIC menunjukkan Macro F1 turun jika landmark dikurangi koordinat hidung, karena frame sudah `reg_*.bmp`.
- **Transfer MMEW → SMIC:** filter `if not key.startswith('fc.')` lalu `load_state_dict(..., strict=False)`. Jangan memuat `fc.weight` 7 kelas ke kepala 3 kelas.
- **`invert_yaxis()`** pada scatter Face Mesh: koordinat citra `y = 0` di **atas**; Matplotlib default `y = 0` di bawah. Tanpa inversi, wajah tampak terbalik.
- **`NUM_WORKERS = 0`** di DataLoader. Di macOS/Jupyter, worker > 0 sering deadlock.
- **Jangan mencampur kernel.** Semua notebook memakai environment `microsense`.
- Checkpoint `.pth` dan folder `dataset/` tidak ikut Git. Simpan salinan lokal sebelum menghapus environment.

---

## 10. Dependensi

File: `python test/requirements.txt`.

```text
numpy>=1.23.5,<2.0
scipy>=1.10.0,<1.14.0
opencv-python>=4.8.0
torch>=2.0.0
torchvision>=0.15.0
pandas>=1.5.0
tqdm>=4.64.0
scikit-learn>=1.3.0
PyYAML>=6.0
matplotlib>=3.7.0
Pillow>=9.5.0
notebook>=7.0.0
timm>=0.9.0
mediapipe>=0.10.0
```

Instalasi:

```bash
conda activate microsense
cd "python test"
pip install -r requirements.txt
```
