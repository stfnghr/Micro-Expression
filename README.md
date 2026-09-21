# Facial Micro-Expression Recognition

Proyek skripsi ini membandingkan **tiga arsitektur deep learning** untuk klasifikasi mikro-ekspresi wajah pada dataset **SMIC** (Spontaneous Micro-Expression Corpus), modalitas **HS** (High Speed).

Tujuannya bukan sekadar “melatih model”, melainkan menjawab pertanyaan metodologis:

> Representasi mana yang paling andal untuk mikro-ekspresi — **graf Face Mesh (ST-GCN)**, **volume video 3D (R3D-18)**, atau **peta Optical Flow (ViT)** — jika ketiganya dievaluasi dengan protokol yang **sama**, yaitu **Leave-One-Subject-Out (LOSO)**?

LOSO dipakai agar klip orang yang sama tidak pernah muncul di data latih dan data uji sekaligus (**Subject Leakage**). Split acak 80/20 tidak sah pada SMIC karena satu subjek (misalnya `s3`) dapat memiliki puluhan klip.

Workspace eksperimen yang aktif ada di direktori **`python test/`**. Dashboard komparatif ada di **`web/`**. README root repositori yang lama (YOLOv5-Face + FER2013) **tidak lagi berlaku**.

---

## Daftar Isi

0. [🚀 Panduan Instalasi & Eksekusi dari Nol (Quick Start Guide)](#quick-start)
1. [Ringkasan eksperimen](#1-ringkasan-eksperimen)
2. [Struktur direktori](#2-struktur-direktori)
3. [Persiapan lingkungan notebook (Conda / eksperimen LOSO)](#3-persiapan-lingkungan-notebook-conda--eksperimen-loso)
4. [Mode lokal vs server (`RUN_ON_SERVER`)](#4-mode-lokal-vs-server-run_on_server)
5. [Dataset](#5-dataset)
6. [Protokol evaluasi LOSO](#6-protokol-evaluasi-loso)
7. [Pipeline per model](#7-pipeline-per-model)
8. [Urutan eksekusi notebook](#8-urutan-eksekusi-notebook)
9. [Hasil evaluasi saat ini](#9-hasil-evaluasi-saat-ini)
10. [Catatan teknis penting](#10-catatan-teknis-penting)
11. [Dependensi](#11-dependensi)
12. [Web dashboard: Real-Time FER Compare](#12-web-dashboard-real-time-fer-compare)

---

<a id="quick-start"></a>

## 🚀 Panduan Instalasi & Eksekusi dari Nol (Quick Start Guide)

Panduan ini untuk laptop kosong setelah `git clone`. Salin setiap blok perintah ke terminal, urut dari atas ke bawah. Dua terminal terpisah diperlukan: satu untuk backend, satu untuk frontend.

**Yang perlu terpasang di sistem:**

| Perangkat | Versi yang disarankan | Dipakai untuk |
|---|---|---|
| Git | apa saja | mengunduh repositori |
| Python | **3.12** (jangan 3.9 atau 3.13) | FastAPI + model overlay |
| Node.js | 18 LTS atau lebih baru (`npm` ikut terpasang) | dashboard Next.js |
| FFmpeg | opsional, tapi disarankan | agar keempat video overlay bisa diputar di Chrome |

Cek cepat:

```bash
git --version
python --version    # Windows: py --version
node --version
npm --version
ffmpeg -version     # boleh gagal; dashboard tetap jalan, video overlay mungkin tidak diputar browser
```

### 1️⃣ Kloning repositori

```bash
git clone https://github.com/stfnghr/Micro-Expression.git
cd Micro-Expression
```

Folder kerja setelah ini adalah akar repositori (`Micro-Expression/`). Semua perintah di bawah dijalankan relatif ke folder itu, kecuali disebutkan lain.

### 2️⃣ Setup backend (FastAPI & ML)

Buka **terminal 1**, lalu:

```bash
cd web/backend
python -m venv .venv
```

Aktifkan virtual environment sesuai sistem operasi.

**Mac / Linux:**

```bash
source .venv/bin/activate
```

**Windows (Command Prompt atau PowerShell):**

```bash
.venv\Scripts\activate
```

Setelah prompt menampilkan `(.venv)`, pasang pustaka:

```bash
pip install -r requirements.txt
```

Biarkan terminal 1 tetap terbuka dan environment-nya **tetap aktif**. Perintah `uvicorn` di langkah 5 dijalankan di sini.

> Backend dashboard memakai `web/backend/requirements.txt` (FastAPI, Uvicorn, NumPy, MediaPipe, Pillow, PyTorch, torchvision, timm). OpenCV ikut terpasang lewat MediaPipe.

### 3️⃣ Setup frontend (Next.js)

Buka **terminal 2 yang baru** (jangan menimpa terminal backend), lalu:

```bash
cd web/frontend
npm install
```

Biarkan terminal 2 terbuka. `npm run dev` di langkah 5 dijalankan di sini.

### 4️⃣ Penempatan file eksternal (wajib)

<a id="external-files"></a>

Dataset **SMIC**, **MMEW**, dan file bobot **`.pth` tidak ikut ter-push ke GitHub** karena ukurannya besar (lihat `.gitignore`). Setelah clone, folder itu kosong. Tanpa file-file ini, notebook eksperimen akan error, dan overlay dashboard jatuh ke heuristik cadangan (bukan prediksi model skripsi).

Salin berkas dari salinan lokal / drive eksperimen ke **path persis** berikut (nama folder dan nama file harus sama):

```text
Micro-Expression/
└── python test/
    ├── r3d18/
    │   └── r3d18_best_model.pth          ★ bobot 3D-CNN (dashboard + eksperimen)
    ├── vit/
    │   └── vit_best_model.pth            ★ bobot ViT
    ├── stgcn/
    │   ├── stgcn_best_model.pth          ★ bobot ST-GCN fine-tune SMIC
    │   └── stgcn_mmew_pretrained.pth     ★ bobot pre-train MMEW (dipakai dashboard jika ada)
    └── dataset/
        ├── SMIC_all_cropped/             ★ dataset uji SMIC
        │   └── HS/
        └── MMEW/                         ★ dataset pre-train ST-GCN
            └── Micro_Expression/
```

**Dashboard web** hanya wajib memiliki keempat (atau minimal tiga) file `.pth` di atas. Backend membaca path keras:

- `python test/r3d18/r3d18_best_model.pth`
- `python test/vit/vit_best_model.pth`
- `python test/stgcn/stgcn_mmew_pretrained.pth` (jika ada; jika tidak, `stgcn_best_model.pth`)

**Notebook eksperimen LOSO** wajib memiliki dataset:

```text
python test/dataset/SMIC_all_cropped/HS/{subjek}/micro/{negative|positive|surprise}/{id_klip}/reg_*.bmp
python test/dataset/MMEW/Micro_Expression/{anger|disgust|fear|happiness|others|sadness|surprise}/{Sxx-yy-zzz}/*.jpg
```

Contoh klip SMIC: `python test/dataset/SMIC_all_cropped/HS/s1/micro/negative/s1_ne_01/reg_00001.bmp`.

Cek cepat apakah bobot sudah di tempat (dari akar repositori):

```bash
ls "python test/r3d18/r3d18_best_model.pth"
ls "python test/vit/vit_best_model.pth"
ls "python test/stgcn/stgcn_best_model.pth"
ls "python test/stgcn/stgcn_mmew_pretrained.pth"
```

Di Windows PowerShell, ganti `ls` dengan `dir`.

### 5️⃣ Menjalankan aplikasi (booting)

Pastikan file `.pth` sudah diletakkan (langkah 4).

**Terminal 1 — backend** (folder `web/backend`, venv sudah aktif):

```bash
uvicorn main:app --reload --port 8000 --reload-exclude "uploads/*" --reload-exclude "models/*" --reload-exclude ".venv/*"
```

Tunggu sampai muncul `Uvicorn running on http://127.0.0.1:8000`.

**Terminal 2 — frontend** (folder `web/frontend`):

```bash
npm run dev
```

Tunggu sampai Next.js menampilkan `Local: http://localhost:3000`.

**Browser:** buka [http://localhost:3000](http://localhost:3000). Unggah (drag & drop) video `.mp4`, `.avi`, atau `.webm`. Setelah *inference* selesai, empat pemutar video tersinkronisasi akan tampil; tombol **Download Results (ZIP)** mengunduh ketiga overlay.

Jika frontend tidak menemukan API, set variabel ini sebelum `npm run dev`:

```bash
# Mac / Linux
export NEXT_PUBLIC_API_URL=http://localhost:8000

# Windows PowerShell
$env:NEXT_PUBLIC_API_URL="http://localhost:8000"
```

**Kalau error umum**

| Gejala | Perbaikan |
|---|---|
| `python: command not found` | Mac/Linux: coba `python3 -m venv .venv`. Windows: `py -3.12 -m venv .venv`. |
| MediaPipe / JAX gagal di Python 3.9 | Buat ulang venv dengan **Python 3.12**. |
| Overlay tanpa label model / `GET /health` tidak menyebut `weights` | File `.pth` belum ada di path langkah 4. |
| Video overlay tidak diputar di Chrome | Pasang FFmpeg, lalu unggah ulang (backend men-transcode H.264). |
| Port 8000 atau 3000 sudah dipakai | Tutup proses lama, atau ganti port dan sesuaikan `NEXT_PUBLIC_API_URL`. |

Untuk **melatih ulang model di Jupyter** (bukan dashboard), lanjut ke [bagian 3](#3-persiapan-lingkungan-notebook-conda--eksperimen-loso). Penjelasan arsitektur, protokol LOSO, dan hasil evaluasi ada di bagian 1–12 di bawah.

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

Seluruh kode, checkpoint, dan data eksperimen berada di `python test/`. Notebook **harus dijalankan dari folder miliknya sendiri** karena path dataset bersifat relatif (`../dataset/...`). Dashboard web ada di `web/` dan tidak mengubah notebook.

```text
Micro-Expression-Detector/
├── python test/                        # eksperimen LOSO (lihat pohon di bawah)
└── web/                                # dashboard FER Compare
    ├── frontend/                       # Next.js App Router, TypeScript, Tailwind
    └── backend/                        # FastAPI + overlay OpenCV/MediaPipe/PyTorch
```

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
| `web/` | Dashboard FER Compare (Next.js + FastAPI). Tidak mengubah notebook. |

Folder `dataset/` dan file `*.pth` diabaikan Git (lihat `.gitignore`) karena ukurannya besar. Setelah clone, letakkan dataset dan bobot secara manual sesuai [Quick Start, langkah 4](#external-files).

---

## 3. Persiapan lingkungan notebook (Conda / eksperimen LOSO)

Bagian ini khusus **notebook eksperimen** di `python test/` (LOSO, ekstraksi, training). Untuk **menjalankan dashboard web** setelah clone, ikuti [Quick Start](#quick-start) di atas — environment Conda `microsense` tidak wajib untuk FastAPI/Next.js.

Eksperimen notebook dijalankan pada **Conda environment** bernama `microsense`, dengan kernel Jupyter yang sama. Langkah di bawah ini cukup untuk komputer baru (macOS, Linux, atau Windows + Anaconda).

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

## 4. Mode lokal vs server (`RUN_ON_SERVER`)

Satu saklar mengatur epoch, batch size, `num_workers`, patience scheduler, dan flag augmentasi. **Default = lokal** (`False`) agar notebook aman dijalankan di laptop.

### 4.1 Yang diubah: satu baris, di cell kode pertama

Buka notebook, scroll ke **sel kode pertama** (setelah judul Markdown). Cari blok ini, lalu ubah **hanya** baris `RUN_ON_SERVER`:

```python
# ==========================================
# KONFIGURASI EKSEKUSI (Lokal vs Server)
# ==========================================
RUN_ON_SERVER = False  # Ubah ke True jika dijalankan di server GPU berkapasitas tinggi
```

| Tempat jalan | Nilai |
|---|---|
| Laptop / Mac (Jupyter lokal) | `RUN_ON_SERVER = False` |
| Server GPU (CUDA, kapasitas tinggi) | `RUN_ON_SERVER = True` |

Setelah diubah, **jalankan ulang cell itu**, lalu Run All ke bawah (atau restart kernel → Run All). Sel akan mencetak `Mode eksekusi : LOKAL` atau `SERVER` beserta dictionary `CFG`.

Notebook di `discovery/` **tidak** punya saklar ini (hanya EDA).

### 4.2 Notebook mana yang harus diubah

Toggle ada di **6 notebook**. Untuk eksperimen training di server, yang **wajib** diubah adalah empat notebook training. Dua notebook ekstraksi cukup diubah jika ingin konsisten (mereka hanya mencetak `CFG`, tidak mengubah jumlah frame/landmark).

| Wajib di server? | Path notebook | Cell | Efek `CFG` |
|---|---|---|---|
| **Ya** | `python test/stgcn/stgcn_mmew_pretraining.ipynb` | Cell 1 (setelah import) | `BATCH_SIZE`, `NUM_EPOCHS`, `NUM_WORKERS` |
| **Ya** | `python test/stgcn/stgcn_smic_training.ipynb` | Cell 1 | `BATCH_SIZE`, `NUM_EPOCHS_PER_FOLD`, `NUM_WORKERS`, `patience` |
| **Ya** | `python test/r3d18/r3d18_smic_pipeline.ipynb` | Cell 1 | `BATCH_SIZE`, `NUM_EPOCHS_3D`, `NUM_WORKERS`, `patience` |
| **Ya** | `python test/vit/vit_optical_flow_pipeline.ipynb` | Cell 1 | `BATCH_SIZE`, `NUM_EPOCHS_VIT`, `NUM_WORKERS`, `patience` |
| Opsional | `python test/stgcn/stgcn_mmew_extraction.ipynb` | Cell 1 | Cetak `CFG` saja |
| Opsional | `python test/stgcn/stgcn_smic_extraction.ipynb` | Cell 1 | Cetak `CFG` saja |

Tidak ada file `.py` terpisah: **ubah di dalam notebook**, bukan di `requirements.txt` atau README.

### 4.3 Nilai yang aktif otomatis

| Kunci `CFG` | Lokal (`False`) | Server (`True`) |
|---|---:|---:|
| `epochs` | 2 | 150 |
| `batch_size` | 4 | 32 |
| `use_augmentation` | `False` | `True` |
| `patience` | 2 | 20 |
| `num_workers` | 0 | 4 |

Di laptop, epoch=2 hanya untuk smoke-test (pipeline jalan tanpa menunggu berjam-jam). **Angka skripsi / sidang** harus dijalankan dengan `RUN_ON_SERVER = True` (atau setara: epoch penuh seperti di tabel hasil bagian 9).

`use_augmentation` sudah ter-export ke variabel `USE_AUGMENTATION`. Jika pipeline augmentasi belum terhubung ke DataLoader, flag ini siap dipakai tanpa mengubah saklar lagi.

### 4.4 Urutan praktis di server

1. Aktifkan environment dan buka Jupyter di folder `python test/` (lihat bagian 3).
2. Set `RUN_ON_SERVER = True` di **empat notebook training** pada tabel di atas.
3. Jalankan notebook sesuai [urutan eksekusi](#8-urutan-eksekusi-notebook).
4. Pastikan output cell konfigurasi bertuliskan `Mode eksekusi : SERVER` sebelum loop training dimulai.

---

## 5. Dataset

### 5.1 SMIC (dataset uji skripsi)

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

### 5.2 MMEW (hanya untuk pre-training ST-GCN)

| Properti | Nilai |
|---|---|
| Partisi yang dipakai | **Micro_Expression** (300 sekuens) |
| Partisi yang diabaikan | **Macro_Expression** (still image, tanpa sumbu waktu) |
| Kelas | 7: anger, disgust, fear, happiness, others, sadness, surprise |
| Resolusi tipikal | 231 × 231 (wajah sudah di-crop) |

Akurasi 43% pada pre-training MMEW adalah **akurasi training-set**, bukan metrik skripsi. Yang dilaporkan ke penguji adalah metrik **global LOSO SMIC**.

---

## 6. Protokol evaluasi LOSO

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

## 7. Pipeline per model

### 7.1 ST-GCN (graf Face Mesh)

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

### 7.2 R3D-18 (3D-CNN)

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

### 7.3 ViT (Optical Flow + Vision Transformer)

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

## 8. Urutan eksekusi notebook

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

## 9. Hasil evaluasi saat ini

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

## 10. Catatan teknis penting

- **`T = 16` wajib.** ST-GCN, R3D-18, dan sampling ViT memakai panjang temporal yang sama agar protokol sebanding. `np.linspace(0, L-1, 16)` menjamin onset dan offset ikut terwakili. Jika \(L < 16\), beberapa indeks berulang (padding implisit).
- **Pose Normalization dimatikan** (`USE_POSE_NORMALIZATION = False`). Ablasi pada SMIC menunjukkan Macro F1 turun jika landmark dikurangi koordinat hidung, karena frame sudah `reg_*.bmp`.
- **Transfer MMEW → SMIC:** filter `if not key.startswith('fc.')` lalu `load_state_dict(..., strict=False)`. Jangan memuat `fc.weight` 7 kelas ke kepala 3 kelas.
- **`invert_yaxis()`** pada scatter Face Mesh: koordinat citra `y = 0` di **atas**; Matplotlib default `y = 0` di bawah. Tanpa inversi, wajah tampak terbalik.
- **`NUM_WORKERS = 0`** di DataLoader. Di macOS/Jupyter, worker > 0 sering deadlock.
- **Saklar `RUN_ON_SERVER`:** ubah di **cell 1** notebook training (lihat [bagian 4](#4-mode-lokal-vs-server-run_on_server)). Jangan mencari file konfigurasi terpisah.
- Checkpoint `.pth` dan folder `dataset/` tidak ikut Git. Simpan salinan lokal sebelum menghapus environment.

---

## 11. Dependensi

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

---

## 12. Web dashboard: Real-Time FER Compare

Sebagai pelengkap dari pipeline evaluasi eksperimental, repositori ini menyertakan *dashboard* web komparatif. Modul ini memvisualisasikan hasil deteksi ketiga model (3D-CNN, ViT, ST-GCN) secara *side-by-side* dalam satu pemutar video tersinkronisasi.

**Stack teknologi**

- **Frontend:** Next.js (App Router), TypeScript, Tailwind CSS
- **Backend:** FastAPI, Python 3.12, OpenCV, MediaPipe, PyTorch

**Fitur utama**

- **Synchronized 2×2 video grid:** memutar 4 video (1 input asli + 3 output model) secara bersamaan dengan satu *master control bar* tanpa *desync*.
- **Live model overlay:** menggambar *bounding box*, *facial mesh* 468 titik, dan *optical flow* secara *frame-by-frame* berdasarkan prediksi file `.pth` hasil *training*.
- **Parallel processing:** memproses tiga arsitektur *deep learning* secara bersamaan (*multi-threading*) untuk memangkas waktu *rendering* tanpa membuang *frame*.
- **Live execution timer:** pelacakan waktu pemrosesan (*latency*) secara *real-time*.

Backend memuat bobot dari `python test/` (`r3d18_best_model.pth`, `vit_best_model.pth`, `stgcn_best_model.pth` / `stgcn_mmew_pretrained.pth`). Letakkan file `.pth` sesuai [Quick Start, langkah 4](#external-files) agar overlay memakai checkpoint skripsi, bukan heuristik cadangan.

### Cara menjalankan dashboard

Perintah *copy-paste* lengkap (clone → venv → `npm install` → `uvicorn` → `npm run dev`) ada di [🚀 Quick Start](#quick-start). Ringkasan: dua terminal, backend di `http://127.0.0.1:8000`, frontend di [http://localhost:3000](http://localhost:3000).

Gunakan Python **3.12** untuk venv dashboard (bukan 3.9). MediaPipe Tasks mengunduh BlazeFace / Face Landmarker ke `web/backend/models/` pada *overlay* pertama. Jika `ffmpeg` ada di `PATH`, keluaran di-transcode ke H.264 `yuv420p` agar Chrome dapat memutar keempat klip. Opsional: `NEXT_PUBLIC_API_URL=http://localhost:8000`. `POST /upload` mengembalikan `file_id` dan `processing_time_sec`; `GET /download/{file_id}` mengunduh ketiga overlay sebagai `FER_Results.zip`.
