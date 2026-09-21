# FER Compare dashboard

Dashboard komparatif untuk memvisualisasikan deteksi **3D-CNN**, **ViT**, dan **ST-GCN** *side-by-side* dalam satu pemutar tersinkronisasi. Notebook skripsi tetap di `python test/`. Ringkasan yang sama ada di [README root, bagian 12](../README.md#12-web-dashboard-real-time-fer-compare).

```text
web/
├── frontend/     # Next.js App Router, TypeScript, Tailwind, Lucide
└── backend/      # FastAPI + OpenCV / MediaPipe / PyTorch overlay
```

## Fitur

- **Synchronized 2×2 video grid** — 1 klip asli + 3 overlay model, satu *master control bar*
- **Live model overlay** — bounding box, Face Mesh 468 titik, Optical Flow Farneback per frame
- **Parallel processing** — tiga model dirender bersamaan tanpa membuang frame
- **Live execution timer** — stopwatch saat *processing*, badge `Processed in Xs` setelah selesai

## Cara menjalankan

Langkah *copy-paste* dari laptop kosong (clone, venv, `npm install`, penempatan `.pth`, `uvicorn`, `npm run dev`) ada di [README root → Quick Start](../README.md#quick-start).

Dua terminal terpisah. Backend memuat `.pth` dari `python test/` (lihat [langkah 4 Quick Start](../README.md#external-files)).

**1. Backend (FastAPI, port 8000)**

```bash
cd web/backend
python -m venv .venv
source .venv/bin/activate          # Mac/Linux
# .venv\Scripts\activate           # Windows
pip install -r requirements.txt

uvicorn main:app --reload --port 8000 \
  --reload-exclude 'uploads/*' \
  --reload-exclude 'models/*' \
  --reload-exclude '.venv/*'
```

Pakai Python **3.12** untuk venv ini. MediaPipe Tasks mengunduh BlazeFace / Face Landmarker ke `web/backend/models/` pada overlay pertama.

**2. Frontend (Next.js)**

```bash
cd web/frontend
npm install
npm run dev
```

**3. Buka** [http://localhost:3000](http://localhost:3000). Drag & drop `.mp4` / `.avi` / `.webm`, lalu tunggu inference.

Opsional: `NEXT_PUBLIC_API_URL=http://localhost:8000`.

`POST /upload` menulis klip sumber, merender tiga overlay secara paralel, dan mengembalikan `processing_time_sec`. Overlay:

- **3D-CNN** — Face Detection, HUD cyan, label R3D-18
- **ViT** — kotak magenta + Farneback pada wajah
- **ST-GCN** — mesh 468 titik, kotak hijau

Jika `ffmpeg` ada di `PATH`, keluaran di-transcode ke H.264 (`yuv420p`) agar Chrome dapat memutar keempat klip.
