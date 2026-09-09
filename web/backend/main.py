from __future__ import annotations

import asyncio
import time
import uuid
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from overlay import render_model_videos, status_payload

UPLOAD_DIR = Path(__file__).resolve().parent / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

ALLOWED_SUFFIXES = {".mp4", ".avi", ".webm"}

app = FastAPI(title="FER Compare API", version="0.2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/media", StaticFiles(directory=UPLOAD_DIR), name="media")


@app.get("/health")
def health() -> dict[str, str]:
    payload = {"status": "ok"}
    try:
        payload.update(status_payload())
    except Exception:
        pass
    return payload


@app.post("/upload")
async def upload(file: UploadFile = File(...)) -> dict[str, str | float]:
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in ALLOWED_SUFFIXES:
        raise HTTPException(
            status_code=400,
            detail="Only .mp4, .avi, and .webm files are accepted.",
        )

    token = uuid.uuid4().hex
    dest = UPLOAD_DIR / f"{token}{suffix}"
    payload = await file.read()
    if not payload:
        raise HTTPException(status_code=400, detail="Empty file.")
    dest.write_bytes(payload)

    started = time.perf_counter()
    try:
        # Overlay path: 16-frame sliding window + softmax confidence (see inference.py).
        overlays = await asyncio.to_thread(render_model_videos, dest)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Overlay rendering failed: {exc}") from exc
    processing_time_sec = round(time.perf_counter() - started, 2)

    return {
        "original": f"/media/{dest.name}",
        "r3d18": f"/media/{overlays['r3d18']}",
        "vit": f"/media/{overlays['vit']}",
        "gcn": f"/media/{overlays['gcn']}",
        "filename": file.filename or dest.name,
        "processing_time_sec": processing_time_sec,
    }
