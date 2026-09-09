from __future__ import annotations

import asyncio
import io
import re
import time
import uuid
import zipfile
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles

from overlay import render_model_videos, status_payload

UPLOAD_DIR = Path(__file__).resolve().parent / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

ALLOWED_SUFFIXES = {".mp4", ".avi", ".webm"}
FILE_ID_RE = re.compile(r"^[0-9a-f]{32}$")
ZIP_ENTRIES = (
    ("r3d18", "3d-cnn_result.mp4"),
    ("vit", "vit_result.mp4"),
    ("gcn", "gcn_result.mp4"),
)

app = FastAPI(title="FER Compare API", version="0.2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["Content-Disposition"],
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
        "file_id": token,
        "processing_time_sec": processing_time_sec,
    }


@app.get("/download/{file_id}")
def download_results(file_id: str) -> StreamingResponse:
    if not FILE_ID_RE.fullmatch(file_id):
        raise HTTPException(status_code=400, detail="Invalid file id.")

    sources = {
        "r3d18": UPLOAD_DIR / f"{file_id}_r3d18.mp4",
        "vit": UPLOAD_DIR / f"{file_id}_vit.mp4",
        "gcn": UPLOAD_DIR / f"{file_id}_gcn.mp4",
    }
    missing = [key for key, path in sources.items() if not path.is_file()]
    if missing:
        raise HTTPException(
            status_code=404,
            detail="Rendered videos not found for this session.",
        )

    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for key, zip_name in ZIP_ENTRIES:
            archive.write(sources[key], arcname=zip_name)
    buffer.seek(0)

    return StreamingResponse(
        buffer,
        media_type="application/zip",
        headers={
            "Content-Disposition": 'attachment; filename="FER_Results.zip"',
        },
    )
