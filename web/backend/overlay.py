"""Frame-by-frame overlay renderer for the three FER dashboard panels."""

from __future__ import annotations

import shutil
import subprocess
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from inference import FARNEBACK_PARAMS, NUM_NODES, Prediction, flow_to_bgr, predict_sequences

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import (
    FaceDetector,
    FaceDetectorOptions,
    FaceLandmarker,
    FaceLandmarkerOptions,
    FaceLandmarksConnections,
    RunningMode,
)

CYAN = (255, 210, 60)
MAGENTA = (255, 60, 220)
GREEN = (90, 230, 140)
WHITE = (245, 245, 245)
BANNER_BG = (18, 18, 22)

MODELS_DIR = Path(__file__).resolve().parent / "models"
FACE_DET_MODEL = MODELS_DIR / "blaze_face_short_range.tflite"
FACE_MESH_MODEL = MODELS_DIR / "face_landmarker.task"
FACE_DET_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_detector/"
    "blaze_face_short_range/float16/latest/blaze_face_short_range.tflite"
)
FACE_MESH_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/latest/face_landmarker.task"
)

MESH_CONNECTIONS = tuple(
    (conn.start, conn.end)
    for conn in FaceLandmarksConnections.FACE_LANDMARKS_TESSELATION
    if conn.start < NUM_NODES and conn.end < NUM_NODES
)


def _even(value: int) -> int:
    return value if value % 2 == 0 else value - 1


def _clamp_box(
    x1: int, y1: int, x2: int, y2: int, width: int, height: int
) -> tuple[int, int, int, int]:
    x1 = int(np.clip(x1, 0, width - 1))
    y1 = int(np.clip(y1, 0, height - 1))
    x2 = int(np.clip(x2, x1 + 1, width))
    y2 = int(np.clip(y2, y1 + 1, height))
    return x1, y1, x2, y2


def _expand_box(
    box: tuple[int, int, int, int], width: int, height: int, margin: float = 0.08
) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = box
    bw, bh = x2 - x1, y2 - y1
    dx, dy = int(bw * margin), int(bh * margin)
    return _clamp_box(x1 - dx, y1 - dy, x2 + dx, y2 + dy, width, height)


def _box_from_detection(detection, width: int, height: int) -> tuple[int, int, int, int]:
    box = detection.bounding_box
    x1, y1 = int(box.origin_x), int(box.origin_y)
    x2 = x1 + int(box.width)
    y2 = y1 + int(box.height)
    return _clamp_box(x1, y1, x2, y2, width, height)


def _box_from_landmarks(points: np.ndarray, width: int, height: int) -> tuple[int, int, int, int]:
    xs, ys = points[:, 0], points[:, 1]
    return _clamp_box(int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()), width, height)


def _landmarks_px(face_landmarks, width: int, height: int) -> tuple[np.ndarray, np.ndarray]:
    coords = np.array(
        [[lm.x, lm.y, lm.z] for lm in face_landmarks[:NUM_NODES]],
        dtype=np.float32,
    )
    pixels = np.stack(
        [
            np.clip(coords[:, 0] * width, 0, width - 1),
            np.clip(coords[:, 1] * height, 0, height - 1),
        ],
        axis=1,
    ).astype(np.int32)
    return coords, pixels


def _draw_hud_box(
    frame: np.ndarray,
    box: tuple[int, int, int, int],
    color: tuple[int, int, int],
    thickness: int = 2,
) -> None:
    x1, y1, x2, y2 = box
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 1, cv2.LINE_AA)
    length = max(10, min(x2 - x1, y2 - y1) // 6)
    corners = (
        ((x1, y1), (x1 + length, y1), (x1, y1 + length)),
        ((x2, y1), (x2 - length, y1), (x2, y1 + length)),
        ((x1, y2), (x1 + length, y2), (x1, y2 - length)),
        ((x2, y2), (x2 - length, y2), (x2, y2 - length)),
    )
    for origin, a, b in corners:
        cv2.line(frame, origin, a, color, thickness, cv2.LINE_AA)
        cv2.line(frame, origin, b, color, thickness, cv2.LINE_AA)


def _draw_banner(
    frame: np.ndarray,
    box: tuple[int, int, int, int],
    text: str,
    color: tuple[int, int, int],
) -> None:
    x1, y1, x2, y2 = box
    frame_w = frame.shape[1]
    scale = float(np.clip(frame_w / 1400.0, 0.28, 0.72))
    thickness = 1
    pad_x, pad_y = 8, 6
    max_width = max(40, frame_w - 12)
    (tw, th), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
    while tw + pad_x * 2 > max_width and scale > 0.22:
        scale -= 0.03
        (tw, th), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
    bw, bh = tw + pad_x * 2, th + baseline + pad_y * 2
    bx1 = max(6, min(x1, frame_w - bw - 6))
    by2 = y1 - 8
    by1 = by2 - bh
    if by1 < 6:
        by1 = 6
        by2 = by1 + bh
    bx2 = min(frame_w - 6, bx1 + bw)
    overlay = frame.copy()
    cv2.rectangle(overlay, (bx1, by1), (bx2, by2), BANNER_BG, -1, cv2.LINE_AA)
    cv2.rectangle(overlay, (bx1, by1), (bx2, by2), color, 1, cv2.LINE_AA)
    cv2.addWeighted(overlay, 0.82, frame, 0.18, 0, frame)
    cv2.putText(
        frame,
        text,
        (bx1 + pad_x, by2 - pad_y - baseline // 2),
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        WHITE,
        thickness,
        cv2.LINE_AA,
    )
    cv2.line(frame, (bx1, by2), (bx1 + 18, by2), color, 2, cv2.LINE_AA)


def _draw_mesh(frame: np.ndarray, pixels: np.ndarray) -> None:
    for a, b in MESH_CONNECTIONS:
        pa = tuple(int(v) for v in pixels[a])
        pb = tuple(int(v) for v in pixels[b])
        cv2.line(frame, pa, pb, (70, 170, 110), 1, cv2.LINE_AA)
    for x, y in pixels:
        cv2.circle(frame, (int(x), int(y)), 1, GREEN, -1, cv2.LINE_AA)


def _blend_flow(frame: np.ndarray, flow_bgr: np.ndarray, box: tuple[int, int, int, int]) -> None:
    x1, y1, x2, y2 = box
    roi = frame[y1:y2, x1:x2]
    flow_roi = flow_bgr[y1:y2, x1:x2]
    if roi.size == 0 or flow_roi.size == 0:
        return
    blended = cv2.addWeighted(roi, 0.55, flow_roi, 0.45, 0)
    frame[y1:y2, x1:x2] = blended


def _download_model(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.is_file() and dest.stat().st_size > 1024:
        return
    urllib.request.urlretrieve(url, dest)


def _ensure_task_models() -> None:
    _download_model(FACE_DET_URL, FACE_DET_MODEL)
    _download_model(FACE_MESH_URL, FACE_MESH_MODEL)


def _open_face_models():
    _ensure_task_models()
    detector = FaceDetector.create_from_options(
        FaceDetectorOptions(
            base_options=BaseOptions(model_asset_path=str(FACE_DET_MODEL)),
            running_mode=RunningMode.VIDEO,
            min_detection_confidence=0.5,
        )
    )
    landmarker = FaceLandmarker.create_from_options(
        FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(FACE_MESH_MODEL)),
            running_mode=RunningMode.VIDEO,
            num_faces=1,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
    )
    return detector, landmarker


_HAAR_CASCADE = None


def _haar_box(gray: np.ndarray, width: int, height: int) -> tuple[int, int, int, int] | None:
    global _HAAR_CASCADE
    if _HAAR_CASCADE is None:
        cascade_path = getattr(cv2.data, "haarcascades", "") + "haarcascade_frontalface_default.xml"
        _HAAR_CASCADE = cv2.CascadeClassifier(cascade_path)
    if _HAAR_CASCADE.empty():
        return None
    faces = _HAAR_CASCADE.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(40, 40))
    if len(faces) == 0:
        return None
    x, y, w, h = max(faces, key=lambda item: item[2] * item[3])
    return _clamp_box(int(x), int(y), int(x + w), int(y + h), width, height)


def _open_capture(path: Path) -> cv2.VideoCapture:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {path}")
    return capture


def _video_meta(capture: cv2.VideoCapture) -> tuple[float, int, int]:
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 0)
    if not np.isfinite(fps) or fps < 1:
        fps = 25.0
    width = _even(int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0))
    height = _even(int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0))
    if width < 2 or height < 2:
        raise RuntimeError("Invalid video dimensions.")
    return fps, width, height


def _transcode_h264(source: Path, dest: Path) -> bool:
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        return False
    command = [
        ffmpeg,
        "-y",
        "-i",
        str(source),
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "20",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(dest),
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    return result.returncode == 0 and dest.is_file() and dest.stat().st_size > 0


def _open_writer(path: Path, fps: float, size: tuple[int, int]) -> tuple[cv2.VideoWriter, Path]:
    tmp = path.with_suffix(".tmp.mp4")
    for codec in ("avc1", "mp4v"):
        writer = cv2.VideoWriter(str(tmp), cv2.VideoWriter_fourcc(*codec), fps, size)
        if writer.isOpened():
            return writer, tmp
        writer.release()
    raise RuntimeError("OpenCV could not create an MP4 writer (tried avc1 and mp4v).")


def _finalize_video(writer: cv2.VideoWriter, tmp: Path, dest: Path) -> None:
    writer.release()
    if _transcode_h264(tmp, dest):
        tmp.unlink(missing_ok=True)
        return
    tmp.replace(dest)


def _render_one_overlay(
    input_path: Path,
    dest: Path,
    fps: float,
    width: int,
    height: int,
    kind: str,
    det_boxes: list[tuple[int, int, int, int] | None],
    mesh_boxes: list[tuple[int, int, int, int] | None],
    mesh_pixels: list[np.ndarray | None],
    preds: list[Prediction],
) -> str:
    writer, tmp = _open_writer(dest, fps, (width, height))
    capture = _open_capture(input_path)
    prev_gray = None
    try:
        index = 0
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            frame = frame[:height, :width]
            canvas = frame.copy()
            pred = preds[min(index, len(preds) - 1)]
            det_box = det_boxes[index] if index < len(det_boxes) else None
            mesh_box = mesh_boxes[index] if index < len(mesh_boxes) else None
            pixels = mesh_pixels[index] if index < len(mesh_pixels) else None

            if kind == "r3d18":
                box = det_box or mesh_box
                if box:
                    _draw_hud_box(canvas, box, CYAN)
                    _draw_banner(canvas, box, pred.banner("3D-CNN"), CYAN)
            elif kind == "vit":
                box = det_box or mesh_box
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                if prev_gray is not None:
                    flow = cv2.calcOpticalFlowFarneback(
                        prev_gray, gray, None, **FARNEBACK_PARAMS
                    )
                    if box:
                        _blend_flow(canvas, flow_to_bgr(flow), box)
                prev_gray = gray
                if box:
                    _draw_hud_box(canvas, box, MAGENTA)
                    _draw_banner(canvas, box, pred.banner("ViT"), MAGENTA)
            else:
                box = mesh_box or det_box
                if pixels is not None:
                    _draw_mesh(canvas, pixels)
                if box:
                    _draw_hud_box(canvas, box, GREEN)
                    _draw_banner(canvas, box, pred.banner("ST-GCN"), GREEN)

            writer.write(canvas)
            index += 1
    finally:
        capture.release()
        _finalize_video(writer, tmp, dest)
    return dest.name


def render_model_videos(input_path: Path) -> dict[str, str]:
    """Render 3D-CNN, ViT, and ST-GCN overlay clips next to `input_path`."""
    capture = _open_capture(input_path)
    fps, width, height = _video_meta(capture)

    detector, landmarker = _open_face_models()

    rgb_frames: list[np.ndarray] = []
    gray_frames: list[np.ndarray] = []
    landmarks: list[np.ndarray | None] = []
    det_boxes: list[tuple[int, int, int, int] | None] = []
    mesh_boxes: list[tuple[int, int, int, int] | None] = []
    mesh_pixels: list[np.ndarray | None] = []

    try:
        index = 0
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            frame = frame[:height, :width]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            timestamp_ms = (index + 1) * 33
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            detection_result = detector.detect_for_video(mp_image, timestamp_ms)
            mesh_result = landmarker.detect_for_video(mp_image, timestamp_ms)

            det_box = None
            if detection_result.detections:
                det_box = _expand_box(
                    _box_from_detection(detection_result.detections[0], width, height),
                    width,
                    height,
                )

            lm_norm = None
            lm_px = None
            mesh_box = None
            if mesh_result.face_landmarks:
                lm_norm, lm_px = _landmarks_px(mesh_result.face_landmarks[0], width, height)
                mesh_box = _expand_box(_box_from_landmarks(lm_px, width, height), width, height)

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if det_box is None and mesh_box is None:
                det_box = _haar_box(gray, width, height)

            face_box = det_box or mesh_box
            if face_box:
                x1, y1, x2, y2 = face_box
                crop = cv2.cvtColor(frame[y1:y2, x1:x2], cv2.COLOR_BGR2RGB)
            else:
                crop = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            if crop.size == 0:
                crop = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            crop_rgb = cv2.resize(crop, (112, 112), interpolation=cv2.INTER_AREA)

            rgb_frames.append(crop_rgb)
            gray_frames.append(cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2GRAY))
            landmarks.append(lm_norm)
            det_boxes.append(det_box)
            mesh_boxes.append(mesh_box)
            mesh_pixels.append(lm_px)
            index += 1
    finally:
        capture.release()
        detector.close()
        landmarker.close()

    if not rgb_frames:
        raise RuntimeError("The uploaded file did not contain readable frames.")

    predictions = predict_sequences(rgb_frames, gray_frames, landmarks)

    dests = {
        "r3d18": input_path.with_name(f"{input_path.stem}_r3d18.mp4"),
        "vit": input_path.with_name(f"{input_path.stem}_vit.mp4"),
        "gcn": input_path.with_name(f"{input_path.stem}_gcn.mp4"),
    }

    with ThreadPoolExecutor(max_workers=3) as pool:
        futures = {
            key: pool.submit(
                _render_one_overlay,
                input_path,
                dests[key],
                fps,
                width,
                height,
                key,
                det_boxes,
                mesh_boxes,
                mesh_pixels,
                predictions[key],
            )
            for key in dests
        }
        return {key: future.result() for key, future in futures.items()}


def status_payload() -> dict[str, str]:
    from inference import HUB, WEIGHTS

    HUB.ensure_loaded()
    return {
        "r3d18": "weights" if HUB.r3d is not None else f"missing:{WEIGHTS['r3d18'].name}",
        "vit": "weights" if HUB.vit is not None else f"missing:{WEIGHTS['vit'].name}",
        "gcn": "weights" if HUB.gcn is not None else f"missing:{WEIGHTS['gcn'].name}",
        "r3d18_labels": ",".join(HUB.r3d_labels),
        "vit_labels": ",".join(HUB.vit_labels),
        "gcn_labels": ",".join(HUB.gcn_labels),
    }
