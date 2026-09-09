"""Sliding-window emotion inference for the dashboard overlays.

SMIC checkpoints have 3 logits; MMEW ST-GCN has 7. Labels are chosen from the
head size so banners never IndexError. Confidence is always softmax, not raw
logits.
"""

from __future__ import annotations

import threading
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np

NUM_FRAMES = 16
INFER_STRIDE = 4
NUM_NODES = 468
NUM_CHANNELS = 3
R3D_SIZE = 112
VIT_SIZE = 224
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# MMEW Micro_Expression order (matches stgcn_mmew_pretrained.pth)
MMEW_7 = ("Angry", "Disgust", "Fear", "Happy", "Others", "Sad", "Surprise")
FER_6 = ("Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise")
# SMIC 3-class (Negative, Positive, Surprise) → nearest specific names
SMIC_3 = ("Angry", "Happy", "Surprise")

REPO_ROOT = Path(__file__).resolve().parents[2]
PYTHON_TEST = REPO_ROOT / "python test"
WEIGHTS = {
    "r3d18": PYTHON_TEST / "r3d18" / "r3d18_best_model.pth",
    "vit": PYTHON_TEST / "vit" / "vit_best_model.pth",
    "gcn": PYTHON_TEST / "stgcn" / "stgcn_best_model.pth",
    "gcn_mmew": PYTHON_TEST / "stgcn" / "stgcn_mmew_pretrained.pth",
}

FARNEBACK_PARAMS = dict(
    pyr_scale=0.5,
    levels=3,
    winsize=15,
    iterations=3,
    poly_n=5,
    poly_sigma=1.2,
    flags=0,
)


@dataclass(frozen=True)
class Prediction:
    label: str
    confidence: float
    source: str

    def banner(self, model_tag: str) -> str:
        pct = int(round(max(0.0, min(1.0, self.confidence)) * 100))
        return f"[{model_tag}] Class: {self.label} ({pct}%)"


def labels_for_num_classes(num_classes: int) -> tuple[str, ...]:
    if num_classes == 7:
        return MMEW_7
    if num_classes == 6:
        return FER_6
    if num_classes == 3:
        return SMIC_3
    if num_classes <= 0:
        return ("Others",)
    if num_classes < 7:
        return MMEW_7[:num_classes]
    extra = tuple(f"Emotion_{i}" for i in range(7, num_classes))
    return MMEW_7 + extra


def _head_num_classes(state: dict, *keys: str) -> int:
    for key in keys:
        tensor = state.get(key)
        if tensor is not None and hasattr(tensor, "shape") and tensor.ndim >= 1:
            return int(tensor.shape[0])
    return 3


def _safe_label(labels: tuple[str, ...], index: int) -> str:
    if not labels:
        return "Others"
    return labels[index] if 0 <= index < len(labels) else labels[-1]


def _imagenet_tensor(image_rgb: np.ndarray, size: int) -> np.ndarray:
    import cv2

    resized = cv2.resize(image_rgb, (size, size), interpolation=cv2.INTER_AREA)
    tensor = resized.astype(np.float32) / 255.0
    tensor = (tensor - IMAGENET_MEAN) / IMAGENET_STD
    return np.transpose(tensor, (2, 0, 1))


def flow_to_bgr(flow: np.ndarray) -> np.ndarray:
    import cv2

    magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    hsv = np.zeros((flow.shape[0], flow.shape[1], 3), dtype=np.uint8)
    hsv[..., 0] = angle * 180 / np.pi / 2
    hsv[..., 1] = 255
    hsv[..., 2] = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def _window_flow(gray_window: list[np.ndarray]) -> tuple[np.ndarray | None, float]:
    import cv2

    if len(gray_window) < 2:
        return None, 0.0
    first, last = gray_window[0], gray_window[-1]
    if first.shape != last.shape:
        last = cv2.resize(last, (first.shape[1], first.shape[0]))
    flow = cv2.calcOpticalFlowFarneback(first, last, None, **FARNEBACK_PARAMS)
    mag = float(np.linalg.norm(flow, axis=2).mean())
    return flow_to_bgr(flow), mag


def _landmark_motion(window: list[np.ndarray | None]) -> float:
    valid = [item for item in window if item is not None]
    if len(valid) < 2:
        return 0.0
    deltas = [
        np.linalg.norm(valid[i][:, :2] - valid[i - 1][:, :2], axis=1).mean()
        for i in range(1, len(valid))
    ]
    return float(np.mean(deltas))


def simulate_window(
    name: str,
    flow_mag: float,
    landmark_motion: float,
    labels: tuple[str, ...],
) -> Prediction:
    n = len(labels)
    logits = np.full(n, 0.15, dtype=np.float32)
    surprise_i = labels.index("Surprise") if "Surprise" in labels else n - 1
    happy_i = labels.index("Happy") if "Happy" in labels else min(1, n - 1)
    angry_i = labels.index("Angry") if "Angry" in labels else 0
    logits[angry_i] += 0.35
    logits[happy_i] += flow_mag * 0.8
    logits[surprise_i] += landmark_motion * 12.0 + flow_mag * 0.25
    if name == "vit":
        logits[happy_i] += 0.12
    elif name == "gcn":
        logits[surprise_i] += 0.08
    shifted = logits - logits.max()
    exp = np.exp(shifted)
    probs = exp / np.clip(exp.sum(), 1e-8, None)
    idx = int(probs.argmax())
    return Prediction(_safe_label(labels, idx), float(probs[idx]), "simulated")


def _build_stgcn(torch, nn):
    class GraphConv(nn.Module):
        def __init__(self, in_channels: int, out_channels: int):
            super().__init__()
            self.transform = nn.Conv2d(in_channels, out_channels, kernel_size=1)

        def forward(self, x, adjacency):
            x = self.transform(x)
            return torch.einsum("nctv,vw->nctw", x, adjacency).contiguous()

    class TemporalConv(nn.Module):
        def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 9):
            super().__init__()
            padding = ((kernel_size - 1) // 2, 0)
            self.conv = nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=(kernel_size, 1),
                padding=padding,
            )

        def forward(self, x):
            return self.conv(x)

    class STGCNBlock(nn.Module):
        def __init__(self, in_channels: int, out_channels: int, temporal_kernel: int = 9, dropout: float = 0.3):
            super().__init__()
            self.graph_conv = GraphConv(in_channels, out_channels)
            self.batch_norm_spatial = nn.BatchNorm2d(out_channels)
            self.temporal_conv = TemporalConv(out_channels, out_channels, temporal_kernel)
            self.batch_norm_temporal = nn.BatchNorm2d(out_channels)
            self.relu = nn.ReLU(inplace=True)
            self.dropout = nn.Dropout(dropout)
            if in_channels == out_channels:
                self.residual = nn.Identity()
            else:
                self.residual = nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, kernel_size=1),
                    nn.BatchNorm2d(out_channels),
                )

        def forward(self, x, adjacency):
            identity = self.residual(x)
            out = self.relu(self.batch_norm_spatial(self.graph_conv(x, adjacency)))
            out = self.dropout(self.batch_norm_temporal(self.temporal_conv(out)))
            return self.relu(out + identity)

    class STGCNModel(nn.Module):
        def __init__(self, in_channels: int = NUM_CHANNELS, num_classes: int = 3, num_nodes: int = NUM_NODES):
            super().__init__()
            self.data_batch_norm = nn.BatchNorm1d(in_channels * num_nodes)
            self.block1 = STGCNBlock(in_channels, 64)
            self.block2 = STGCNBlock(64, 64)
            self.block3 = STGCNBlock(64, 128)
            self.fc = nn.Linear(128, num_classes)

        def forward(self, x, adjacency):
            batch_size, channels, num_frames, num_nodes = x.shape
            x = x.permute(0, 1, 3, 2).contiguous()
            x = x.view(batch_size, channels * num_nodes, num_frames)
            x = self.data_batch_norm(x)
            x = x.view(batch_size, channels, num_nodes, num_frames)
            x = x.permute(0, 1, 3, 2).contiguous()
            x = self.block1(x, adjacency)
            x = self.block2(x, adjacency)
            x = self.block3(x, adjacency)
            x = x.mean(dim=(2, 3))
            return self.fc(x)

    return STGCNModel


def _load_state(path: Path):
    import torch

    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


class ModelHub:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._r3d_lock = threading.Lock()
        self._vit_lock = threading.Lock()
        self._gcn_lock = threading.Lock()
        self._ready = False
        self._torch = None
        self.device = None
        self.r3d = None
        self.vit = None
        self.gcn = None
        self.adjacency = None
        self.r3d_labels = SMIC_3
        self.vit_labels = SMIC_3
        self.gcn_labels = SMIC_3

    def ensure_loaded(self) -> None:
        if self._ready:
            return
        with self._lock:
            if self._ready:
                return
            try:
                import torch

                self._torch = torch
                self.device = torch.device("cpu")
            except Exception as exc:
                print(f"[inference] PyTorch unavailable: {exc}")
                self._ready = True
                return

            self._load_r3d()
            self._load_vit()
            self._load_gcn()
            self._ready = True

    def _load_r3d(self) -> None:
        if not WEIGHTS["r3d18"].is_file() or self._torch is None:
            return
        try:
            from torch import nn
            from torchvision.models.video import r3d_18

            state = _load_state(WEIGHTS["r3d18"])
            num_classes = _head_num_classes(state, "fc.weight")
            self.r3d_labels = labels_for_num_classes(num_classes)
            try:
                model = r3d_18(weights=None)
            except TypeError:
                model = r3d_18(pretrained=False)
            model.fc = nn.Linear(model.fc.in_features, num_classes)
            model.load_state_dict(state)
            model.eval()
            self.r3d = model
        except Exception as exc:
            print(f"[inference] R3D-18 load skipped: {exc}")

    def _load_vit(self) -> None:
        if not WEIGHTS["vit"].is_file() or self._torch is None:
            return
        try:
            import timm
            from torch import nn

            state = _load_state(WEIGHTS["vit"])
            num_classes = _head_num_classes(state, "head.weight")
            self.vit_labels = labels_for_num_classes(num_classes)
            model = timm.create_model("vit_base_patch16_224", pretrained=False)
            model.head = nn.Linear(model.head.in_features, num_classes)
            model.load_state_dict(state)
            model.eval()
            self.vit = model
        except Exception as exc:
            print(f"[inference] ViT load skipped: {exc}")

    def _load_gcn(self) -> None:
        if self._torch is None:
            return
        path = WEIGHTS["gcn_mmew"] if WEIGHTS["gcn_mmew"].is_file() else WEIGHTS["gcn"]
        if not path.is_file():
            return
        try:
            from mediapipe.tasks.python.vision import FaceLandmarksConnections
            from torch import nn

            torch = self._torch
            state = _load_state(path)
            num_classes = _head_num_classes(state, "fc.weight")
            self.gcn_labels = labels_for_num_classes(num_classes)
            STGCNModel = _build_stgcn(torch, nn)
            adjacency = np.zeros((NUM_NODES, NUM_NODES), dtype=np.float32)
            for conn in FaceLandmarksConnections.FACE_LANDMARKS_TESSELATION:
                source, target = conn.start, conn.end
                if source >= NUM_NODES or target >= NUM_NODES:
                    continue
                adjacency[source, target] = 1.0
                adjacency[target, source] = 1.0
            eye = np.eye(NUM_NODES, dtype=np.float32)
            tilde = adjacency + eye
            degree = tilde.sum(axis=1)
            inv = np.power(degree, -0.5, where=degree > 0)
            inv[degree == 0] = 0.0
            scale = np.diag(inv).astype(np.float32)
            self.adjacency = torch.from_numpy(scale @ tilde @ scale)
            model = STGCNModel(num_classes=num_classes)
            model.load_state_dict(state)
            model.eval()
            self.gcn = model
        except Exception as exc:
            print(f"[inference] ST-GCN load skipped: {exc}")

    def _forward(self, model, labels: tuple[str, ...], *args) -> Prediction | None:
        torch = self._torch
        if model is None or torch is None:
            return None
        try:
            import torch.nn.functional as F

            with torch.no_grad():
                logits = model(*args)
                if logits.ndim == 1:
                    logits = logits.unsqueeze(0)
                probs = F.softmax(logits, dim=1)
                confidence, index = torch.max(probs, dim=1)
            idx = int(index[0].item())
            return Prediction(_safe_label(labels, idx), float(confidence[0].item()), "weights")
        except Exception as exc:
            print(f"[inference] forward failed: {exc}")
            return None

    def predict_r3d_window(self, rgb_window: list[np.ndarray]) -> Prediction:
        flow_bgr, flow_mag = _window_flow(
            [np.mean(frame, axis=2).astype(np.uint8) if frame.ndim == 3 else frame for frame in rgb_window]
        )
        if self.r3d is None or self._torch is None or len(rgb_window) < 2:
            return simulate_window("r3d18", flow_mag, 0.0, self.r3d_labels)
        torch = self._torch
        frames = [_imagenet_tensor(frame, R3D_SIZE) for frame in rgb_window]
        video = np.stack(frames, axis=1)
        tensor = torch.from_numpy(video).unsqueeze(0)
        with self._r3d_lock:
            pred = self._forward(self.r3d, self.r3d_labels, tensor)
        return pred or simulate_window("r3d18", flow_mag, 0.0, self.r3d_labels)

    def predict_vit_window(self, gray_window: list[np.ndarray]) -> Prediction:
        flow_bgr, flow_mag = _window_flow(gray_window)
        if flow_bgr is None or self.vit is None or self._torch is None:
            return simulate_window("vit", flow_mag, 0.0, self.vit_labels)
        torch = self._torch
        rgb = flow_bgr[:, :, ::-1].copy()
        tensor = torch.from_numpy(_imagenet_tensor(rgb, VIT_SIZE)).unsqueeze(0)
        with self._vit_lock:
            pred = self._forward(self.vit, self.vit_labels, tensor)
        return pred or simulate_window("vit", flow_mag, 0.0, self.vit_labels)

    def predict_gcn_window(self, landmark_window: list[np.ndarray | None]) -> Prediction:
        motion = _landmark_motion(landmark_window)
        filled = _fill_window_landmarks(landmark_window)
        if filled is None or self.gcn is None or self.adjacency is None or self._torch is None:
            return simulate_window("gcn", 0.0, motion, self.gcn_labels)
        torch = self._torch
        tensor = torch.from_numpy(filled.transpose(2, 0, 1)).unsqueeze(0)
        with self._gcn_lock:
            pred = self._forward(self.gcn, self.gcn_labels, tensor, self.adjacency)
        return pred or simulate_window("gcn", 0.0, motion, self.gcn_labels)


def _fill_window_landmarks(landmarks: list[np.ndarray | None]) -> np.ndarray | None:
    last = None
    clip: list[np.ndarray] = []
    for item in landmarks:
        if item is not None:
            last = item.astype(np.float32)[:NUM_NODES]
        if last is not None:
            clip.append(last)
    if not clip:
        return None
    while len(clip) < NUM_FRAMES:
        clip.insert(0, clip[0])
    return np.stack(clip[-NUM_FRAMES:], axis=0)


HUB = ModelHub()


def predict_sequences(
    rgb_frames: list[np.ndarray],
    gray_frames: list[np.ndarray],
    landmarks: list[np.ndarray | None],
) -> dict[str, list[Prediction]]:
    """Run a sliding 16-frame window and return one prediction per frame."""
    HUB.ensure_loaded()
    length = len(rgb_frames)
    if length == 0:
        empty = Prediction("Others", 0.0, "simulated")
        return {"r3d18": [empty], "vit": [empty], "gcn": [empty]}

    rgb_buf: deque[np.ndarray] = deque(maxlen=NUM_FRAMES)
    gray_buf: deque[np.ndarray] = deque(maxlen=NUM_FRAMES)
    lm_buf: deque[np.ndarray | None] = deque(maxlen=NUM_FRAMES)

    def infer_r3d() -> Prediction:
        return HUB.predict_r3d_window(list(rgb_buf))

    def infer_vit() -> Prediction:
        return HUB.predict_vit_window(list(gray_buf))

    def infer_gcn() -> Prediction:
        return HUB.predict_gcn_window(list(lm_buf))

    r3d: list[Prediction | None] = [None] * length
    vit: list[Prediction | None] = [None] * length
    gcn: list[Prediction | None] = [None] * length
    current_r3d = current_vit = current_gcn = None

    with ThreadPoolExecutor(max_workers=3) as pool:
        for index in range(length):
            rgb_buf.append(rgb_frames[index])
            gray_buf.append(gray_frames[index] if index < len(gray_frames) else rgb_frames[index])
            lm_buf.append(landmarks[index] if index < len(landmarks) else None)
            ready = len(rgb_buf) == NUM_FRAMES
            stepped = (index + 1 - NUM_FRAMES) % INFER_STRIDE == 0
            if (ready and (current_r3d is None or stepped)) or (
                index == length - 1 and current_r3d is None
            ):
                while len(rgb_buf) < NUM_FRAMES and rgb_buf:
                    rgb_buf.appendleft(rgb_buf[0])
                    gray_buf.appendleft(gray_buf[0])
                    lm_buf.appendleft(lm_buf[0])
                fut_r3d = pool.submit(infer_r3d)
                fut_vit = pool.submit(infer_vit)
                fut_gcn = pool.submit(infer_gcn)
                current_r3d = fut_r3d.result()
                current_vit = fut_vit.result()
                current_gcn = fut_gcn.result()
            r3d[index] = current_r3d
            vit[index] = current_vit
            gcn[index] = current_gcn

    fallback = Prediction("Others", 0.0, "simulated")
    fill = lambda seq: [item or next((x for x in seq if x is not None), fallback) for item in seq]
    return {"r3d18": fill(r3d), "vit": fill(vit), "gcn": fill(gcn)}
