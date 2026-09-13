"""Tiện ích chung: seed, I/O ảnh JSRT, Grad-CAM overlay."""
import os
import random
import numpy as np


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def read_jsrt_img(path: str, shape=(2048, 2048)) -> np.ndarray:
    """Đọc file .IMG gốc của JSRT: 16-bit big-endian, không header.
    Giá trị cao = tối (film), nên thường cần đảo. Trả về float32 [0,1].
    Nếu là PNG/JPG thì đọc bằng OpenCV, có nhiều lớp dự phòng khi OpenCV
    trả None (PNG hỏng nhẹ, profile màu lạ, hoặc 16-bit)."""
    ext = os.path.splitext(path)[1].lower()
    if ext in (".png", ".jpg", ".jpeg", ".tif", ".tiff"):
        import cv2
        if not os.path.exists(path):
            raise FileNotFoundError(path)
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if img is None:
            # dự phòng 1: đọc grayscale trực tiếp
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            # dự phòng 2: đọc qua PIL (chịu được PNG profile lạ tốt hơn)
            try:
                from PIL import Image
                img = np.array(Image.open(path).convert("L"))
            except Exception:
                img = None
        if img is None:
            # dự phòng 3: decode từ bytes bằng imdecode
            try:
                buf = np.fromfile(path, dtype=np.uint8)
                img = cv2.imdecode(buf, cv2.IMREAD_GRAYSCALE)
            except Exception:
                img = None
        if img is None:
            raise ValueError(f"Không đọc được ảnh (file hỏng?): {path}")
        if img.ndim == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = img.astype(np.float32)
    else:
        raw = np.fromfile(path, dtype=">u2")  # big-endian uint16
        if raw.size != shape[0] * shape[1]:
            raise ValueError(f"{path}: {raw.size} px, kỳ vọng {shape[0]*shape[1]}")
        img = raw.reshape(shape).astype(np.float32)
    img = (img - img.min()) / (img.max() - img.min() + 1e-8)
    return img


def maybe_invert(img: np.ndarray) -> np.ndarray:
    """JSRT gốc: xương sáng trên nền tối là mong muốn. Nếu trung vị > 0.5
    (nền sáng), đảo lại cho nhất quán."""
    return 1.0 - img if np.median(img) > 0.5 else img


def gradcam_overlay(image: np.ndarray, cam: np.ndarray, alpha: float = 0.4):
    """Chồng heatmap Grad-CAM lên ảnh xám để kiểm chứng mô hình 'nhìn đúng chỗ'."""
    import cv2
    img8 = (np.clip(image, 0, 1) * 255).astype(np.uint8)
    img8 = cv2.cvtColor(img8, cv2.COLOR_GRAY2BGR)
    cam_r = cv2.resize(cam, (img8.shape[1], img8.shape[0]))
    cam_r = (cam_r - cam_r.min()) / (cam_r.max() - cam_r.min() + 1e-8)
    heat = cv2.applyColorMap((cam_r * 255).astype(np.uint8), cv2.COLORMAP_JET)
    return cv2.addWeighted(img8, 1 - alpha, heat, alpha, 0)
