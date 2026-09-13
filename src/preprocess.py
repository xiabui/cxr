"""Tiền xử lý JSRT: đảo màu -> CLAHE -> crop vùng phổi (nếu có mask) -> resize.
Lưu ra PNG 8-bit + ghi lại hệ số biến đổi toạ độ để cập nhật nhãn nốt.
"""
import os
import json
import argparse
import numpy as np
import pandas as pd
import cv2
import yaml
from tqdm import tqdm

from .utils import read_jsrt_img, maybe_invert, set_seed


def apply_clahe(img01: np.ndarray, clip: float, grid: int) -> np.ndarray:
    img8 = (np.clip(img01, 0, 1) * 255).astype(np.uint8)
    clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(grid, grid))
    return clahe.apply(img8).astype(np.float32) / 255.0


def lung_bbox_from_mask(mask: np.ndarray, pad: float = 0.05):
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return None
    h, w = mask.shape
    x0, x1 = xs.min(), xs.max()
    y0, y1 = ys.min(), ys.max()
    px, py = int((x1 - x0) * pad), int((y1 - y0) * pad)
    return (max(0, x0 - px), max(0, y0 - py),
            min(w, x1 + px), min(h, y1 + py))


def process_one(img_path, mask_path, cfg):
    img = maybe_invert(read_jsrt_img(img_path)) if cfg["preprocess"]["invert_if_needed"] \
        else read_jsrt_img(img_path)
    H0, W0 = img.shape
    ox, oy, scale = 0, 0, 1.0

    if cfg["preprocess"]["lung_crop"] and mask_path and os.path.exists(mask_path):
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is not None:
            mask = cv2.resize(mask, (W0, H0), interpolation=cv2.INTER_NEAREST)
            bb = lung_bbox_from_mask(mask)
            if bb:
                ox, oy, x1, y1 = bb
                img = img[oy:y1, ox:x1]

    if cfg["preprocess"]["clahe"]:
        img = apply_clahe(img, cfg["preprocess"]["clahe_clip"],
                          cfg["preprocess"]["clahe_grid"])

    size = cfg["data"]["image_size"]
    h, w = img.shape
    scale = size / max(h, w)
    img_r = cv2.resize(img, (int(w * scale), int(h * scale)))
    canvas = np.zeros((size, size), np.float32)  # pad về vuông
    canvas[:img_r.shape[0], :img_r.shape[1]] = img_r
    # transform: (x_orig,y_orig) -> (x_new,y_new) = ((x-ox)*scale, (y-oy)*scale)
    return canvas, {"ox": ox, "oy": oy, "scale": scale, "H0": H0, "W0": W0}


def main(config):
    with open(config) as f:
        cfg = yaml.safe_load(f)
    set_seed(cfg["seed"])
    d = cfg["data"]
    img_dir = os.path.join(d["root"], d["images_dir"])
    out_dir = os.path.join(d["processed_dir"], "images")
    os.makedirs(out_dir, exist_ok=True)
    labels = pd.read_csv(os.path.join(d["root"], d["labels_csv"])) \
        if os.path.exists(os.path.join(d["root"], d["labels_csv"])) else pd.DataFrame()

    transforms, new_rows = {}, []
    skipped = []
    files = [f for f in os.listdir(img_dir)
             if f.lower().endswith((".img", ".png", ".jpg", ".jpeg"))]
    for fn in tqdm(files, desc="preprocess"):
        mask_path = os.path.join(d["root"], d["masks_dir"],
                                 os.path.splitext(fn)[0] + ".png") if d["masks_dir"] else None
        try:
            img, tf = process_one(os.path.join(img_dir, fn), mask_path, cfg)
        except Exception as e:
            skipped.append((fn, str(e)))
            continue
        stem = os.path.splitext(fn)[0]
        cv2.imwrite(os.path.join(out_dir, stem + ".png"),
                    (img * 255).astype(np.uint8))
        transforms[fn] = tf
        # cập nhật toạ độ nốt về ảnh đã xử lý
        rows = labels[labels["filename"] == fn] if len(labels) else []
        for _, r in (rows.iterrows() if len(labels) else []):
            nx = (r["x"] - tf["ox"]) * tf["scale"]
            ny = (r["y"] - tf["oy"]) * tf["scale"]
            new_rows.append({"filename": stem + ".png", "x": nx, "y": ny,
                             "diameter_mm": r.get("diameter_mm", np.nan),
                             "subtlety": int(r.get("subtlety", 3)),
                             "malignant": int(r.get("malignant", 1))})

    if skipped:
        print(f"\nBỏ qua {len(skipped)} ảnh không đọc được:")
        for fn, err in skipped[:10]:
            print(f"  {fn}: {err}")

    with open(os.path.join(d["processed_dir"], "transforms.json"), "w") as f:
        json.dump(transforms, f)
    pd.DataFrame(new_rows).to_csv(
        os.path.join(d["processed_dir"], "labels_processed.csv"), index=False)
    print(f"Xong. Ảnh -> {out_dir} | nốt: {len(new_rows)} dòng")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/baseline.yaml")
    main(ap.parse_args().config)
