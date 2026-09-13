"""
gen_bs_channel.py
-----------------
Sau khi train U-Net khử xương, sinh kênh khử xương cho TẤT CẢ ảnh JSRT đã
preprocess, lưu vào processed_dir/images_bs/ (cùng tên file với images/).
Detector input 2 kênh sẽ đọc từ đây.

Dùng (Colab chạy bằng biến, hoặc):
    python -m src.gen_bs_channel --config <cfg> --bs-ckpt <ckpt>
"""
import os
import glob
import argparse
import numpy as np
import cv2
import yaml

import torch
from .bone_suppression import load_bs_model, suppress_image_masked


def main(config, bs_ckpt, blend=0.7):
    with open(config) as f:
        cfg = yaml.safe_load(f)
    proc = cfg["data"]["processed_dir"]
    img_dir = os.path.join(proc, "images")
    out_dir = os.path.join(proc, "images_bs")
    os.makedirs(out_dir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, size = load_bs_model(bs_ckpt, device)
    print(f"Nạp U-Net khử xương ({size}px) trên {device} | blend={blend}, lung-mask")

    files = glob.glob(os.path.join(img_dir, "*.png"))
    print(f"Sinh kênh khử xương cho {len(files)} ảnh...")
    for i, f in enumerate(files):
        img = cv2.imread(f, cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0
        bs = suppress_image_masked(model, img, device, size, blend=blend)
        cv2.imwrite(os.path.join(out_dir, os.path.basename(f)),
                    (bs * 255).astype(np.uint8))
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(files)}")
    print(f"Xong. Ảnh khử xương (masked+blend) -> {out_dir}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--bs-ckpt", required=True)
    ap.add_argument("--blend", type=float, default=0.7)
    args = ap.parse_args()
    main(args.config, args.bs_ckpt, args.blend)
