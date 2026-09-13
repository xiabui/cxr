"""
raddino_extract.py — Trích xuất đặc trưng RAD-DINO cho toàn bộ ảnh JSRT (chạy MỘT LẦN).

RAD-DINO là vision transformer được tiền huấn luyện tự giám sát (DINOv2) trên
882.775 ảnh X-quang ngực. Ta dùng nó ở chế độ ĐÓNG BĂNG: chỉ chạy suy luận,
không lan truyền ngược, nên GPU 6GB thừa sức.

Đặt file này ở thư mục gốc dự án (cùng cấp với src/).

    pip install transformers pillow
    python raddino_extract.py --check          # kiểm tra nạp model, in kích thước
    python raddino_extract.py                  # trích xuất toàn bộ

Kết quả: features_raddino/<stem>.npy, mỗi file là mảng [C, G, G] kiểu float16.
Dung lượng ước tính: ~250 ảnh x ~2MB = ~520MB.
"""
import os
import sys
import argparse
import glob

import numpy as np
import torch
import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

MODEL_ID = "microsoft/rad-dino"
OUT_DIR = "features_raddino"


def load_model(device):
    from transformers import AutoModel, AutoImageProcessor
    print(f"Nạp {MODEL_ID} ... (lần đầu sẽ tải về, cần mạng)")
    proc = AutoImageProcessor.from_pretrained(MODEL_ID)
    model = AutoModel.from_pretrained(MODEL_ID).to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    n = sum(p.numel() for p in model.parameters())
    print(f"OK. Số tham số: {n/1e6:.1f} triệu (đóng băng toàn bộ)")
    return model, proc


@torch.no_grad()
def encode(model, proc, img_gray_uint8, device):
    """img_gray_uint8: mảng HxW uint8. Trả về [C, G, G] float32."""
    from PIL import Image
    im = Image.fromarray(img_gray_uint8).convert("RGB")
    inputs = proc(images=im, return_tensors="pt").to(device)
    out = model(**inputs)
    h = out.last_hidden_state          # [1, 1+N, C]  (token đầu là CLS)
    patch = h[:, 1:, :]                # bỏ CLS
    n = patch.shape[1]
    g = int(round(n ** 0.5))
    assert g * g == n, f"số patch {n} không phải số chính phương"
    feat = patch[0].transpose(0, 1).reshape(-1, g, g)   # [C, G, G]
    return feat.float().cpu().numpy()


def main(check_only):
    import cv2
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Thiết bị:", device)
    model, proc = load_model(device)

    cfg = yaml.safe_load(open("configs/jsrt_10fold.yaml"))
    img_dir = os.path.join(cfg["data"]["processed_dir"], "images")
    files = sorted(glob.glob(os.path.join(img_dir, "*.png")))
    if not files:
        raise SystemExit(f"Không thấy ảnh trong {img_dir}. Chạy preprocess trước.")
    print(f"Tìm thấy {len(files)} ảnh trong {img_dir}")

    if check_only:
        g = cv2.imread(files[0], cv2.IMREAD_GRAYSCALE)
        f = encode(model, proc, g, device)
        print(f"\nKIỂM TRA OK")
        print(f"  ảnh mẫu : {os.path.basename(files[0])} {g.shape}")
        print(f"  đặc trưng: {f.shape}  (kênh={f.shape[0]}, lưới={f.shape[1]}x{f.shape[2]})")
        print(f"  dung lượng mỗi ảnh (float16): {f.size*2/1e6:.2f} MB")
        print(f"  tổng ước tính: {f.size*2*len(files)/1e6:.0f} MB")
        print("\nNếu số trên hợp lý, chạy lại không có --check để trích xuất toàn bộ.")
        return

    os.makedirs(OUT_DIR, exist_ok=True)
    for i, p in enumerate(files, 1):
        stem = os.path.splitext(os.path.basename(p))[0]
        out = os.path.join(OUT_DIR, stem + ".npy")
        if os.path.exists(out):
            continue
        g = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        if g is None:
            print(f"  bỏ qua (không đọc được): {p}")
            continue
        f = encode(model, proc, g, device)
        np.save(out, f.astype(np.float16))
        if i % 25 == 0 or i == len(files):
            print(f"  {i}/{len(files)}")
    print(f"\nXong. Đặc trưng -> {os.path.abspath(OUT_DIR)}")
    print("Bước tiếp theo: python raddino_train.py --config configs/jsrt_10fold.yaml")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true",
                    help="chỉ kiểm tra nạp model và in kích thước đặc trưng")
    main(ap.parse_args().check)
