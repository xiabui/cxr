"""
raddino_extract2.py — Trích xuất đặc trưng RAD-DINO CÓ TĂNG CƯỜNG DỮ LIỆU
và CÓ KÊNH KHỬ XƯƠNG, để so sánh công bằng với các cấu hình cũ.

Khác bản 1 ở hai điểm:
  1. Mỗi ảnh được trích xuất K biến thể (1 bản gốc + K-1 bản tăng cường).
     Phép biến hình affine được ghi lại và ÁP DỤNG LUÔN cho tọa độ nốt, nên
     nhãn luôn khớp ảnh — tránh lỗi lệch nhãn.
  2. Ba kênh đầu vào của ViT là [ảnh gốc, ảnh khử xương, ảnh gốc] thay vì lặp
     ảnh xám ba lần, nhờ đó đóng góp khử bóng xương được đưa vào.

    python raddino_extract2.py --check          # thử 1 ảnh, in kích thước
    python raddino_extract2.py --variants 8     # trích xuất thật

Kết quả:
    features_aug/<stem>_v<k>.npy   đặc trưng [C,G,G] float16
    features_aug/coords.json       tọa độ nốt đã biến đổi cho từng biến thể

Ước tính: 247 ảnh x 8 biến thể x ~2 MB = ~4 GB đĩa, chừng 45-60 phút trên RTX 3060.
"""
import os
import sys
import json
import glob
import argparse

import numpy as np
import cv2
import torch
import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

MODEL_ID = "microsoft/rad-dino"
OUT_DIR = "features_aug"


# ---------------------------------------------------------------- biến hình
def make_affine(rng, size, first):
    """Trả về ma trận affine 2x3. Biến thể đầu tiên là bản gốc (không đổi)."""
    if first:
        return np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], np.float32)
    ang = rng.uniform(-7, 7)              # xoay nhẹ, giữ giải phẫu hợp lý
    scale = rng.uniform(0.93, 1.07)
    tx = rng.uniform(-0.04, 0.04) * size
    ty = rng.uniform(-0.04, 0.04) * size
    c = size / 2.0
    M = cv2.getRotationMatrix2D((c, c), ang, scale)
    M[0, 2] += tx
    M[1, 2] += ty
    return M.astype(np.float32)


def apply_affine_pts(M, pts):
    """pts: list [(x,y)] -> list [(x',y')]."""
    out = []
    for x, y in pts:
        xn = M[0, 0] * x + M[0, 1] * y + M[0, 2]
        yn = M[1, 0] * x + M[1, 1] * y + M[1, 2]
        out.append((float(xn), float(yn)))
    return out


def photometric(img_u8, rng, first):
    """Nhiễu độ sáng/tương phản/gamma. Không đổi tọa độ."""
    if first:
        return img_u8
    x = img_u8.astype(np.float32) / 255.0
    x = np.clip(x * rng.uniform(0.90, 1.10) + rng.uniform(-0.05, 0.05), 0, 1)
    x = np.power(x, rng.uniform(0.85, 1.18))
    return (x * 255).astype(np.uint8)


# ---------------------------------------------------------------- model
def load_model(device):
    from transformers import AutoModel, AutoImageProcessor
    print(f"Nạp {MODEL_ID} ...")
    proc = AutoImageProcessor.from_pretrained(MODEL_ID)
    model = AutoModel.from_pretrained(MODEL_ID).to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    print(f"OK, {sum(p.numel() for p in model.parameters())/1e6:.1f} triệu tham số (đóng băng)")
    return model, proc


@torch.no_grad()
def encode_rgb(model, proc, rgb_u8, device):
    """rgb_u8: HxWx3 uint8 -> đặc trưng [C,G,G] float32."""
    from PIL import Image
    inputs = proc(images=Image.fromarray(rgb_u8), return_tensors="pt").to(device)
    h = model(**inputs).last_hidden_state       # [1, 1+N, C]
    patch = h[:, 1:, :]
    n = patch.shape[1]
    g = int(round(n ** 0.5))
    assert g * g == n, f"số patch {n} không phải số chính phương"
    return patch[0].transpose(0, 1).reshape(-1, g, g).float().cpu().numpy()


# ---------------------------------------------------------------- main
def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Thiết bị:", device)
    cfg = yaml.safe_load(open(args.config))
    proc_dir = cfg["data"]["processed_dir"]
    img_dir = os.path.join(proc_dir, "images")
    bs_dir = os.path.join(proc_dir, "images_bs")
    size = cfg["data"]["image_size"]

    files = sorted(glob.glob(os.path.join(img_dir, "*.png")))
    if not files:
        raise SystemExit(f"Không thấy ảnh trong {img_dir}")
    use_bs = os.path.isdir(bs_dir)
    print(f"{len(files)} ảnh | kênh khử xương: {'CÓ' if use_bs else 'KHÔNG (dùng ảnh gốc)'}")

    # nhãn nốt
    import pandas as pd
    lab = pd.read_csv(os.path.join(proc_dir, "labels_processed.csv"))
    by_stem = {}
    for _, r in lab.iterrows():
        st = os.path.splitext(str(r["filename"]))[0]
        by_stem.setdefault(st, []).append(
            (float(r["x"]), float(r["y"]), int(r["subtlety"]),
             float(r.get("diameter_mm", np.nan))))

    model, proc = load_model(device)

    def build_rgb(stem, M, rng, first):
        g = cv2.imread(os.path.join(img_dir, stem + ".png"), cv2.IMREAD_GRAYSCALE)
        if g is None:
            return None
        if g.shape[0] != size:
            g = cv2.resize(g, (size, size))
        g = cv2.warpAffine(g, M, (size, size), flags=cv2.INTER_LINEAR,
                           borderMode=cv2.BORDER_REPLICATE)
        g = photometric(g, rng, first)
        if use_bs:
            b = cv2.imread(os.path.join(bs_dir, stem + ".png"), cv2.IMREAD_GRAYSCALE)
            if b is not None:
                if b.shape[0] != size:
                    b = cv2.resize(b, (size, size))
                b = cv2.warpAffine(b, M, (size, size), flags=cv2.INTER_LINEAR,
                                   borderMode=cv2.BORDER_REPLICATE)
                b = photometric(b, rng, first)
            else:
                b = g
        else:
            b = g
        return np.stack([g, b, g], axis=-1)

    if args.check:
        stem = os.path.splitext(os.path.basename(files[0]))[0]
        rng = np.random.RandomState(0)
        M = make_affine(rng, size, first=True)
        rgb = build_rgb(stem, M, rng, True)
        f = encode_rgb(model, proc, rgb, device)
        per = f.size * 2 / 1e6
        print(f"\nKIỂM TRA OK")
        print(f"  ảnh    : {stem}  RGB {rgb.shape}")
        print(f"  đặc trưng: {f.shape}  ({per:.2f} MB/biến thể ở float16)")
        print(f"  tổng ước tính với {args.variants} biến thể: "
              f"{per*len(files)*args.variants/1000:.1f} GB")
        # thử biến thể có biến hình + kiểm tra tọa độ
        M2 = make_affine(np.random.RandomState(1), size, first=False)
        pts = [(p[0], p[1]) for p in by_stem.get(stem, [])]
        if pts:
            print(f"  tọa độ nốt gốc      : {[(round(a),round(b)) for a,b in pts]}")
            print(f"  sau biến hình       : {[(round(a),round(b)) for a,b in apply_affine_pts(M2, pts)]}")
        return

    os.makedirs(OUT_DIR, exist_ok=True)
    coords = {}
    K = args.variants
    for i, p in enumerate(files, 1):
        stem = os.path.splitext(os.path.basename(p))[0]
        rng = np.random.RandomState(abs(hash(stem)) % (2**31))
        nods = by_stem.get(stem, [])
        for v in range(K):
            key = f"{stem}_v{v}"
            out = os.path.join(OUT_DIR, key + ".npy")
            M = make_affine(rng, size, first=(v == 0))
            pts = apply_affine_pts(M, [(a, b) for a, b, _, _ in nods])
            coords[key] = [
                {"x": pts[j][0], "y": pts[j][1],
                 "subtlety": nods[j][2], "diameter_mm": nods[j][3]}
                for j in range(len(nods))
                if 0 <= pts[j][0] < size and 0 <= pts[j][1] < size]
            if os.path.exists(out):
                continue
            rgb = build_rgb(stem, M, rng, first=(v == 0))
            if rgb is None:
                continue
            np.save(out, encode_rgb(model, proc, rgb, device).astype(np.float16))
        if i % 20 == 0 or i == len(files):
            print(f"  {i}/{len(files)} ảnh ({i*K} biến thể)")

    with open(os.path.join(OUT_DIR, "coords.json"), "w") as f:
        json.dump({"variants": K, "size": size, "use_bs": use_bs,
                   "coords": coords}, f)
    print(f"\nXong -> {os.path.abspath(OUT_DIR)}")
    print("Bước tiếp: python raddino_train3.py --config configs/jsrt_10fold.yaml")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/jsrt_10fold.yaml")
    ap.add_argument("--variants", type=int, default=8,
                    help="số biến thể mỗi ảnh (biến thể 0 là bản gốc)")
    ap.add_argument("--check", action="store_true")
    main(ap.parse_args())
