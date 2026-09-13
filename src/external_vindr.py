"""
external_vindr.py
-----------------
External validation: đánh giá model (đã train trên JSRT) trên VinDr-CXR để đo
tính tổng quát hóa (Gap 3 trong tổng quan). Chỉ INFERENCE, không train.

VinDr-CXR: dataset công khai VN, có bounding box nhiều tổn thương. Ta lọc các
box nhãn 'Nodule/Mass', chuyển tâm box thành điểm nốt, rồi đo FROC như JSRT.

LƯU Ý: VinDr ảnh DICOM/PNG lớn, nhãn ở train.csv với cột
[image_id, class_name, x_min, y_min, x_max, y_max]. Ta chỉ cần class chứa
'nodule' hoặc 'mass'.

Dùng:
    python -m src.external_vindr --config <cfg> --model <ckpt> \
        --vindr-images <dir> --vindr-csv <train.csv>
"""
import os
import argparse
import glob
import numpy as np

import torch

from .froc import extract_peaks, match_detections, compute_froc


def load_vindr_nodules(csv_path, images_dir):
    """Đọc VinDr CSV -> dict{image_path: [nodule points]}.
    Chỉ giữ box có class chứa 'nodule' hoặc 'mass'."""
    import pandas as pd
    df = pd.read_csv(csv_path)
    # tìm cột class
    col_cls = next((c for c in df.columns if "class" in c.lower()
                    and "name" in c.lower()), None)
    col_id = next((c for c in df.columns if c.lower() in
                   ("image_id", "imageid", "image")), df.columns[0])
    box_cols = {}
    for key in ["x_min", "y_min", "x_max", "y_max"]:
        box_cols[key] = next((c for c in df.columns if c.lower() == key), None)

    if not col_cls or any(v is None for v in box_cols.values()):
        raise SystemExit(f"CSV thiếu cột cần thiết. Có: {list(df.columns)}")

    df = df[df[col_cls].astype(str).str.lower().str.contains("nodule|mass",
                                                             na=False)]
    have = {os.path.splitext(os.path.basename(p))[0]: p
            for p in glob.glob(os.path.join(images_dir, "*.png"))
            + glob.glob(os.path.join(images_dir, "*.jpg"))}

    out = {}
    for _, r in df.iterrows():
        iid = str(r[col_id])
        if iid not in have:
            continue
        cx = (r[box_cols["x_min"]] + r[box_cols["x_max"]]) / 2
        cy = (r[box_cols["y_min"]] + r[box_cols["y_max"]]) / 2
        diam = max(r[box_cols["x_max"]] - r[box_cols["x_min"]],
                   r[box_cols["y_max"]] - r[box_cols["y_min"]])
        out.setdefault(have[iid], []).append(
            {"x": float(cx), "y": float(cy), "diameter_px": float(diam),
             "subtlety": 3})  # VinDr không có subtlety -> gán trung bình
    return out


@torch.no_grad()
def evaluate_vindr(cfg, model, nodule_map, device):
    """Inference model trên ảnh VinDr, đo FROC. Ảnh được resize về image_size."""
    import cv2
    model.eval()
    size = cfg["data"]["image_size"]
    stride = cfg["model"]["heatmap_stride"]
    e = cfg["eval"]
    in_ch = cfg["model"].get("in_channels", 1)

    all_results, n_nod, n_img = [], 0, 0
    for path, nods in nodule_map.items():
        img0 = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img0 is None:
            continue
        h0, w0 = img0.shape
        sx, sy = size / w0, size / h0
        img = cv2.resize(img0, (size, size)).astype(np.float32) / 255.0
        # CLAHE cho khớp tiền xử lý JSRT
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        img = clahe.apply((img * 255).astype(np.uint8)).astype(np.float32) / 255.0

        if in_ch == 2:
            # không có ảnh khử xương cho VinDr -> lặp kênh gốc (fallback)
            x = np.stack([img, img], axis=0)
        else:
            x = img[None]
        xt = torch.from_numpy(x[None]).float().to(device)
        pred = model(xt)[0, 0].cpu().numpy()
        peaks = extract_peaks(pred, e["peak_min_distance"])

        # scale tâm nốt về ảnh đã resize
        nods_scaled = [{"x": n["x"] * sx, "y": n["y"] * sy,
                        "subtlety": n["subtlety"]} for n in nods]
        mm_per_px = 0.175  # xấp xỉ; VinDr không có spacing chuẩn -> dùng hằng số
        results, _ = match_detections(peaks, nods_scaled, stride, mm_per_px,
                                      e["hit_radius_mm"])
        all_results += results
        n_nod += len(nods_scaled)
        n_img += 1

    fppi, sens, sens_at, cpm = compute_froc(all_results, n_img, n_nod,
                                            e["fp_rates"])
    print(f"\n=== External Validation trên VinDr-CXR ===")
    print(f"  Số ảnh: {n_img} | số nốt: {n_nod}")
    for r in e["fp_rates"]:
        print(f"  Sensitivity @ {r:>5} FP/ảnh : {sens_at[r]*100:5.1f}%")
    print(f"  CPM : {cpm*100:5.1f}%")
    print("  (So với CPM trên JSRT -> đo mức sụt giảm tổng quát hóa)")
    return sens_at, cpm


def main(config, model_ckpt, vindr_images, vindr_csv):
    import yaml
    from .train import load_full_model
    with open(config) as f:
        cfg = yaml.safe_load(f)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_full_model(cfg, model_ckpt, device)
    nodule_map = load_vindr_nodules(vindr_csv, vindr_images)
    print(f"VinDr: {len(nodule_map)} ảnh có nốt/mass")
    evaluate_vindr(cfg, model, nodule_map, device)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--vindr-images", required=True)
    ap.add_argument("--vindr-csv", required=True)
    args = ap.parse_args()
    main(args.config, args.model, args.vindr_images, args.vindr_csv)
