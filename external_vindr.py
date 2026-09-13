"""
external_vindr.py — External validation trên VinDr-CXR (BẢN VIẾT LẠI).

Sửa các vấn đề của bản cũ:
  - Bản cũ hard-code mm_per_px = 0.175, sai với cả JSRT lẫn VinDr. Bản này
    dùng tiêu chí "đỉnh dự đoán nằm TRONG hộp thật", vốn là cách đối sánh
    tự nhiên khi nhãn là bounding box và tránh hoàn toàn vấn đề pixel spacing.
  - Bản cũ không sinh kênh khử xương cho VinDr mà lặp ảnh gốc, nên không
    đánh giá đúng cấu hình tốt nhất. Bản này chạy U-Net khử xương trên VinDr.
  - Bản cũ chỉ hỗ trợ ResNet34. Bản này hỗ trợ cả RAD-DINO đóng băng.

    python external_vindr.py --vindr-dir data/vindr --backbone raddino \
        --bs-ckpt checkpoints/bone_suppression_unet.pt \
        --head checkpoints/raddino_head.pt
"""
import os, sys, json, argparse
from collections import defaultdict

import numpy as np
import cv2
import torch
import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.froc import extract_peaks

MODEL_ID = "microsoft/rad-dino"


def peak_xy_score(pk):
    if isinstance(pk, dict):
        return float(pk["x"]), float(pk["y"]), float(pk.get("score", pk.get("val", 0)))
    return float(pk[0]), float(pk[1]), float(pk[2])


def preprocess(gray, size, clip=2.0, grid=8):
    img = cv2.resize(gray, (size, size))
    if img.dtype != np.uint8:
        img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return cv2.createCLAHE(clipLimit=clip, tileGridSize=(grid, grid)).apply(img)


@torch.no_grad()
def gen_bone_suppressed(u8, bs_model, device, size, blend=0.7):
    """Chạy U-Net khử xương, pha trộn với ảnh gốc như khi làm trên JSRT."""
    if bs_model is None:
        return u8
    ins = bs_model.get("in_size", 256)
    x = cv2.resize(u8, (ins, ins)).astype(np.float32) / 255.0
    t = torch.from_numpy(x[None, None]).to(device)
    y = bs_model["net"](t)[0, 0].cpu().numpy()
    y = np.clip(y, 0, 1)
    y = cv2.resize(y, (size, size))
    base = u8.astype(np.float32) / 255.0
    out = blend * y + (1 - blend) * base
    return (np.clip(out, 0, 1) * 255).astype(np.uint8)


def main(a):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    meta = json.load(open(os.path.join(a.vindr_dir, "vindr_gt.json")))
    size, GT = meta["size"], meta["gt"]
    img_dir = os.path.join(a.vindr_dir, "images")
    print(f"VinDr: {len(GT)} ảnh | nguồn nhãn: {meta.get('source')} "
          f"| đã gộp hộp: {meta.get('merged')}")

    # ---- U-Net khử xương ----
    bs_model = None
    if a.bs_ckpt and os.path.exists(a.bs_ckpt):
        from src.bone_suppression import UNet
        ck = torch.load(a.bs_ckpt, map_location=device, weights_only=False)
        state = ck.get("model_state", ck.get("state", ck))
        net = UNet().to(device)
        net.load_state_dict(state); net.eval()
        for p in net.parameters(): p.requires_grad_(False)
        bs_model = {"net": net, "in_size": a.bs_size}
        print(f"U-Net khử xương: {a.bs_ckpt} (chạy ở {a.bs_size}px, blend {a.blend})")
    else:
        print("KHÔNG dùng kênh khử xương (thiếu --bs-ckpt)")

    # ---- backbone + head ----
    if a.backbone == "raddino":
        from transformers import AutoModel, AutoImageProcessor
        from raddino_train3 import HeatmapHead
        proc = AutoImageProcessor.from_pretrained(MODEL_ID)
        enc = AutoModel.from_pretrained(MODEL_ID).to(device).eval()
        for p in enc.parameters(): p.requires_grad_(False)
        ck = torch.load(a.head, map_location=device, weights_only=False)
        head = HeatmapHead(ck["in_ch"], ck["out_size"]).to(device)
        head.load_state_dict(ck["state"]); head.eval()
        stride = ck["stride"]
        print(f"Backbone: RAD-DINO (đóng băng) + head {a.head}")

        @torch.no_grad()
        def predict(rgb_u8):
            from PIL import Image
            inp = proc(images=Image.fromarray(rgb_u8), return_tensors="pt").to(device)
            h = enc(**inp).last_hidden_state[:, 1:, :]
            g = int(round(h.shape[1] ** 0.5))
            f = h[0].transpose(0, 1).reshape(-1, g, g)[None]
            return head(f)[0, 0].cpu().numpy()
    else:
        from src.train import load_full_model
        cfg = yaml.safe_load(open(a.config))
        model = load_full_model(cfg, a.head, device)
        stride = cfg["model"]["heatmap_stride"]
        in_ch = cfg["model"].get("in_channels", 1)
        print(f"Backbone: ResNet34 ({a.head}), in_channels={in_ch}")

        @torch.no_grad()
        def predict(rgb_u8):
            g = rgb_u8[:, :, 0].astype(np.float32) / 255.0
            b = rgb_u8[:, :, 1].astype(np.float32) / 255.0
            x = np.stack([g, b], 0) if in_ch == 2 else g[None]
            return model(torch.from_numpy(x[None]).float().to(device))[0, 0].cpu().numpy()

    # ---- chạy trên toàn bộ ảnh ----
    nod_scores, fp_scores, n_img, n_nod = [], [], 0, 0
    recs = []          # điểm thô theo từng ảnh, đầu vào cho bootstrap_ci.py
    for k, (iid, boxes) in enumerate(sorted(GT.items()), 1):
        p = os.path.join(img_dir, iid + ".png")
        gray = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        if gray is None:
            continue
        u8 = preprocess(gray, size)
        bs = gen_bone_suppressed(u8, bs_model, device, size, a.blend)
        rgb = np.stack([u8, bs, u8], axis=-1)

        pred = predict(rgb)
        peaks = [peak_xy_score(x) for x in extract_peaks(pred, a.peak_dist)]
        best = [0.0] * len(boxes)
        fps_img = []
        for px, py, sc in peaks:
            ax, ay = px * stride, py * stride     # về hệ tọa độ ảnh
            hit = -1
            for gi, nd in enumerate(boxes):
                x1, y1, x2, y2 = nd["box"]
                m = a.margin
                if (x1 - m) <= ax <= (x2 + m) and (y1 - m) <= ay <= (y2 + m):
                    hit = gi; break
            if hit >= 0:
                best[hit] = max(best[hit], sc)
            else:
                fps_img.append(sc)
        fp_scores += fps_img
        nod_scores += best
        # VinDr không có nhãn subtlety, để 0 — bootstrap_ci.py sẽ bỏ qua
        # phần phân tầng và chỉ tính khoảng tin cậy cho CPM/độ nhạy tổng thể.
        recs.append({"image": iid,
                     "nodules": [{"score": float(v), "subtlety": 0} for v in best],
                     "fps": [float(v) for v in fps_img]})
        n_nod += len(boxes); n_img += 1
        if k % 200 == 0:
            print(f"  {k}/{len(GT)}")

    # ---- FROC + CPM, cùng định nghĩa ngưỡng như trên JSRT ----
    rates = [0.125, 0.25, 0.5, 1.0, 2.0, 4.0]
    fps = np.sort(np.array(fp_scores))[::-1] if fp_scores else np.array([])
    thr = {}
    for f in rates:
        kk = int(round(n_img * f))
        thr[f] = (0.0 if len(fps) == 0 or kk >= len(fps)
                  else (float(fps[0]) + 1e-9 if kk <= 0 else float(fps[kk - 1])))
    tot = len(nod_scores)
    sens = {f: (sum(1 for s in nod_scores if s >= thr[f] and s > 0) / tot if tot else 0.0)
            for f in rates}
    cpm = float(np.mean([sens[f] for f in rates]))

    print("\n" + "=" * 58)
    print("EXTERNAL VALIDATION — VinDr-CXR")
    print("=" * 58)
    print(f"  Số ảnh: {n_img} | số nốt: {n_nod}")
    print(f"  Tiêu chí hit: đỉnh nằm trong hộp thật (biên {a.margin} px)")
    for f in rates:
        print(f"  Sensitivity @ {f:>5} FP/ảnh : {sens[f]*100:5.1f}%")
    print(f"  CPM                          : {cpm*100:5.1f}%")
    print("\n  So sánh nội bộ trên JSRT (LOOCV): baseline 29,5% | "
          "ResNet34+U-Net+CBAM 34,2% | RAD-DINO+U-Net 42,5%")
    print("  Lưu ý: tiêu chí hit khác nhau (JSRT dùng bán kính mm, VinDr dùng hộp),")
    print("         nên hai con số KHÔNG so sánh trực tiếp được. Mức sụt giảm chỉ")
    print("         mang tính tham khảo về khả năng tổng quát hóa.")

    os.makedirs("runs", exist_ok=True)
    json.dump({"dataset": "vindr-cxr", "backbone": a.backbone,
               "bone_suppression": bs_model is not None,
               "n_images": n_img, "n_nodules": n_nod, "cpm": cpm,
               "sens_at": {str(k): v for k, v in sens.items()}},
              open("runs/results_vindr.json", "w"), indent=2)
    print("\nĐã lưu runs/results_vindr.json")

    json.dump({"tag": f"vindr_{a.backbone}", "backbone": a.backbone,
               "protocol": "external", "dataset": "vindr-cxr",
               "fp_rates": list(rates), "n_images": n_img, "n_nodules": n_nod,
               "images": recs}, open("runs/scores_vindr.json", "w"))
    print("Đã lưu runs/scores_vindr.json  (chạy: python bootstrap_ci.py "
          "runs/scores_vindr.json)")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--vindr-dir", default="data/vindr")
    p.add_argument("--backbone", choices=["raddino", "resnet"], default="raddino")
    p.add_argument("--head", default="checkpoints/raddino_head.pt")
    p.add_argument("--config", default="configs/jsrt_10fold.yaml")
    p.add_argument("--bs-ckpt", default="checkpoints/bone_suppression_unet.pt")
    p.add_argument("--bs-size", type=int, default=256)
    p.add_argument("--blend", type=float, default=0.7)
    p.add_argument("--peak-dist", type=int, default=10)
    p.add_argument("--margin", type=float, default=0.0,
                   help="nới biên hộp khi đối sánh, đơn vị điểm ảnh")
    main(p.parse_args())
