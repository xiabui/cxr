"""
raddino_train3.py — Huấn luyện đầu phát hiện trên đặc trưng RAD-DINO CÓ tăng cường.

Ba điểm phương pháp quan trọng:
  1. CHIA FOLD THEO ẢNH, không theo biến thể. Mọi biến thể của cùng một ảnh
     luôn nằm cùng phía train hoặc test, nếu không sẽ rò rỉ dữ liệu.
  2. ĐÁNH GIÁ CHỈ DÙNG BẢN GỐC (biến thể 0). Bản tăng cường chỉ để huấn luyện.
  3. Ngưỡng NHẤT QUÁN giữa FROC và bảng subtlety (giữ từ bản train2).

    python raddino_train3.py --config configs/jsrt_10fold.yaml
    python raddino_train3.py --config configs/jsrt_loocv.yaml    # chính thức
"""
import os
import sys
import json
import glob
import argparse
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.froc import extract_peaks
from src.model import focal_mse_loss

FEAT_DIR = "features_aug"


class HeatmapHead(nn.Module):
    def __init__(self, in_ch, out_size, width=256, drop=0.1):
        super().__init__()
        self.out_size = out_size
        self.proj = nn.Sequential(
            nn.Conv2d(in_ch, width, 1), nn.BatchNorm2d(width), nn.ReLU(inplace=True),
            nn.Dropout2d(drop))
        self.up1 = nn.Sequential(
            nn.Conv2d(width, width // 2, 3, padding=1),
            nn.BatchNorm2d(width // 2), nn.ReLU(inplace=True))
        self.up2 = nn.Sequential(
            nn.Conv2d(width // 2, width // 4, 3, padding=1),
            nn.BatchNorm2d(width // 4), nn.ReLU(inplace=True))
        self.out = nn.Conv2d(width // 4, 1, 1)
        nn.init.constant_(self.out.bias, -2.19)

    def forward(self, x):
        x = self.proj(x)
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        x = self.up1(x)
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        x = self.up2(x)
        x = F.interpolate(x, size=(self.out_size, self.out_size),
                          mode="bilinear", align_corners=False)
        return torch.sigmoid(self.out(x))


def gauss(hs, cx, cy, sigma):
    ys, xs = np.mgrid[0:hs, 0:hs]
    return np.exp(-((xs - cx) ** 2 + (ys - cy) ** 2) / (2 * sigma ** 2)).astype(np.float32)


def build_heatmap(nods, hs, stride, sigma_px):
    """Dựng bản đồ nhiệt đích, khớp đúng cách làm trong dataset.py."""
    hm = np.zeros((hs, hs), np.float32)
    s = max(1.0, sigma_px / stride)
    for nd in nods:
        cx, cy = nd["x"] / stride, nd["y"] / stride
        if not (0 <= cx < hs and 0 <= cy < hs):
            continue
        hm = np.maximum(hm, gauss(hs, cx, cy, s))
        ix, iy = int(round(cx)), int(round(cy))
        if 0 <= ix < hs and 0 <= iy < hs:
            hm[iy, ix] = 1.0          # ép tâm = 1.0 (bài học kỹ thuật 1)
    return hm


def peak_xy_score(pk):
    if isinstance(pk, dict):
        return float(pk["x"]), float(pk["y"]), float(pk.get("score", pk.get("val", 0)))
    return float(pk[0]), float(pk[1]), float(pk[2])


def main(a):
    cfg = yaml.safe_load(open(a.config))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    size = cfg["data"]["image_size"]
    stride = cfg["model"]["heatmap_stride"]
    sigma_px = cfg["model"].get("gaussian_sigma", 8)
    hs = size // stride
    e = cfg["eval"]
    mm_per_px = 0.175 * (2048.0 / size)      # JSRT gốc 2048, 0,175 mm/điểm ảnh

    meta_p = os.path.join(FEAT_DIR, "coords.json")
    if not os.path.exists(meta_p):
        raise SystemExit(f"Chưa có {meta_p}. Chạy raddino_extract2.py trước.")
    meta = json.load(open(meta_p))
    K, coords = meta["variants"], meta["coords"]
    print(f"Biến thể mỗi ảnh: {K} | kênh khử xương: {meta.get('use_bs')}")

    stems = sorted({k.rsplit("_v", 1)[0] for k in coords})
    stems = [s for s in stems if os.path.exists(os.path.join(FEAT_DIR, f"{s}_v0.npy"))]
    n = len(stems)
    print(f"Số ảnh: {n}")

    probe = np.load(os.path.join(FEAT_DIR, f"{stems[0]}_v0.npy"))
    in_ch = probe.shape[0]
    print(f"Kênh đặc trưng: {in_ch} | lưới {probe.shape[1]}x{probe.shape[2]} "
          f"| bản đồ nhiệt {hs}x{hs}")

    # bản đồ nhiệt đích dựng sẵn (nhẹ), đặc trưng nạp lười từ đĩa
    HM = {k: build_heatmap(v, hs, stride, sigma_px) for k, v in coords.items()}

    def feat(key):
        return np.load(os.path.join(FEAT_DIR, key + ".npy")).astype(np.float32)

    def train_head(train_stems, epochs, lr):
        head = HeatmapHead(in_ch, hs).to(device)
        opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=1e-4)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
        keys = [f"{s}_v{v}" for s in train_stems for v in range(min(K, a.max_variants))
                if os.path.exists(os.path.join(FEAT_DIR, f"{s}_v{v}.npy"))]
        bs = 8
        head.train()
        for ep in range(epochs):
            np.random.shuffle(keys)
            for i in range(0, len(keys), bs):
                kk = keys[i:i + bs]
                x = torch.from_numpy(np.stack([feat(k) for k in kk])).to(device)
                y = torch.from_numpy(np.stack([HM[k][None] for k in kk])).to(device)
                opt.zero_grad()
                loss = focal_mse_loss(head(x), y)
                if not torch.isfinite(loss):
                    continue
                loss.backward()
                torch.nn.utils.clip_grad_norm_(head.parameters(), 5.0)
                opt.step()
            sch.step()
        return head

    @torch.no_grad()
    def eval_stems(head, test_stems):
        """Trả về bản ghi THEO TỪNG ẢNH.

        Gom theo ảnh chứ không trộn phẳng, vì bootstrap phải lấy mẫu lại ở
        cấp độ ẢNH: ngưỡng FROC được định nghĩa theo số FP trên mỗi ảnh, nên
        nốt và dương tính giả của cùng một ảnh phải đi cùng nhau.
        """
        head.eval()
        recs = []
        for s in test_stems:
            key = f"{s}_v0"                     # CHỈ bản gốc
            x = torch.from_numpy(feat(key)[None]).to(device)
            pred = head(x)[0, 0].cpu().numpy()
            peaks = [peak_xy_score(p) for p in
                     extract_peaks(pred, e["peak_min_distance"])]
            gt = [(nd["x"] / stride, nd["y"] / stride, nd["subtlety"])
                  for nd in coords[key]]
            best = [0.0] * len(gt)
            fps_img = []
            for px, py, sc in peaks:
                hi, hd = -1, None
                for gi, (gx, gy, _) in enumerate(gt):
                    d = np.hypot(px - gx, py - gy) * stride * mm_per_px
                    if d <= e["hit_radius_mm"] and (hd is None or d < hd):
                        hi, hd = gi, d
                if hi >= 0:
                    best[hi] = max(best[hi], sc)
                else:
                    fps_img.append(sc)
            recs.append({
                "image": s,
                "nodules": [{"score": float(best[gi]), "subtlety": int(gt[gi][2])}
                            for gi in range(len(gt))],
                "fps": [float(v) for v in fps_img],
            })
        return recs

    proto = cfg["train"].get("protocol", "kfold")
    seed = a.seed if a.seed is not None else int(cfg.get("seed", 42))
    np.random.seed(seed)
    torch.manual_seed(seed)
    print(f"Seed: {seed}")
    idx = np.arange(n)
    if proto == "loco":
        folds = [([j for j in range(n) if j != i], [i]) for i in range(n)]
    else:
        k = cfg["train"].get("folds", 10)
        np.random.RandomState(seed).shuffle(idx)
        parts = np.array_split(idx, k)
        folds = [([j for j in idx if j not in set(p)], list(p)) for p in parts]
    print(f"Giao thức: {proto} | {len(folds)} fold "
          f"(chia theo ẢNH — mọi biến thể cùng phía)\n")

    all_recs = []
    for fi, (tr, te) in enumerate(folds, 1):
        if proto != "loco" or fi == 1 or fi % 25 == 0:
            print(f"=== Fold {fi}/{len(folds)} (train={len(tr)} ảnh x {K} biến thể, "
                  f"test={len(te)}) ===")
        head = train_head([stems[j] for j in tr], a.epochs, a.lr)
        all_recs += eval_stems(head, [stems[j] for j in te])
        del head
        torch.cuda.empty_cache()

    all_nod = [(nd["score"], nd["subtlety"]) for r in all_recs for nd in r["nodules"]]
    all_fp = [v for r in all_recs for v in r["fps"]]

    # ---- ngưỡng nhất quán ----
    fps = np.sort(np.array(all_fp))[::-1] if all_fp else np.array([])
    rates = e["fp_rates"]
    thr = {}
    for f in rates:
        kk = int(round(n * f))
        thr[f] = (0.0 if len(fps) == 0 or kk >= len(fps)
                  else (float(fps[0]) + 1e-9 if kk <= 0 else float(fps[kk - 1])))
    tot = len(all_nod)
    sens_at = {f: sum(1 for sc, _ in all_nod if sc >= thr[f] and sc > 0) / tot
               for f in rates}
    cpm = float(np.mean([sens_at[f] for f in rates]))

    print("\n" + "=" * 60)
    print("KẾT QUẢ — RAD-DINO + tăng cường + kênh khử xương")
    print("=" * 60)
    print(f"  Tổng số nốt: {tot} | số ảnh test: {n}")
    for f in rates:
        print(f"  Sensitivity @ {f:>5} FP/ảnh : {sens_at[f]*100:5.1f}%")
    print(f"  CPM (trung bình FROC)       : {cpm*100:5.1f}%")

    groups = defaultdict(list)
    for sc, sub in all_nod:
        groups[sub].append(sc)
    names = {1: "1 Cực khó thấy", 2: "2 Rất khó thấy", 3: "3 Trung bình",
             4: "4 Tương đối rõ", 5: "5 Rõ ràng"}
    print("\n  Độ nhạy phân tầng (trung bình qua các mức FP):")
    by_sub = {}
    for sub in sorted(groups):
        sc = groups[sub]
        m = float(np.mean([sum(1 for s in sc if s >= thr[f] and s > 0) / len(sc)
                           for f in rates]))
        by_sub[sub] = m
        print(f"    {names.get(sub, sub):<16} n={len(sc):<3d}  {m*100:5.1f}%")

    print("\n  Chi tiết tại 1,0 FP/ảnh:")
    f1 = min(rates, key=lambda z: abs(z - 1.0))
    for sub in sorted(groups):
        sc = groups[sub]
        h = sum(1 for s in sc if s >= thr[f1] and s > 0)
        print(f"    {names.get(sub, sub):<16} {h:3d}/{len(sc):<3d}  {100*h/len(sc):5.1f}%")

    os.makedirs("runs", exist_ok=True)
    tag = a.tag or f"raddino_aug_{proto}_seed{seed}"
    res_p = f"runs/results_{tag}.json"
    json.dump({"backbone": "rad-dino-frozen+aug+bs", "protocol": proto,
               "variants": K, "cpm": cpm, "seed": seed,
               "sens_at": {str(k): v for k, v in sens_at.items()},
               "subtlety_cpm": {str(k): v for k, v in by_sub.items()},
               "n_nodules": tot, "n_images": n},
              open(res_p, "w"), indent=2)
    print(f"\nĐã lưu {res_p}")

    # Điểm thô theo từng ảnh — đầu vào cho bootstrap_ci.py.
    # Bắt buộc phải lưu: không có điểm thô thì không tính lại được ngưỡng
    # trong mỗi lần lấy mẫu bootstrap.
    sc_p = f"runs/scores_{tag}.json"
    json.dump({"tag": tag, "backbone": "rad-dino-frozen+aug+bs",
               "protocol": proto, "seed": seed, "variants": K,
               "config": os.path.basename(a.config),
               "fp_rates": list(rates), "n_images": n, "n_nodules": tot,
               "images": all_recs},
              open(sc_p, "w"))
    print(f"Đã lưu {sc_p}  (dùng cho bootstrap_ci.py)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/jsrt_10fold.yaml")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--max-variants", type=int, default=99)
    ap.add_argument("--seed", type=int, default=None,
                    help="ghi đè seed trong config, dùng khi chạy nhiều seed")
    ap.add_argument("--tag", default=None,
                    help="hậu tố tên file kết quả; mặc định suy ra từ protocol+seed")
    main(ap.parse_args())
