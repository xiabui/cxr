"""
raddino_train2.py — BẢN SỬA LỖI NGƯỠNG.

Khác biệt duy nhất so với bản trước: độ nhạy phân tầng theo subtlety được tính
TẠI CÙNG CÁC MỨC DƯƠNG TÍNH GIẢ như CPM, thay vì đếm mọi phát hiện không áp
ngưỡng. Bản trước cho con số bị thổi phồng và mất xu hướng đơn điệu.

Cách tính ở đây tự chứa, không phụ thuộc match_detections, nên tránh được mọi
khác biệt ngầm về ngưỡng:
  1. Với mỗi nốt thật, tìm điểm tin cậy CAO NHẤT trong số các đỉnh nằm trong
     bán kính dung sai -> "điểm số của nốt".
  2. Các đỉnh không khớp nốt nào là dương tính giả; gom điểm số của chúng.
  3. Với mỗi mức FP/ảnh f, ngưỡng = điểm số của dương tính giả thứ (n_ảnh * f).
  4. Độ nhạy nhóm = tỷ lệ nốt trong nhóm có điểm số >= ngưỡng.
  5. Báo cáo trung bình qua các mức FP -> tương đương CPM cho từng nhóm.

    python raddino_train2.py --config configs/jsrt_10fold.yaml
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

from src.dataset import JSRTNoduleDataset
from src.froc import extract_peaks
from src.model import focal_mse_loss

FEAT_DIR = "features_raddino"


class HeatmapHead(nn.Module):
    def __init__(self, in_ch, out_size, width=256):
        super().__init__()
        self.out_size = out_size
        self.proj = nn.Sequential(
            nn.Conv2d(in_ch, width, 1), nn.BatchNorm2d(width), nn.ReLU(inplace=True))
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


def load_feat(stem, device):
    stem = os.path.splitext(os.path.basename(str(stem)))[0]
    p = os.path.join(FEAT_DIR, stem + ".npy")
    return torch.from_numpy(np.load(p).astype(np.float32)).to(device)


def build_cache(cfg, files, device):
    ds = JSRTNoduleDataset(cfg, files, augment=False)
    cache = []
    for i in range(len(ds)):
        item = ds[i]
        meta = item[-1]
        hm = item[1]
        cache.append((load_feat(meta["filename"], device), hm.to(device), meta))
    return cache


def train_head(cache_train, in_ch, out_size, device, epochs, lr):
    head = HeatmapHead(in_ch, out_size).to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    bs = 8
    head.train()
    for ep in range(epochs):
        idx = torch.randperm(len(cache_train))
        for s in range(0, len(idx), bs):
            batch = [cache_train[j] for j in idx[s:s + bs]]
            x = torch.stack([b[0] for b in batch])
            y = torch.stack([b[1] for b in batch])
            opt.zero_grad()
            loss = focal_mse_loss(head(x), y)
            if not torch.isfinite(loss):
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), 5.0)
            opt.step()
        sched.step()
    return head


def peak_xy_score(pk):
    """extract_peaks có thể trả tuple hoặc dict — chuẩn hóa về (x, y, score)."""
    if isinstance(pk, dict):
        return float(pk["x"]), float(pk["y"]), float(pk.get("score", pk.get("val", 0)))
    return float(pk[0]), float(pk[1]), float(pk[2])


@torch.no_grad()
def eval_fold(head, cache_test, cfg):
    """Trả về (nốt kèm điểm số, điểm số dương tính giả, bản ghi theo từng ảnh).

    Thành phần thứ ba gom theo ẢNH và là đầu vào của bootstrap_ci.py: bootstrap
    phải lấy mẫu lại ở cấp độ ảnh vì ngưỡng FROC định nghĩa theo số FP mỗi ảnh.
    """
    head.eval()
    e = cfg["eval"]
    stride = cfg["model"]["heatmap_stride"]
    radius_mm = e["hit_radius_mm"]
    nod_scores, fp_scores, recs = [], [], []

    for f, hm, meta in cache_test:
        pred = head(f[None])[0, 0].cpu().numpy()
        peaks = [peak_xy_score(p) for p in
                 extract_peaks(pred, e["peak_min_distance"])]
        mm = meta["mm_per_px"]
        nods = meta["nodules"]
        # tọa độ nốt trên lưới bản đồ nhiệt
        gt = [(nd["x"] / stride, nd["y"] / stride, nd["subtlety"]) for nd in nods]
        used = set()
        best = [0.0] * len(gt)
        fps_img = []
        for pi, (px, py, sc) in enumerate(peaks):
            hit_i, hit_d = -1, None
            for gi, (gx, gy, _) in enumerate(gt):
                d_mm = np.hypot(px - gx, py - gy) * stride * mm
                if d_mm <= radius_mm and (hit_d is None or d_mm < hit_d):
                    hit_i, hit_d = gi, d_mm
            if hit_i >= 0:
                best[hit_i] = max(best[hit_i], sc)
                used.add(pi)
            else:
                fps_img.append(sc)
        fp_scores += fps_img
        for gi, (_, _, sub) in enumerate(gt):
            nod_scores.append((best[gi], sub))
        recs.append({
            "image": os.path.splitext(os.path.basename(str(meta["filename"])))[0],
            "nodules": [{"score": float(best[gi]), "subtlety": int(gt[gi][2])}
                        for gi in range(len(gt))],
            "fps": [float(v) for v in fps_img],
        })
    return nod_scores, fp_scores, recs


def sens_by_subtlety(nod_scores, fp_scores, n_images, fp_rates):
    """Độ nhạy phân tầng tại từng mức FP/ảnh + trung bình (CPM theo nhóm)."""
    fps = np.sort(np.array(fp_scores))[::-1] if fp_scores else np.array([])
    groups = defaultdict(list)
    for sc, sub in nod_scores:
        groups[sub].append(sc)

    per_rate, table = {}, defaultdict(dict)
    for f in fp_rates:
        k = int(round(n_images * f))
        if len(fps) == 0:
            thr = 0.0
        elif k <= 0:
            thr = float(fps[0]) + 1e-9
        elif k >= len(fps):
            thr = 0.0
        else:
            thr = float(fps[k - 1])
        per_rate[f] = thr
        for sub, scores in groups.items():
            hit = sum(1 for s in scores if s >= thr and s > 0)
            table[sub][f] = (hit, len(scores))
    return table, per_rate, groups


def main(config, epochs, lr):
    cfg = yaml.safe_load(open(config))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    size = cfg["data"]["image_size"]
    stride = cfg["model"]["heatmap_stride"]
    out_size = size // stride
    e = cfg["eval"]

    if not os.path.isdir(FEAT_DIR):
        raise SystemExit(f"Chưa có {FEAT_DIR}. Chạy raddino_extract.py trước.")

    from src.train import list_images
    files = list_images(cfg)
    have = {os.path.splitext(os.path.basename(p))[0]
            for p in glob.glob(os.path.join(FEAT_DIR, "*.npy"))}
    files = [f for f in files
             if os.path.splitext(os.path.basename(str(f)))[0] in have]
    print(f"Số ảnh có đặc trưng: {len(files)}")

    print("Nạp đặc trưng vào bộ nhớ...")
    cache = build_cache(cfg, files, device)
    in_ch = cache[0][0].shape[0]
    print(f"Kênh đặc trưng: {in_ch} | lưới: {tuple(cache[0][0].shape[1:])} "
          f"| bản đồ nhiệt: {out_size}x{out_size}")

    proto = cfg["train"].get("protocol", "kfold")
    n = len(cache)
    if proto == "loco":
        folds = [([j for j in range(n) if j != i], [i]) for i in range(n)]
    else:
        k = cfg["train"].get("folds", 10)
        idx = np.arange(n)
        np.random.RandomState(cfg.get("seed", 42)).shuffle(idx)
        parts = np.array_split(idx, k)
        folds = [([j for j in idx if j not in set(p)], list(p)) for p in parts]
    print(f"Giao thức: {proto} | {len(folds)} fold\n")

    all_nod, all_fp, all_recs = [], [], []
    for fi, (tr, te) in enumerate(folds, 1):
        if proto != "loco" or fi == 1 or fi % 25 == 0:
            print(f"=== Fold {fi}/{len(folds)} (train={len(tr)}, test={len(te)}) ===")
        head = train_head([cache[j] for j in tr], in_ch, out_size, device, epochs, lr)
        ns, fs, rc = eval_fold(head, [cache[j] for j in te], cfg)
        all_nod += ns; all_fp += fs; all_recs += rc
        del head
        torch.cuda.empty_cache()

    fp_rates = e["fp_rates"]
    table, thr, groups = sens_by_subtlety(all_nod, all_fp, n, fp_rates)

    # độ nhạy tổng thể tại từng mức FP + CPM
    sens_at, tot = {}, len(all_nod)
    for f in fp_rates:
        hit = sum(1 for sc, _ in all_nod if sc >= thr[f] and sc > 0)
        sens_at[f] = hit / tot if tot else 0.0
    cpm = float(np.mean([sens_at[f] for f in fp_rates]))

    print("\n" + "=" * 56)
    print("KẾT QUẢ — RAD-DINO đóng băng + đầu nhẹ (ngưỡng NHẤT QUÁN)")
    print("=" * 56)
    print(f"  Tổng số nốt: {tot} | số ảnh test: {n}")
    for f in fp_rates:
        print(f"  Sensitivity @ {f:>5} FP/ảnh : {sens_at[f]*100:5.1f}%   (ngưỡng {thr[f]:.3f})")
    print(f"  CPM (trung bình FROC)       : {cpm*100:5.1f}%")

    names = {1: "1 Cực khó thấy", 2: "2 Rất khó thấy", 3: "3 Trung bình",
             4: "4 Tương đối rõ", 5: "5 Rõ ràng"}
    print("\n  Độ nhạy phân tầng (trung bình qua các mức FP — so sánh được với CPM):")
    by_sub_cpm = {}
    for sub in sorted(table):
        vals = [table[sub][f][0] / table[sub][f][1] for f in fp_rates
                if table[sub][f][1]]
        m = float(np.mean(vals)) if vals else 0.0
        by_sub_cpm[sub] = m
        ntot = table[sub][fp_rates[0]][1]
        print(f"    {names.get(sub, sub):<16} n={ntot:<3d}  {m*100:5.1f}%")

    print("\n  Chi tiết tại 1,0 FP/ảnh:")
    f1 = min(fp_rates, key=lambda z: abs(z - 1.0))
    for sub in sorted(table):
        h, t = table[sub][f1]
        print(f"    {names.get(sub, sub):<16} {h:3d}/{t:<3d}  {100*h/t if t else 0:5.1f}%")

    os.makedirs("runs", exist_ok=True)
    json.dump({"backbone": "rad-dino-frozen", "protocol": proto, "cpm": cpm,
               "sens_at": {str(k): v for k, v in sens_at.items()},
               "subtlety_cpm": {str(k): v for k, v in by_sub_cpm.items()},
               "n_nodules": tot, "n_images": n},
              open("runs/results_raddino.json", "w"), indent=2)
    print("\nĐã lưu runs/results_raddino.json")

    json.dump({"tag": f"raddino_plain_{proto}", "backbone": "rad-dino-frozen",
               "protocol": proto, "seed": int(cfg.get("seed", 42)),
               "config": os.path.basename(config),
               "fp_rates": list(fp_rates), "n_images": n, "n_nodules": tot,
               "images": all_recs}, open("runs/scores_raddino_plain.json", "w"))
    print("Đã lưu runs/scores_raddino_plain.json  (dùng cho bootstrap_ci.py)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/jsrt_10fold.yaml")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--lr", type=float, default=1e-3)
    a = ap.parse_args()
    main(a.config, a.epochs, a.lr)
