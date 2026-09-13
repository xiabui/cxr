"""
raddino_finetune.py — Tinh chỉnh MỘT PHẦN các tầng cuối của RAD-DINO.

Đây là mục 8.4 trong CLAUDE.md, hạng mục duy nhất bị chặn bởi 6GB VRAM của
RTX 3060. Chạy trên Colab Pro (A100/L4) thì thoải mái.

TẠI SAO THÍ NGHIỆM NÀY QUAN TRỌNG HƠN VẺ NGOÀI CỦA NÓ

Phát hiện 2 của bài báo nói: tăng cường dữ liệu PHẢN TÁC DỤNG (−20,5 điểm) và
lời giải thích là "vì backbone đóng băng nên không thích nghi được với ảnh đã
biến đổi". Đó mới chỉ là GIẢ THUYẾT — chưa có thí nghiệm nào kiểm chứng.

Script này kiểm chứng trực tiếp: nếu mở băng vài tầng cuối mà tăng cường dữ
liệu hết phản tác dụng, giả thuyết được xác nhận. Nếu vẫn phản tác dụng thì
lời giải thích trong bài báo SAI và phải viết lại. Chạy cả hai nhánh bằng cờ
--variants.

Tương tự, phát hiện 3 nói nhóm subtlety 1 "kháng lại mọi cấu hình". Nếu tinh
chỉnh cũng không cải thiện nhóm 1 thì kết luận mạnh hơn hẳn.

BÀI HỌC KỸ THUẬT ĐÃ MÃ HÓA SẴN (CLAUDE.md mục 5, đừng gỡ ra):
  - amp: KHÔNG dùng. AMP từng làm loss bằng đúng 0 và mô hình không học.
    Cờ --amp có tồn tại nhưng mặc định TẮT và có cảnh báo.
  - clip gradient norm 5.0.
  - tâm Gaussian bị ép đúng 1.0 (dùng lại build_heatmap của raddino_train3).
  - ngưỡng FROC và bảng subtlety dùng CHUNG một ngưỡng.
  - chia fold theo ẢNH.

    # kiểm tra nhanh trước khi chạy thật
    python raddino_finetune.py --config configs/jsrt_colab.yaml --check

    # nhánh A: tinh chỉnh, KHÔNG tăng cường
    python raddino_finetune.py --config configs/jsrt_colab.yaml \
        --unfreeze 4 --epochs 30 --variants 1 --tag ft4_noaug

    # nhánh B: tinh chỉnh, CÓ tăng cường  <- nhánh trả lời phát hiện 2
    python raddino_finetune.py --config configs/jsrt_colab.yaml \
        --unfreeze 4 --epochs 30 --variants 8 --tag ft4_aug
"""
import os
import sys
import json
import glob
import time
import argparse
from collections import defaultdict

import numpy as np
import cv2
import torch
import torch.nn as nn
import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.froc import extract_peaks
from src.model import focal_mse_loss
from raddino_train3 import HeatmapHead, build_heatmap, peak_xy_score

MODEL_ID = "microsoft/rad-dino"
SUB_NAMES = {1: "1 Cực khó thấy", 2: "2 Rất khó thấy", 3: "3 Trung bình",
             4: "4 Tương đối rõ", 5: "5 Rõ ràng"}


# ------------------------------------------------------------------ dữ liệu
def make_affine(rng, size, first):
    """Giữ ĐÚNG dải biến đổi của raddino_extract2.py để so sánh được."""
    if first:
        return np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], np.float32)
    ang = rng.uniform(-7, 7)
    scale = rng.uniform(0.93, 1.07)
    c = size / 2.0
    M = cv2.getRotationMatrix2D((c, c), ang, scale)
    M[0, 2] += rng.uniform(-0.04, 0.04) * size
    M[1, 2] += rng.uniform(-0.04, 0.04) * size
    return M.astype(np.float32)


def apply_affine_pts(M, pts):
    return [(float(M[0, 0] * x + M[0, 1] * y + M[0, 2]),
             float(M[1, 0] * x + M[1, 1] * y + M[1, 2])) for x, y in pts]


def photometric(u8, rng, first):
    if first:
        return u8
    x = u8.astype(np.float32) / 255.0
    x = np.clip(x * rng.uniform(0.90, 1.10) + rng.uniform(-0.05, 0.05), 0, 1)
    return (np.power(x, rng.uniform(0.85, 1.18)) * 255).astype(np.uint8)


class Images:
    """Nạp ảnh gốc + ảnh khử xương, sinh biến thể, chuẩn hóa cho ViT.

    Giữ ảnh ở uint8 trong RAM (247 ảnh ~200MB) rồi chuẩn hóa theo lô, thay vì
    lưu sẵn tensor float (sẽ tốn gấp bốn lần).
    """

    def __init__(self, cfg, proc, size_vit):
        d = cfg["data"]
        pdir = d["processed_dir"]
        self.img_dir = os.path.join(pdir, "images")
        self.bs_dir = os.path.join(pdir, "images_bs")
        self.use_bs = os.path.isdir(self.bs_dir)
        self.size = d["image_size"]
        self.size_vit = size_vit
        self.mean = np.array(proc.image_mean, np.float32).reshape(3, 1, 1)
        self.std = np.array(proc.image_std, np.float32).reshape(3, 1, 1)

        import pandas as pd
        lab = pd.read_csv(os.path.join(pdir, "labels_processed.csv"))
        self.nod = defaultdict(list)
        for _, r in lab.iterrows():
            st = os.path.splitext(str(r["filename"]))[0]
            self.nod[st].append({"x": float(r["x"]), "y": float(r["y"]),
                                 "subtlety": int(r["subtlety"])})

        self.stems = [os.path.splitext(os.path.basename(p))[0]
                      for p in sorted(glob.glob(os.path.join(self.img_dir, "*.png")))]
        self.cache = {}

    def raw(self, stem):
        if stem not in self.cache:
            g = cv2.imread(os.path.join(self.img_dir, stem + ".png"),
                           cv2.IMREAD_GRAYSCALE)
            if g is None:
                raise SystemExit(f"không đọc được ảnh {stem}")
            b = None
            if self.use_bs:
                b = cv2.imread(os.path.join(self.bs_dir, stem + ".png"),
                               cv2.IMREAD_GRAYSCALE)
            self.cache[stem] = (g, b if b is not None else g)
        return self.cache[stem]

    def variant(self, stem, v, rng):
        """Trả về (pixel_values[3,S,S] float32, danh sách nốt đã biến đổi)."""
        g, b = self.raw(stem)
        first = (v == 0)
        M = make_affine(rng, self.size, first)
        if not first:
            g = cv2.warpAffine(g, M, (self.size, self.size), flags=cv2.INTER_LINEAR,
                               borderMode=cv2.BORDER_REFLECT101)
            b = cv2.warpAffine(b, M, (self.size, self.size), flags=cv2.INTER_LINEAR,
                               borderMode=cv2.BORDER_REFLECT101)
            g = photometric(g, rng, first)
        # ba kênh [gốc, khử xương, gốc] — giống hệt raddino_extract2.py
        rgb = np.stack([g, b, g], axis=-1)
        rgb = cv2.resize(rgb, (self.size_vit, self.size_vit))
        x = rgb.astype(np.float32).transpose(2, 0, 1) / 255.0
        x = (x - self.mean) / self.std

        pts = [(n["x"], n["y"]) for n in self.nod.get(stem, [])]
        moved = apply_affine_pts(M, pts) if not first else pts
        nods = [{"x": p[0], "y": p[1], "subtlety": n["subtlety"]}
                for p, n in zip(moved, self.nod.get(stem, []))]
        return x, nods


# ------------------------------------------------------------------ mô hình
class FineTuneNet(nn.Module):
    """RAD-DINO với N tầng cuối mở băng + đầu heatmap."""

    def __init__(self, enc, head):
        super().__init__()
        self.enc = enc
        self.head = head

    def forward(self, pixel_values):
        h = self.enc(pixel_values=pixel_values).last_hidden_state[:, 1:, :]
        n = h.shape[1]
        g = int(round(n ** 0.5))
        f = h.transpose(1, 2).reshape(h.shape[0], -1, g, g)
        return self.head(f)


def build_model(unfreeze, device, grad_ckpt):
    from transformers import AutoModel, AutoImageProcessor
    proc = AutoImageProcessor.from_pretrained(MODEL_ID)
    enc = AutoModel.from_pretrained(MODEL_ID)
    for p in enc.parameters():
        p.requires_grad_(False)

    layers = enc.encoder.layer
    n_open = max(0, min(unfreeze, len(layers)))
    opened = []
    for blk in layers[len(layers) - n_open:]:
        for p in blk.parameters():
            p.requires_grad_(True)
        opened.append(blk)
    if n_open > 0 and hasattr(enc, "layernorm"):
        for p in enc.layernorm.parameters():
            p.requires_grad_(True)

    if grad_ckpt:
        enc.gradient_checkpointing_enable()
    enc.to(device)
    tot = sum(p.numel() for p in enc.parameters())
    tr = sum(p.numel() for p in enc.parameters() if p.requires_grad)
    print(f"RAD-DINO: {len(layers)} tầng, mở băng {n_open} tầng cuối | "
          f"{tr/1e6:.1f}/{tot/1e6:.1f} triệu tham số học được "
          f"({100*tr/tot:.1f}%)")
    return enc, proc


# ---------------------------------------------------------------- đánh giá
@torch.no_grad()
def eval_stems(net, imgs, stems, cfg, device, hs, stride, mm_per_px):
    """Y HỆT raddino_train3.eval_stems, chỉ khác là chạy cả backbone.

    Chỉ đánh giá trên BẢN GỐC (biến thể 0). Bản tăng cường chỉ để huấn luyện.
    """
    e = cfg["eval"]
    net.eval()
    rng = np.random.RandomState(0)
    recs = []
    for s in stems:
        x, nods = imgs.variant(s, 0, rng)
        t = torch.from_numpy(x[None]).to(device)
        pred = net(t)[0, 0].float().cpu().numpy()
        peaks = [peak_xy_score(p) for p in extract_peaks(pred, e["peak_min_distance"])]
        gt = [(n["x"] / stride, n["y"] / stride, n["subtlety"]) for n in nods]
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
        recs.append({"image": s,
                     "nodules": [{"score": float(best[i]), "subtlety": int(gt[i][2])}
                                 for i in range(len(gt))],
                     "fps": [float(v) for v in fps_img]})
    return recs


def report(all_recs, rates, n, tag, extra):
    """Ngưỡng NHẤT QUÁN giữa FROC và bảng subtlety (mục 5, đã mắc hai lần)."""
    all_nod = [(nd["score"], nd["subtlety"]) for r in all_recs for nd in r["nodules"]]
    all_fp = [v for r in all_recs for v in r["fps"]]
    fps = np.sort(np.array(all_fp))[::-1] if all_fp else np.array([])
    thr = {}
    for f in rates:
        k = int(round(n * f))
        thr[f] = (0.0 if len(fps) == 0 or k >= len(fps)
                  else (float(fps[0]) + 1e-9 if k <= 0 else float(fps[k - 1])))
    tot = len(all_nod)
    sens = {f: sum(1 for sc, _ in all_nod if sc >= thr[f] and sc > 0) / tot
            for f in rates}
    cpm = float(np.mean([sens[f] for f in rates]))

    print("\n" + "=" * 62)
    print(f"KẾT QUẢ — RAD-DINO TINH CHỈNH MỘT PHẦN  [{tag}]")
    print("=" * 62)
    print(f"  Tổng số nốt: {tot} | số ảnh: {n}")
    for f in rates:
        print(f"  Sensitivity @ {f:>5} FP/ảnh : {sens[f]*100:5.1f}%")
    print(f"  CPM                         : {cpm*100:5.1f}%")

    groups = defaultdict(list)
    for sc, sub in all_nod:
        groups[sub].append(sc)
    f1 = min(rates, key=lambda z: abs(z - 1.0))
    by_sub = {}
    print("\n  Độ nhạy phân tầng tại 1,0 FP/ảnh:")
    for sub in sorted(groups):
        g = groups[sub]
        h = sum(1 for s in g if s >= thr[f1] and s > 0)
        by_sub[sub] = float(np.mean([sum(1 for s in g if s >= thr[f] and s > 0)/len(g)
                                     for f in rates]))
        print(f"    {SUB_NAMES.get(sub, sub):<16} {h:3d}/{len(g):<3d}  "
              f"{100*h/len(g):5.1f}%")

    os.makedirs("runs", exist_ok=True)
    json.dump({"backbone": "rad-dino-finetuned", "cpm": cpm, "tag": tag,
               "sens_at": {str(k): v for k, v in sens.items()},
               "subtlety_cpm": {str(k): v for k, v in by_sub.items()},
               "n_nodules": tot, "n_images": n, **extra},
              open(f"runs/results_{tag}.json", "w"), indent=2)
    json.dump({"tag": tag, "backbone": "rad-dino-finetuned", **extra,
               "fp_rates": list(rates), "n_images": n, "n_nodules": tot,
               "images": all_recs}, open(f"runs/scores_{tag}.json", "w"))
    print(f"\nĐã lưu runs/results_{tag}.json và runs/scores_{tag}.json")
    print(f"Chạy tiếp: python bootstrap_ci.py runs/scores_{tag}.json")
    return cpm


# ------------------------------------------------------------------- huấn luyện
def main(a):
    cfg = yaml.safe_load(open(a.config))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    size = cfg["data"]["image_size"]
    stride = cfg["model"]["heatmap_stride"]
    sigma = cfg["model"].get("gaussian_sigma", 8)
    hs = size // stride
    rates = cfg["eval"]["fp_rates"]
    mm_per_px = 0.175 * (2048.0 / size)

    if a.amp:
        print("CẢNH BÁO: đang bật AMP. CLAUDE.md mục 5 ghi rõ AMP từng làm loss")
        print("          bằng đúng 0 và mô hình không học gì. Hãy kiểm tra loss.")

    torch.manual_seed(a.seed)
    np.random.seed(a.seed)

    enc, proc = build_model(a.unfreeze, device, a.grad_ckpt)
    size_vit = a.vit_size or (proc.size.get("shortest_edge")
                              if isinstance(proc.size, dict) else 518)
    imgs = Images(cfg, proc, size_vit)
    print(f"{len(imgs.stems)} ảnh | kênh khử xương: "
          f"{'CÓ' if imgs.use_bs else 'KHÔNG'} | ViT nhận {size_vit}px")

    # dò số kênh đặc trưng
    with torch.no_grad():
        x0, _ = imgs.variant(imgs.stems[0], 0, np.random.RandomState(0))
        h = enc(pixel_values=torch.from_numpy(x0[None]).to(device)).last_hidden_state
    in_ch, grid = h.shape[-1], int(round((h.shape[1] - 1) ** 0.5))
    print(f"Đặc trưng: {in_ch} kênh, lưới {grid}x{grid} -> heatmap {hs}x{hs}")

    if a.check:
        print("\n--check: nạp mô hình và dữ liệu OK, thoát trước khi huấn luyện.")
        return

    stems = imgs.stems
    n = len(stems)
    proto = cfg["train"].get("protocol", "kfold")
    idx = np.arange(n)
    if proto == "loco":
        folds = [([j for j in range(n) if j != i], [i]) for i in range(n)]
    else:
        k = cfg["train"].get("folds", 10)
        np.random.RandomState(a.seed).shuffle(idx)
        parts = np.array_split(idx, k)
        folds = [([j for j in idx if j not in set(p)], list(p)) for p in parts]
    print(f"Giao thức: {proto} | {len(folds)} fold | chia theo ẢNH | "
          f"{a.variants} biến thể\n")

    all_recs = []
    t0 = time.time()
    for fi, (tr, te) in enumerate(folds, 1):
        head = HeatmapHead(in_ch, hs).to(device)
        enc_f, _ = build_model(a.unfreeze, device, a.grad_ckpt) if fi > 1 else (enc, None)
        net = FineTuneNet(enc_f, head).to(device)

        bb = [p for p in net.enc.parameters() if p.requires_grad]
        groups = [{"params": net.head.parameters(), "lr": a.lr_head}]
        if bb:
            groups.append({"params": bb, "lr": a.lr_backbone})
        opt = torch.optim.AdamW(groups, weight_decay=1e-4)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.epochs)
        scaler = torch.cuda.amp.GradScaler(enabled=a.amp)

        train_stems = [stems[j] for j in tr]
        rng = np.random.RandomState(a.seed * 1000 + fi)
        net.train()
        for ep in range(a.epochs):
            order = [(s, v) for s in train_stems for v in range(a.variants)]
            rng.shuffle(order)
            run, nb = 0.0, 0
            for i in range(0, len(order), a.batch):
                chunk = order[i:i + a.batch]
                xs, ys = [], []
                for s, v in chunk:
                    x, nods = imgs.variant(s, v, rng)
                    xs.append(x)
                    ys.append(build_heatmap(nods, hs, stride, sigma)[None])
                xb = torch.from_numpy(np.stack(xs)).to(device)
                yb = torch.from_numpy(np.stack(ys)).to(device)
                opt.zero_grad(set_to_none=True)
                with torch.cuda.amp.autocast(enabled=a.amp):
                    loss = focal_mse_loss(net(xb), yb)
                if not torch.isfinite(loss):
                    continue
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(net.parameters(), 5.0)
                scaler.step(opt)
                scaler.update()
                run += loss.item(); nb += 1
            sch.step()
            if fi == 1 and (ep + 1) % max(1, a.epochs // 5) == 0:
                print(f"  fold1 epoch {ep+1:3d}/{a.epochs}  loss={run/max(nb,1):.4f}")

        all_recs += eval_stems(net, imgs, [stems[j] for j in te], cfg, device,
                               hs, stride, mm_per_px)
        el = time.time() - t0
        print(f"=== Fold {fi}/{len(folds)} xong | {el/60:.1f} phút | "
              f"còn ~{el/fi*(len(folds)-fi)/60:.1f} phút ===", flush=True)
        del net, head, opt
        if fi > 1:
            del enc_f
        torch.cuda.empty_cache()

    report(all_recs, rates, n, a.tag, {
        "protocol": proto, "seed": a.seed, "unfreeze": a.unfreeze,
        "variants": a.variants, "epochs": a.epochs,
        "lr_backbone": a.lr_backbone, "lr_head": a.lr_head, "amp": a.amp})


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/jsrt_colab.yaml")
    p.add_argument("--unfreeze", type=int, default=4,
                   help="số tầng transformer cuối được mở băng; 0 = đóng băng hoàn toàn")
    p.add_argument("--variants", type=int, default=1,
                   help="1 = không tăng cường; 8 = như raddino_extract2.py")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--lr-backbone", type=float, default=1e-5)
    p.add_argument("--lr-head", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--tag", default="finetune")
    p.add_argument("--vit-size", type=int, default=None)
    p.add_argument("--grad-ckpt", action="store_true",
                   help="đánh đổi tốc độ lấy bộ nhớ; bật nếu GPU nhỏ")
    p.add_argument("--amp", action="store_true",
                   help="KHÔNG NÊN BẬT — xem CLAUDE.md mục 5")
    p.add_argument("--check", action="store_true")
    main(p.parse_args())
