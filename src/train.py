"""Huấn luyện + đánh giá theo giao thức LOCO hoặc k-fold phân tầng subtlety.
Kết thúc mỗi fold: in FROC/CPM và bảng độ nhạy theo subtlety, gộp toàn bộ fold
để ra kết quả cuối.
"""
import os
import json
import argparse
import numpy as np
import pandas as pd
import yaml
from tqdm import tqdm

from .utils import set_seed
from .dataset import JSRTNoduleDataset, collate
from .froc import (extract_peaks, match_detections, compute_froc,
                   sensitivity_by_subtlety, format_subtlety_table)
import torch.multiprocessing as mp
mp.set_sharing_strategy('file_system')

def list_images(cfg):
    img_dir = os.path.join(cfg["data"]["processed_dir"], "images")
    return sorted(f for f in os.listdir(img_dir) if f.endswith(".png"))


def make_folds(cfg, files):
    """k-fold phân tầng theo subtlety chủ đạo mỗi ảnh (nốt khó nhất);
    ảnh normal đưa vào tầng 0. LOCO = mỗi ảnh dương tính là 1 fold test."""
    lp = os.path.join(cfg["data"]["processed_dir"], "labels_processed.csv")
    labels = pd.read_csv(lp) if os.path.exists(lp) else pd.DataFrame()
    def strat(f):
        r = labels[labels["filename"] == f] if len(labels) else []
        return int(r["subtlety"].min()) if len(r) else 0  # nốt khó nhất

    if cfg["train"]["protocol"] == "loco":
        # LOOCV chuẩn: bỏ ra TỪNG ảnh (cả normal lẫn nốt) làm test,
        # train trên tất cả ảnh còn lại. Test cả normal để FROC đếm FP đúng.
        return [([f2 for f2 in files if f2 != f], [f]) for f in files]

    from sklearn.model_selection import StratifiedKFold
    y = np.array([strat(f) for f in files])
    skf = StratifiedKFold(n_splits=cfg["train"]["folds"], shuffle=True,
                          random_state=cfg["seed"])
    return [([files[i] for i in tr], [files[i] for i in te])
            for tr, te in skf.split(files, y)]


def train_one_fold(cfg, train_files, device, verbose=True):
    import torch
    from torch.utils.data import DataLoader
    from .model import HeatmapDetector, build_loss

    ds = JSRTNoduleDataset(cfg, train_files, augment=True)
    nw = cfg["train"]["num_workers"]
    dl = DataLoader(ds, batch_size=cfg["train"]["batch_size"], shuffle=True,
                    num_workers=nw, collate_fn=collate,
                    pin_memory=True, drop_last=True,
                    persistent_workers=False,
                    prefetch_factor=(4 if nw > 0 else None))
    model = HeatmapDetector(cfg).to(device)
    # Giai đoạn 2: nạp backbone đã pre-train trên NIH (nếu có trong config)
    ckpt = cfg["model"].get("pretrained_backbone")
    if ckpt:
        import os
        if os.path.exists(ckpt):
            model.load_pretrained_backbone(ckpt, verbose=verbose)
        else:
            print(f"  CẢNH BÁO: không thấy checkpoint {ckpt} — train từ ImageNet init")
    loss_fn = build_loss(cfg["train"]["loss"])
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["train"]["lr"],
                            weight_decay=cfg["train"]["weight_decay"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, cfg["train"]["epochs"])
    use_amp = cfg["train"]["amp"] and device == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    use_weighted = cfg["train"].get("subtlety_weighted_loss", False)
    model.train()
    for ep in range(cfg["train"]["epochs"]):
        tot = 0.0; nb = 0
        for imgs, hms, wmaps, _ in dl:
            imgs, hms = imgs.to(device), hms.to(device)
            wmaps = wmaps.to(device) if use_weighted else None
            opt.zero_grad()
            if use_amp:
                with torch.amp.autocast("cuda", enabled=True):
                    pred = model(imgs)
                    loss = loss_fn(pred, hms, weight_map=wmaps)
                if not torch.isfinite(loss):
                    continue
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                scaler.step(opt); scaler.update()
            else:
                pred = model(imgs)
                loss = loss_fn(pred, hms, weight_map=wmaps)
                if not torch.isfinite(loss):
                    continue
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                opt.step()
            tot += loss.item(); nb += 1
        sched.step()
        if verbose and (ep + 1) % 10 == 0:
            avg = tot / nb if nb > 0 else float("nan")
            skip_note = "" if nb == len(dl) else f"  (skip {len(dl)-nb} batch!)"
            print(f"    epoch {ep+1:>3}/{cfg['train']['epochs']}  "
                  f"loss={avg:.4f}  lr={sched.get_last_lr()[0]:.2e}{skip_note}")
    return model


def evaluate(cfg, model, test_files, device):
    import torch
    model.eval()
    ds = JSRTNoduleDataset(cfg, test_files, augment=False)
    stride = cfg["model"]["heatmap_stride"]
    e = cfg["eval"]
    all_results, matched_per_image, n_nodules = [], [], 0
    with torch.no_grad():
        for i in range(len(ds)):
            img, _, _, meta = ds[i]
            pred = model(img[None].to(device))[0, 0].cpu().numpy()
            peaks = extract_peaks(pred, e["peak_min_distance"])
            results, unmatched = match_detections(
                peaks, meta["nodules"], stride, meta["mm_per_px"], e["hit_radius_mm"])
            all_results += results
            matched_per_image.append((results, unmatched))
            n_nodules += meta["n_nodules"]
    return all_results, matched_per_image, n_nodules


def main(config):
    with open(config) as f:
        cfg = yaml.safe_load(f)
    set_seed(cfg["seed"])
    try:
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        raise SystemExit("Cần cài torch: pip install -r requirements.txt")

    files = list_images(cfg)
    if not files:
        raise SystemExit("Chưa có ảnh đã xử lý. Chạy: python -m src.preprocess trước.")
    folds = make_folds(cfg, files)
    print(f"Giao thức: {cfg['train']['protocol']} | {len(folds)} fold | device={device}")

def main(config):
    with open(config) as f:
        cfg = yaml.safe_load(f)
    set_seed(cfg["seed"])
    try:
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        raise SystemExit("Cần cài torch: pip install -r requirements.txt")

    files = list_images(cfg)
    if not files:
        raise SystemExit("Chưa có ảnh đã xử lý. Chạy: python -m src.preprocess trước.")
    folds = make_folds(cfg, files)
    proto = cfg["train"]["protocol"]
    print(f"Giao thức: {proto} | {len(folds)} fold | device={device}")

    # thư mục lưu tiến độ từng fold (để resume). Đặt trên Drive nếu cấu hình.
    prog_dir = cfg.get("resume_dir", "runs/folds")
    os.makedirs(prog_dir, exist_ok=True)

    import pickle
    for k, (tr, te) in enumerate(folds):
        fold_file = os.path.join(prog_dir, f"fold_{k:04d}.pkl")
        if os.path.exists(fold_file):
            continue  # đã xong -> bỏ qua (resume)
        # in thưa hơn cho LOOCV để đỡ ngập log
        verbose = (proto != "loco") or (k % 10 == 0)
        if verbose:
            print(f"\n=== Fold {k+1}/{len(folds)}  (train={len(tr)}, test={len(te)}) ===")
        model = train_one_fold(cfg, tr, device, verbose=verbose)
        res, matched, nod = evaluate(cfg, model, te, device)
        with open(fold_file, "wb") as f:
            pickle.dump({"results": res, "matched": matched, "nod": nod,
                         "n_test": len(te)}, f)
        if proto == "loco" and k % 10 == 0 and k > 0:
            print(f"  ...đã xong {k+1}/{len(folds)} fold")
        del model
        if device == "cuda":
            torch.cuda.empty_cache()

    # gộp tất cả fold đã lưu
    agg_results, agg_matched, agg_nod, n_images = [], [], 0, 0
    for k in range(len(folds)):
        fold_file = os.path.join(prog_dir, f"fold_{k:04d}.pkl")
        if not os.path.exists(fold_file):
            print(f"  CẢNH BÁO: thiếu fold {k}, bỏ qua khi gộp")
            continue
        with open(fold_file, "rb") as f:
            d = pickle.load(f)
        agg_results += d["results"]; agg_matched += d["matched"]
        agg_nod += d["nod"]; n_images += d["n_test"]

    fppi, sens, sens_at, cpm = compute_froc(
        agg_results, n_images, agg_nod, cfg["eval"]["fp_rates"])
    print("\n" + "=" * 52)
    print("KẾT QUẢ TỔNG (gộp tất cả fold)")
    print("=" * 52)
    print(f"  Tổng số nốt: {agg_nod} | số ảnh test: {n_images}")
    for r in cfg["eval"]["fp_rates"]:
        print(f"  Sensitivity @ {r:>5} FP/ảnh : {sens_at[r]*100:5.1f}%")
    print(f"  CPM (trung bình FROC)       : {cpm*100:5.1f}%")

    if cfg["eval"]["report_by_subtlety"]:
        by_sub = sensitivity_by_subtlety(agg_matched, 1.0, n_images)
        print("\n  Độ nhạy phân tầng theo subtlety (mục tiêu chính của Hướng B):")
        print(format_subtlety_table(by_sub))

    os.makedirs("runs", exist_ok=True)
    with open("runs/results.json", "w") as f:
        json.dump({"sens_at": sens_at, "cpm": cpm, "n_nodules": agg_nod,
                   "n_images": n_images, "protocol": proto}, f, indent=2)
    print("\nĐã lưu runs/results.json")


def train_full(cfg, device=None, save_path=None):
    """Train 1 model trên TOÀN BỘ ảnh (không chia fold) để phân tích định tính
    (heatmap, Grad-CAM). Không dùng cho báo cáo số liệu — chỉ để trực quan hóa."""
    import torch
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    files = list_images(cfg)
    print(f"Train model đầy đủ trên {len(files)} ảnh để phân tích...")
    model = train_one_fold(cfg, files, device, verbose=True)
    if save_path:
        torch.save({"model_state": model.state_dict(),
                    "cfg_model": cfg["model"]}, save_path)
        print(f"Đã lưu model -> {save_path}")
    return model


def load_full_model(cfg, ckpt_path, device):
    """Nạp lại model đã train bởi train_full."""
    import torch
    from .model import HeatmapDetector
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = HeatmapDetector(cfg).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/baseline.yaml")
    main(ap.parse_args().config)
