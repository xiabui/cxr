"""
raddino_export_head.py — Huấn luyện decoder head trên TOÀN BỘ JSRT rồi lưu lại,
để dùng cho external validation trên VinDr.

Không có rò rỉ dữ liệu vì tập đánh giá (VinDr) hoàn toàn tách biệt với tập
huấn luyện (JSRT). Đây là cách làm chuẩn cho external validation.

    python raddino_export_head.py --config configs/jsrt_10fold.yaml
Kết quả: checkpoints/raddino_head.pt
"""
import os, sys, json, glob, argparse
import numpy as np
import torch, yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.model import focal_mse_loss
from raddino_train3 import HeatmapHead, build_heatmap   # tái dùng, tránh lệch định nghĩa

FEAT_DIR = "features_aug"


def main(a):
    cfg = yaml.safe_load(open(a.config))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    size = cfg["data"]["image_size"]; stride = cfg["model"]["heatmap_stride"]
    sigma = cfg["model"].get("gaussian_sigma", 8); hs = size // stride

    meta = json.load(open(os.path.join(FEAT_DIR, "coords.json")))
    coords = meta["coords"]
    stems = sorted({k.rsplit("_v", 1)[0] for k in coords})
    stems = [s for s in stems if os.path.exists(os.path.join(FEAT_DIR, f"{s}_v0.npy"))]
    probe = np.load(os.path.join(FEAT_DIR, f"{stems[0]}_v0.npy"))
    in_ch = probe.shape[0]
    print(f"{len(stems)} ảnh | đặc trưng {in_ch}×{probe.shape[1]}×{probe.shape[2]}")

    HM = {f"{s}_v0": build_heatmap(coords[f"{s}_v0"], hs, stride, sigma) for s in stems}
    keys = [f"{s}_v0" for s in stems]          # chỉ bản gốc, không tăng cường

    head = HeatmapHead(in_ch, hs).to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=a.lr, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.epochs)
    head.train()
    for ep in range(a.epochs):
        np.random.shuffle(keys); tot, nb = 0.0, 0
        for i in range(0, len(keys), 8):
            kk = keys[i:i + 8]
            x = torch.from_numpy(np.stack([
                np.load(os.path.join(FEAT_DIR, k + ".npy")).astype(np.float32)
                for k in kk])).to(device)
            y = torch.from_numpy(np.stack([HM[k][None] for k in kk])).to(device)
            opt.zero_grad()
            loss = focal_mse_loss(head(x), y)
            if not torch.isfinite(loss):
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), 5.0)
            opt.step(); tot += loss.item(); nb += 1
        sch.step()
        if (ep + 1) % max(1, a.epochs // 6) == 0:
            print(f"  epoch {ep+1:3d}/{a.epochs}  loss={tot/max(nb,1):.4f}")

    os.makedirs("checkpoints", exist_ok=True)
    out = "checkpoints/raddino_head.pt"
    torch.save({"state": head.state_dict(), "in_ch": in_ch, "out_size": hs,
                "size": size, "stride": stride}, out)
    print(f"\nĐã lưu {out}")
    print("LƯU Ý: head này train trên toàn bộ JSRT, CHỈ dùng cho external validation")
    print("       trên dataset khác. Không dùng để báo cáo số trên chính JSRT.")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/jsrt_10fold.yaml")
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--lr", type=float, default=1e-3)
    main(p.parse_args())
