"""
analysis.py
-----------
Phân tích định tính cho chương thực nghiệm:
  - visualize_predictions: vẽ ảnh + target + prediction cho các ca (theo subtlety)
  - gradcam: bản đồ Grad-CAM cho thấy model 'nhìn' vào đâu
  - error_analysis: thống kê nốt bắt được / bỏ sót theo subtlety, kích thước, vị trí

Dùng cho báo cáo, không ảnh hưởng train/eval.
"""
import os
import numpy as np

import torch
import torch.nn.functional as F

from .dataset import JSRTNoduleDataset
from .froc import extract_peaks, match_detections


@torch.no_grad()
def _predict(model, img_tensor, device):
    model.eval()
    return model(img_tensor[None].to(device))[0, 0].cpu().numpy()


def visualize_predictions(cfg, model, files, device, n=6, out_path=None,
                          prefer_hard=True):
    """Vẽ lưới: mỗi hàng 1 ca [ảnh gốc | target | prediction].
    prefer_hard: ưu tiên hiển thị nốt khó (subtlety thấp)."""
    import matplotlib.pyplot as plt
    import pandas as pd

    lp = os.path.join(cfg["data"]["processed_dir"], "labels_processed.csv")
    labels = pd.read_csv(lp)
    # chọn ca có nốt, ưu tiên subtlety thấp
    order = labels.sort_values("subtlety") if prefer_hard else labels
    pick = []
    for fn in order["filename"].unique():
        if fn in files:
            pick.append(fn)
        if len(pick) >= n:
            break

    ds = JSRTNoduleDataset(cfg, pick, augment=False)
    stride = cfg["model"]["heatmap_stride"]
    fig, axes = plt.subplots(len(pick), 3, figsize=(12, 4 * len(pick)))
    if len(pick) == 1:
        axes = axes[None]
    for i in range(len(pick)):
        img, hm, wmap, meta = ds[i]
        ch0 = img.numpy()[0]  # kênh gốc
        pred = _predict(model, img, device)
        peaks = extract_peaks(pred, cfg["eval"]["peak_min_distance"])
        sub = meta["nodules"][0]["subtlety"] if meta["nodules"] else "-"

        axes[i, 0].imshow(ch0, cmap="gray")
        axes[i, 0].set_title(f"Ảnh (nốt sub={sub})")
        # khoanh nốt thật
        for nd in meta["nodules"]:
            axes[i, 0].add_patch(plt.Circle((nd["x"], nd["y"]), 25,
                                            color="lime", fill=False, lw=2))
        axes[i, 1].imshow(hm.numpy()[0], cmap="hot"); axes[i, 1].set_title("Target")
        axes[i, 2].imshow(pred, cmap="hot")
        axes[i, 2].set_title(f"Prediction ({len(peaks)} peaks, max={pred.max():.2f})")
        for ax in axes[i]:
            ax.axis("off")
    plt.tight_layout()
    if out_path:
        plt.savefig(out_path, dpi=120, bbox_inches="tight")
        print("Đã lưu:", out_path)
    plt.show()


class GradCAM:
    """Grad-CAM trên lớp cuối backbone của HeatmapDetector."""

    def __init__(self, model):
        self.model = model
        self.acts = None
        self.grads = None
        # hook vào lớp cuối backbone (layer4 của resnet nằm ở cuối Sequential)
        target = model.backbone[-1]
        target.register_forward_hook(self._fwd)
        target.register_full_backward_hook(self._bwd)

    def _fwd(self, m, inp, out):
        self.acts = out.detach()

    def _bwd(self, m, gin, gout):
        self.grads = gout[0].detach()

    def __call__(self, img_tensor, device):
        self.model.eval()
        x = img_tensor[None].to(device).requires_grad_(True)
        out = self.model(x)
        score = out.max()  # kích hoạt mạnh nhất (nơi model tin có nốt)
        self.model.zero_grad()
        score.backward()
        # Grad-CAM: trọng số kênh = trung bình gradient
        w = self.grads.mean(dim=(2, 3), keepdim=True)
        cam = F.relu((w * self.acts).sum(dim=1, keepdim=True))
        cam = F.interpolate(cam, size=x.shape[2:], mode="bilinear",
                            align_corners=False)[0, 0].cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam


def show_gradcam(cfg, model, files, device, n=4, out_path=None):
    """Vẽ Grad-CAM chồng lên ảnh cho vài ca nốt khó."""
    import matplotlib.pyplot as plt
    import pandas as pd
    import cv2

    lp = os.path.join(cfg["data"]["processed_dir"], "labels_processed.csv")
    labels = pd.read_csv(lp).sort_values("subtlety")
    pick = [fn for fn in labels["filename"].unique() if fn in files][:n]
    ds = JSRTNoduleDataset(cfg, pick, augment=False)
    cam_gen = GradCAM(model)

    fig, axes = plt.subplots(1, len(pick), figsize=(5 * len(pick), 5))
    if len(pick) == 1:
        axes = [axes]
    for i in range(len(pick)):
        img, hm, wmap, meta = ds[i]
        ch0 = img.numpy()[0]
        cam = cam_gen(img, device)
        img8 = (np.clip(ch0, 0, 1) * 255).astype(np.uint8)
        img8 = cv2.cvtColor(img8, cv2.COLOR_GRAY2BGR)
        heat = cv2.applyColorMap((cam * 255).astype(np.uint8), cv2.COLORMAP_JET)
        overlay = cv2.addWeighted(img8, 0.6, heat, 0.4, 0)
        axes[i].imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
        sub = meta["nodules"][0]["subtlety"] if meta["nodules"] else "-"
        axes[i].set_title(f"Grad-CAM (sub={sub})")
        for nd in meta["nodules"]:
            axes[i].add_patch(plt.Circle((nd["x"], nd["y"]), 25,
                                         color="lime", fill=False, lw=2))
        axes[i].axis("off")
    plt.tight_layout()
    if out_path:
        plt.savefig(out_path, dpi=120, bbox_inches="tight")
        print("Đã lưu:", out_path)
    plt.show()


def error_analysis(cfg, model, files, device):
    """Thống kê nốt bắt được/bỏ sót theo subtlety và vị trí (trái/phải, trên/dưới)."""
    import pandas as pd
    from collections import defaultdict

    ds = JSRTNoduleDataset(cfg, files, augment=False)
    stride = cfg["model"]["heatmap_stride"]
    e = cfg["eval"]
    by_sub = defaultdict(lambda: [0, 0])   # sub -> [hit, total]
    by_region = defaultdict(lambda: [0, 0])
    size = cfg["data"]["image_size"]

    for i in range(len(ds)):
        img, hm, wmap, meta = ds[i]
        pred = _predict(model, img, device)
        peaks = extract_peaks(pred, e["peak_min_distance"])
        _, unmatched = match_detections(peaks, meta["nodules"], stride,
                                        meta["mm_per_px"], e["hit_radius_mm"])
        for nd in unmatched:
            sub = nd["subtlety"]
            by_sub[sub][1] += 1
            if nd["hit"]:
                by_sub[sub][0] += 1
            # vị trí: trên/dưới (theo y), trái/phải (theo x)
            vert = "trên" if nd["y"] < size / 2 else "dưới"
            horiz = "trái" if nd["x"] < size / 2 else "phải"
            region = f"{vert}-{horiz}"
            by_region[region][1] += 1
            if nd["hit"]:
                by_region[region][0] += 1

    print("=== Độ nhạy theo subtlety ===")
    for s in sorted(by_sub):
        h, t = by_sub[s]
        print(f"  sub {s}: {h}/{t} = {100*h/t if t else 0:.1f}%")
    print("\n=== Độ nhạy theo vùng giải phẫu ===")
    for r in sorted(by_region):
        h, t = by_region[r]
        print(f"  {r:12}: {h}/{t} = {100*h/t if t else 0:.1f}%")
    return dict(by_sub), dict(by_region)
