"""Dataset JSRT cho detection dạng heatmap.
Mỗi ảnh -> heatmap Gaussian quanh tâm mỗi nốt. Giữ lại subtlety & mm/px
để đánh giá FROC phân tầng ở bước sau.
"""
import os
import json
import numpy as np
import pandas as pd

try:
    import torch
    from torch.utils.data import Dataset
except ImportError:
    Dataset = object


def gaussian_2d(h, w, cx, cy, sigma):
    ys, xs = np.ogrid[:h, :w]
    return np.exp(-((xs - cx) ** 2 + (ys - cy) ** 2) / (2 * sigma ** 2))


class JSRTNoduleDataset(Dataset):
    """Trả về (image[1,H,W], heatmap[1,h,w], meta). meta chứa nốt gốc + subtlety."""

    def __init__(self, cfg, filenames, augment=False):
        self.cfg = cfg
        d = cfg["data"]
        self.img_dir = os.path.join(d["processed_dir"], "images")
        # thư mục kênh khử xương (nếu dùng input 2 kênh). Có = bật, không = baseline.
        self.bs_dir = os.path.join(d["processed_dir"], "images_bs")
        self.in_channels = cfg["model"].get("in_channels", 1)
        self.size = d["image_size"]
        self.stride = cfg["model"]["heatmap_stride"]
        self.sigma = cfg["model"]["gaussian_sigma"]
        self.augment = augment
        lp = os.path.join(d["processed_dir"], "labels_processed.csv")
        self.labels = pd.read_csv(lp) if os.path.exists(lp) else pd.DataFrame(
            columns=["filename", "x", "y", "diameter_mm", "subtlety", "malignant"])
        self.files = filenames
        # mm/px trên ảnh đã xử lý (JSRT gốc 0.175 mm/px @2048)
        with open(os.path.join(d["processed_dir"], "transforms.json")) as f:
            self.tf = json.load(f)

    def __len__(self):
        return len(self.files)

    def mm_per_px(self, stem):
        # tìm transform theo tên gốc (bất kể đuôi)
        for k, v in self.tf.items():
            if os.path.splitext(k)[0] == os.path.splitext(stem)[0]:
                return 0.175 / v["scale"]
        return 0.175

    def __getitem__(self, i):
        import cv2
        stem = self.files[i]
        img = cv2.imread(os.path.join(self.img_dir, stem), cv2.IMREAD_GRAYSCALE)
        img = img.astype(np.float32) / 255.0

        # kênh 2: ảnh khử xương (nếu input 2 kênh)
        img_bs = None
        if self.in_channels == 2:
            bs_path = os.path.join(self.bs_dir, stem)
            bs = cv2.imread(bs_path, cv2.IMREAD_GRAYSCALE)
            if bs is None:
                img_bs = img.copy()  # fallback: chưa có ảnh khử xương -> lặp kênh gốc
            else:
                img_bs = bs.astype(np.float32) / 255.0

        rows = self.labels[self.labels["filename"] == stem]

        if self.augment and np.random.rand() < 0.5:
            img = np.ascontiguousarray(img[:, ::-1])  # lật ngang (an toàn cho nốt)
            if img_bs is not None:
                img_bs = np.ascontiguousarray(img_bs[:, ::-1])
            rows = rows.copy()
            rows["x"] = self.size - 1 - rows["x"]

        hs = self.size // self.stride
        hm = np.zeros((hs, hs), np.float32)
        wmap = np.ones((hs, hs), np.float32)   # weight map cho loss (đóng góp 3)
        # trọng số theo subtlety: nốt càng khó (sub thấp) -> trọng số càng cao
        sub_weight = {1: 3.0, 2: 2.5, 3: 2.0, 4: 1.5, 5: 1.0}
        nodules = []
        for _, r in rows.iterrows():
            cx, cy = r["x"] / self.stride, r["y"] / self.stride
            sub = int(r["subtlety"])
            if 0 <= cx < hs and 0 <= cy < hs:
                sigma_hm = self.sigma / self.stride     # sigma tính trên lưới heatmap
                sigma_hm = max(1.0, sigma_hm)
                g = gaussian_2d(hs, hs, cx, cy, sigma_hm)
                hm = np.maximum(hm, g)
                # ÉP pixel tâm (làm tròn) = đúng 1.0 để target.eq(1) có điểm dương
                ix, iy = int(round(cx)), int(round(cy))
                if 0 <= ix < hs and 0 <= iy < hs:
                    hm[iy, ix] = 1.0
                # gán trọng số vùng quanh nốt theo subtlety (lấy max nếu chồng)
                w = sub_weight.get(sub, 1.0)
                region = g > 0.1
                wmap[region] = np.maximum(wmap[region], w)
            nodules.append({"x": float(r["x"]), "y": float(r["y"]),
                            "subtlety": sub,
                            "diameter_mm": float(r.get("diameter_mm", np.nan))})

        meta = {"filename": stem, "nodules": nodules,
                "mm_per_px": self.mm_per_px(stem), "n_nodules": len(nodules)}

        # xếp kênh: 1 kênh (baseline) hoặc 2 kênh [gốc, khử xương]
        if self.in_channels == 2:
            stacked = np.stack([img, img_bs], axis=0)  # [2,H,W]
        else:
            stacked = img[None]                          # [1,H,W]

        if Dataset is object:  # môi trường không có torch
            return stacked, hm[None], wmap[None], meta
        return (torch.from_numpy(stacked).float(),
                torch.from_numpy(hm[None]).float(),
                torch.from_numpy(wmap[None]).float(), meta)


def collate(batch):
    imgs = torch.stack([b[0] for b in batch])
    hms = torch.stack([b[1] for b in batch])
    wmaps = torch.stack([b[2] for b in batch])
    metas = [b[3] for b in batch]
    return imgs, hms, wmaps, metas
