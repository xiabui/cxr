"""
bone_suppression.py
-------------------
Đóng góp 1: Khử bóng xương (bone suppression) bằng U-Net PyTorch.

Train trên dataset công khai Kaggle 'hmchuong/xray-bone-shadow-supression':
  - JSRT/     : ảnh gốc (có xương)
  - BSE_JSRT/ : ảnh đã khử xương (Bone Shadow Eliminated), do phần mềm chuyên dụng tạo
Mô hình học ánh xạ: ảnh gốc -> ảnh khử xương.

LƯU Ý PHƯƠNG PHÁP (ghi vào luận văn):
- BSE_JSRT tạo từ chính ảnh JSRT. U-Net chỉ học "cách xóa cấu trúc xương",
  KHÔNG dùng nhãn nốt, nên không rò rỉ thông tin vị trí nốt vào detector.
- Sau khi train, U-Net dùng để sinh kênh khử xương cho MỌI ảnh JSRT ở bước
  preprocess. Detector nhận đầu vào 2 kênh: [ảnh gốc, ảnh khử xương].

Dùng (Colab, chạy bằng biến — xem notebook), hoặc:
    python -m src.bone_suppression --jsrt <dir> --bse <dir> --out <ckpt> --epochs 80
"""
import os
import argparse
import glob
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from .utils import set_seed


# ---------------- U-Net ----------------
class DoubleConv(nn.Module):
    def __init__(self, ci, co):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(ci, co, 3, padding=1), nn.BatchNorm2d(co), nn.ReLU(True),
            nn.Conv2d(co, co, 3, padding=1), nn.BatchNorm2d(co), nn.ReLU(True))

    def forward(self, x):
        return self.net(x)


class BoneSuppressionUNet(nn.Module):
    """U-Net: 1 kênh vào (ảnh gốc) -> 1 kênh ra (ảnh khử xương).
    base=48 (tăng từ 32) để khử xương triệt để hơn."""

    def __init__(self, base=48):
        super().__init__()
        self.d1 = DoubleConv(1, base)
        self.d2 = DoubleConv(base, base * 2)
        self.d3 = DoubleConv(base * 2, base * 4)
        self.d4 = DoubleConv(base * 4, base * 8)
        self.pool = nn.MaxPool2d(2)
        self.up3 = nn.ConvTranspose2d(base * 8, base * 4, 2, stride=2)
        self.u3 = DoubleConv(base * 8, base * 4)
        self.up2 = nn.ConvTranspose2d(base * 4, base * 2, 2, stride=2)
        self.u2 = DoubleConv(base * 4, base * 2)
        self.up1 = nn.ConvTranspose2d(base * 2, base, 2, stride=2)
        self.u1 = DoubleConv(base * 2, base)
        self.out = nn.Conv2d(base, 1, 1)

    def forward(self, x):
        c1 = self.d1(x)
        c2 = self.d2(self.pool(c1))
        c3 = self.d3(self.pool(c2))
        c4 = self.d4(self.pool(c3))
        x = self.u3(torch.cat([self.up3(c4), c3], 1))
        x = self.u2(torch.cat([self.up2(x), c2], 1))
        x = self.u1(torch.cat([self.up1(x), c1], 1))
        return torch.sigmoid(self.out(x))   # ảnh [0,1]


# ---------------- Dataset ----------------
def _match_pairs(jsrt_dir, bse_dir):
    """Ghép cặp ảnh gốc <-> ảnh khử xương theo tên file (bỏ tiền tố khác biệt)."""
    def stem_key(p):
        s = os.path.splitext(os.path.basename(p))[0]
        return s.replace("BSE_", "").replace("bse_", "").lower()
    jsrt = {stem_key(p): p for p in glob.glob(os.path.join(jsrt_dir, "*.png"))}
    bse = {stem_key(p): p for p in glob.glob(os.path.join(bse_dir, "*.png"))}
    keys = sorted(set(jsrt) & set(bse))
    return [(jsrt[k], bse[k]) for k in keys]


class BSEPairDataset(Dataset):
    def __init__(self, pairs, size=512, augment=False):
        self.pairs = pairs
        self.size = size
        self.augment = augment

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, i):
        import cv2
        src_p, tgt_p = self.pairs[i]
        src = cv2.imread(src_p, cv2.IMREAD_GRAYSCALE)
        tgt = cv2.imread(tgt_p, cv2.IMREAD_GRAYSCALE)
        src = cv2.resize(src, (self.size, self.size)).astype(np.float32) / 255.0
        tgt = cv2.resize(tgt, (self.size, self.size)).astype(np.float32) / 255.0
        if self.augment and np.random.rand() < 0.5:
            src = np.ascontiguousarray(src[:, ::-1])
            tgt = np.ascontiguousarray(tgt[:, ::-1])
        return (torch.from_numpy(src[None]).float(),
                torch.from_numpy(tgt[None]).float())


# ---------------- Loss: MSE + MS-SSIM ----------------
def gaussian_window(size, sigma, device):
    coords = torch.arange(size, dtype=torch.float32, device=device) - size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = (g / g.sum()).unsqueeze(0)
    return (g.t() @ g).unsqueeze(0).unsqueeze(0)


def ssim(x, y, window, eps=1e-6):
    mu_x = F.conv2d(x, window, padding=window.shape[-1] // 2)
    mu_y = F.conv2d(y, window, padding=window.shape[-1] // 2)
    mu_x2, mu_y2, mu_xy = mu_x * mu_x, mu_y * mu_y, mu_x * mu_y
    sx = F.conv2d(x * x, window, padding=window.shape[-1] // 2) - mu_x2
    sy = F.conv2d(y * y, window, padding=window.shape[-1] // 2) - mu_y2
    sxy = F.conv2d(x * y, window, padding=window.shape[-1] // 2) - mu_xy
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    s = ((2 * mu_xy + c1) * (2 * sxy + c2)) / \
        ((mu_x2 + mu_y2 + c1) * (sx + sy + c2) + eps)
    return s.mean()


def bs_loss(pred, target, window):
    """MSE + (1 - SSIM): kết hợp sai số pixel và cấu trúc, chuẩn cho khử xương."""
    return F.mse_loss(pred, target) + 0.5 * (1 - ssim(pred, target, window))


# ---------------- Train ----------------
def train_bone_suppression(jsrt_dir, bse_dir, out_ckpt, size=512, epochs=80,
                           batch_size=4, lr=1e-3, val_frac=0.1, seed=42,
                           device=None, verbose=True):
    set_seed(seed)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    pairs = _match_pairs(jsrt_dir, bse_dir)
    if not pairs:
        raise SystemExit(f"Không ghép được cặp ảnh từ {jsrt_dir} và {bse_dir}")
    if verbose:
        print(f"Số cặp ảnh (gốc <-> khử xương): {len(pairs)}")

    n_val = max(1, int(len(pairs) * val_frac))
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(pairs))
    val_p = [pairs[i] for i in idx[:n_val]]
    tr_p = [pairs[i] for i in idx[n_val:]]

    dl_tr = DataLoader(BSEPairDataset(tr_p, size, augment=True),
                       batch_size=batch_size, shuffle=True, num_workers=4,
                       pin_memory=True, drop_last=True)
    dl_va = DataLoader(BSEPairDataset(val_p, size, augment=False),
                       batch_size=batch_size, shuffle=False, num_workers=2,
                       pin_memory=True)

    model = BoneSuppressionUNet().to(device)
    window = gaussian_window(11, 1.5, device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)

    best_val = float("inf")
    for ep in range(1, epochs + 1):
        model.train(); tr = 0.0
        for src, tgt in dl_tr:
            src, tgt = src.to(device), tgt.to(device)
            opt.zero_grad()
            loss = bs_loss(model(src), tgt, window)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step(); tr += loss.item()
        sched.step()

        model.eval(); va = 0.0
        with torch.no_grad():
            for src, tgt in dl_va:
                src, tgt = src.to(device), tgt.to(device)
                va += bs_loss(model(src), tgt, window).item()
        va /= max(1, len(dl_va))
        if verbose and ep % 10 == 0:
            print(f"[{ep:03d}/{epochs}] train {tr/len(dl_tr):.4f} | val {va:.4f}")
        if va < best_val:
            best_val = va
            torch.save({"model_state": model.state_dict(), "size": size,
                        "val_loss": va}, out_ckpt)
    if verbose:
        print(f"Xong. Val loss tốt nhất: {best_val:.4f} -> {out_ckpt}")
    return out_ckpt


@torch.no_grad()
def suppress_image(model, img01, device, size=512):
    """Áp U-Net lên 1 ảnh [0,1] HxW -> ảnh khử xương [0,1] cùng kích thước."""
    import cv2
    h0, w0 = img01.shape
    x = cv2.resize(img01.astype(np.float32), (size, size))
    x = torch.from_numpy(x[None, None]).float().to(device)
    y = model(x)[0, 0].cpu().numpy()
    return cv2.resize(y, (w0, h0))


def estimate_lung_mask(img01):
    """Sinh lung mask thô bằng ngưỡng Otsu + hình thái học.
    Phổi = vùng sáng (khí) trong lồng ngực. Trả về mask nhị phân [0,1]."""
    import cv2
    img8 = (np.clip(img01, 0, 1) * 255).astype(np.uint8)
    img8 = cv2.GaussianBlur(img8, (5, 5), 0)
    # phổi sáng hơn trung thất/xương -> ngưỡng Otsu rồi lấy vùng sáng
    _, th = cv2.threshold(img8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # đóng lỗ + bỏ nhiễu nhỏ
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, k)
    th = cv2.morphologyEx(th, cv2.MORPH_OPEN, k)
    # giữ 2 thành phần liên thông lớn nhất (2 lá phổi), bỏ rìa
    n, lab, stats, _ = cv2.connectedComponentsWithStats(th)
    if n > 1:
        areas = stats[1:, cv2.CC_STAT_AREA]
        order = np.argsort(-areas)[:2]  # 2 vùng lớn nhất
        mask = np.zeros_like(th)
        for idx in order:
            mask[lab == idx + 1] = 255
        th = mask
    # làm mềm biên để blend mượt
    th = cv2.GaussianBlur(th, (31, 31), 0).astype(np.float32) / 255.0
    return th


def suppress_image_masked(model, img01, device, size=512, blend=0.7):
    """Khử xương CHỈ trong vùng phổi, và blend có kiểm soát để giảm artifact.
    kênh ra = trong phổi: blend*khử_xương + (1-blend)*gốc; ngoài phổi: giữ gốc.
    Đây là bản tinh chỉnh chống nhiễu viền + khử quá tay (A+B+lung mask)."""
    bs = suppress_image(model, img01, device, size)
    bs = np.clip(bs, 0, 1)
    mask = estimate_lung_mask(img01)
    # trong phổi: pha khử xương với gốc theo blend; ngoài phổi: giữ gốc
    inside = blend * bs + (1 - blend) * img01
    out = mask * inside + (1 - mask) * img01
    return np.clip(out, 0, 1)


def load_bs_model(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = BoneSuppressionUNet().to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, ckpt.get("size", 512)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsrt", required=True, help="thư mục ảnh gốc JSRT")
    ap.add_argument("--bse", required=True, help="thư mục ảnh khử xương BSE_JSRT")
    ap.add_argument("--out", required=True, help="đường dẫn lưu checkpoint")
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-3)
    args = ap.parse_args()
    train_bone_suppression(args.jsrt, args.bse, args.out, args.size,
                           args.epochs, args.batch_size, args.lr)
