"""
pretrain_nih.py
---------------
Giai đoạn 1: Pre-train backbone trên NIH ChestX-ray14 bằng bài toán phân loại
nhị phân "có nốt/khối (Nodule OR Mass)" vs "không". Mục tiêu KHÔNG phải đạt SOTA
classification, mà để backbone HỌC ĐẶC TRƯNG ẢNH PHỔI từ dữ liệu lớn, rồi
chuyển sang giai đoạn 2 (detection trên JSRT).

Điểm mấu chốt: backbone dùng lại HỆT make_backbone() trong model.py, nên
checkpoint lưu ra sẽ nạp được trực tiếp vào HeatmapDetector ở giai đoạn 2.

Tự lo mọi bước cho NIH (không cần preprocess riêng):
  - đọc Data_Entry_2017.csv -> nhãn nhị phân
  - chia train/val theo bệnh nhân (patient-wise, tránh rò rỉ)
  - load ảnh on-the-fly + augment nhẹ
  - train, theo dõi AUC, lưu backbone tốt nhất

Dùng (trên Colab):
    python -m src.pretrain_nih \
        --images /content/nih_images \
        --labels /content/drive/MyDrive/.../Data_Entry_2017.csv \
        --out    /content/drive/MyDrive/.../checkpoints \
        --backbone resnet34 --epochs 10 --batch-size 128 --img-size 224
"""
import os
import argparse
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupShuffleSplit
from tqdm.auto import tqdm

from .model import make_backbone
from .utils import set_seed


# ---------------- Dataset ----------------
class NIHClsDataset(Dataset):
    """Phân loại nhị phân: 1 nếu ảnh có 'Nodule' hoặc 'Mass', ngược lại 0."""

    def __init__(self, df, images_dir, img_size=224, train=False):
        self.df = df.reset_index(drop=True)
        self.images_dir = images_dir
        self.img_size = img_size
        self.train = train

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        import cv2
        row = self.df.iloc[i]
        path = os.path.join(self.images_dir, row["filename"])
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            # ảnh lỗi/thiếu -> trả ảnh đen, nhãn 0 (hiếm, không ảnh hưởng lớn)
            img = np.zeros((self.img_size, self.img_size), np.uint8)
        img = cv2.resize(img, (self.img_size, self.img_size))
        img = img.astype(np.float32) / 255.0

        if self.train:
            if np.random.rand() < 0.5:
                img = np.ascontiguousarray(img[:, ::-1])          # lật ngang
            if np.random.rand() < 0.3:
                d = np.random.uniform(-0.1, 0.1)
                img = np.clip(img + d, 0, 1)                       # đổi độ sáng
        # chuẩn hoá về mean/std ~ ImageNet grayscale
        img = (img - 0.5) / 0.25
        x = torch.from_numpy(img[None]).float()
        y = torch.tensor([row["label"]], dtype=torch.float32)
        return x, y


# ---------------- Model ----------------
class NIHClassifier(nn.Module):
    """backbone (giống model.py) + global pool + head nhị phân."""

    def __init__(self, backbone_name="resnet34", pretrained=True):
        super().__init__()
        self.backbone, feat_ch = make_backbone(backbone_name, pretrained)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(
            nn.Flatten(), nn.Dropout(0.2), nn.Linear(feat_ch, 1))

    def forward(self, x):
        f = self.backbone(x)
        return self.head(self.pool(f))     # logits [B,1]


# ---------------- Nhãn NIH ----------------
def build_labels(labels_csv, images_dir):
    """Đọc Data_Entry_2017.csv -> DataFrame(filename, label, patient_id).
    label = 1 nếu 'Finding Labels' chứa Nodule hoặc Mass."""
    df = pd.read_csv(labels_csv)
    # tên cột NIH chuẩn: 'Image Index', 'Finding Labels', 'Patient ID'
    col_img = next((c for c in df.columns if c.lower().replace(" ", "") in
                    ("imageindex", "filename", "image")), df.columns[0])
    col_find = next((c for c in df.columns if "finding" in c.lower()), None)
    col_pat = next((c for c in df.columns if "patient" in c.lower()
                    and "id" in c.lower()), None)

    def has_nodule_mass(s):
        s = str(s).lower()
        return int(("nodule" in s) or ("mass" in s))

    out = pd.DataFrame({
        "filename": df[col_img].astype(str),
        "label": df[col_find].apply(has_nodule_mass) if col_find else 0,
        "patient_id": df[col_pat] if col_pat else range(len(df)),
    })
    # chỉ giữ ảnh thực sự tồn tại trong thư mục (phòng khi tải thiếu)
    have = set(os.listdir(images_dir))
    out = out[out["filename"].isin(have)].reset_index(drop=True)
    return out


# ---------------- Train ----------------
def run_epoch(model, loader, device, criterion, optimizer=None, scaler=None):
    train = optimizer is not None
    model.train() if train else model.eval()
    losses, ys, ps = [], [], []
    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for x, y in tqdm(loader, leave=False, desc="train" if train else "val"):
            x, y = x.to(device), y.to(device)
            if train:
                optimizer.zero_grad()
                with torch.cuda.amp.autocast(enabled=scaler is not None):
                    logit = model(x)
                    loss = criterion(logit, y)
                scaler.scale(loss).backward()
                scaler.step(optimizer); scaler.update()
            else:
                with torch.cuda.amp.autocast(enabled=scaler is not None):
                    logit = model(x)
                    loss = criterion(logit, y)
            losses.append(loss.item())
            ys.append(y.detach().cpu().numpy())
            ps.append(torch.sigmoid(logit).detach().cpu().numpy())
    ys, ps = np.concatenate(ys), np.concatenate(ps)
    auc = roc_auc_score(ys, ps) if len(np.unique(ys)) > 1 else float("nan")
    return float(np.mean(losses)), auc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True, help="thư mục ảnh NIH đã giải nén")
    ap.add_argument("--labels", required=True, help="Data_Entry_2017.csv")
    ap.add_argument("--out", required=True, help="thư mục lưu checkpoint")
    ap.add_argument("--backbone", default="resnet34")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--img-size", type=int, default=224)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    set_seed(args.seed)
    os.makedirs(args.out, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else \
        ("mps" if torch.backends.mps.is_available() else "cpu")
    print("device:", device)

    # nhãn + chia theo bệnh nhân (tránh cùng 1 bệnh nhân ở cả train lẫn val)
    data = build_labels(args.labels, args.images)
    print(f"Tổng ảnh dùng được: {len(data)} | "
          f"dương tính (nodule/mass): {int(data['label'].sum())} "
          f"({data['label'].mean()*100:.1f}%)")
    gss = GroupShuffleSplit(n_splits=1, test_size=args.val_frac,
                            random_state=args.seed)
    tr_idx, va_idx = next(gss.split(data, groups=data["patient_id"]))
    df_tr, df_va = data.iloc[tr_idx], data.iloc[va_idx]
    print(f"train: {len(df_tr)} | val: {len(df_va)}")

    ds_tr = NIHClsDataset(df_tr, args.images, args.img_size, train=True)
    ds_va = NIHClsDataset(df_va, args.images, args.img_size, train=False)
    dl_tr = DataLoader(ds_tr, batch_size=args.batch_size, shuffle=True,
                       num_workers=args.workers, pin_memory=True, drop_last=True)
    dl_va = DataLoader(ds_va, batch_size=args.batch_size, shuffle=False,
                       num_workers=args.workers, pin_memory=True)

    model = NIHClassifier(args.backbone, pretrained=True).to(device)
    # mất cân bằng: nốt/mass là thiểu số -> pos_weight cho BCE
    pos = max(1, int(df_tr["label"].sum()))
    neg = max(1, len(df_tr) - pos)
    pos_weight = torch.tensor([neg / pos], device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)
    scaler = torch.cuda.amp.GradScaler(enabled=(device == "cuda"))

    best_auc = -1.0
    for ep in range(1, args.epochs + 1):
        tr_loss, tr_auc = run_epoch(model, dl_tr, device, criterion,
                                    optimizer, scaler)
        va_loss, va_auc = run_epoch(model, dl_va, device, criterion,
                                    None, scaler)
        sched.step()
        print(f"[{ep:02d}/{args.epochs}] "
              f"train loss {tr_loss:.4f} auc {tr_auc:.4f} | "
              f"val loss {va_loss:.4f} auc {va_auc:.4f}")

        # LƯU CHECKPOINT BACKBONE (chỉ backbone, để giai đoạn 2 nạp) khi val AUC cải thiện
        if not np.isnan(va_auc) and va_auc > best_auc:
            best_auc = va_auc
            ckpt = os.path.join(args.out, f"nih_backbone_{args.backbone}.pt")
            torch.save({
                "backbone_state": model.backbone.state_dict(),
                "backbone_name": args.backbone,
                "img_size": args.img_size,
                "val_auc": va_auc,
                "epoch": ep,
            }, ckpt)
            print(f"    -> lưu checkpoint tốt nhất (val AUC {va_auc:.4f}) tại {ckpt}")

    print(f"\nXong. Val AUC tốt nhất: {best_auc:.4f}")
    print("Checkpoint backbone đã lưu ở:", args.out)
    print("Giai đoạn 2 sẽ nạp trọng số này vào HeatmapDetector để fine-tune JSRT.")


if __name__ == "__main__":
    main()
