"""
vindr_prepare.py — Chuẩn bị dữ liệu VinDr-CXR cho external validation.

Xử lý ba việc mà script cũ làm sai hoặc bỏ sót:
  1. Tập TRAIN của VinDr được BA bác sĩ gán nhãn độc lập, nên cùng một tổn
     thương xuất hiện tối đa ba hộp trùng nhau. Nếu không gộp thì số nốt thật
     bị đếm gấp ba và độ nhạy bị tính sai. Script gộp hộp theo IoU.
     Tập TEST là đồng thuận của năm bác sĩ nên không cần gộp, và vì vậy được
     khuyến nghị dùng cho external validation.
  2. Ảnh gốc là DICOM, cần áp VOI LUT và xử lý MONOCHROME1 (ảnh đảo màu).
  3. Ghi ra tọa độ hộp đã quy đổi về kích thước ảnh đầu ra.

    python vindr_prepare.py --dicom-dir /data/vindr/test \
        --ann annotations_test.csv --out-dir data/vindr --size 512

Kết quả: data/vindr/images/*.png và data/vindr/vindr_gt.json
"""
import os, json, argparse, glob
from collections import defaultdict

import numpy as np
import cv2
import pandas as pd

NODULE_CLASS = "Nodule/Mass"     # đúng tên nhãn trong VinDr-CXR


def read_dicom(path):
    """Đọc DICOM, áp VOI LUT, chuẩn hóa về uint8 với quy ước phổi sáng."""
    import pydicom
    from pydicom.pixel_data_handlers.util import apply_voi_lut
    ds = pydicom.dcmread(path)
    arr = apply_voi_lut(ds.pixel_array, ds)
    # MONOCHROME1 nghĩa là giá trị lớn = tối, cần đảo
    if str(getattr(ds, "PhotometricInterpretation", "")).strip() == "MONOCHROME1":
        arr = np.max(arr) - arr
    arr = arr.astype(np.float32)
    lo, hi = np.percentile(arr, [0.5, 99.5])
    if hi <= lo:
        lo, hi = float(arr.min()), float(arr.max() + 1e-6)
    arr = np.clip((arr - lo) / (hi - lo), 0, 1)
    return (arr * 255).astype(np.uint8)


def lung_is_bright(img):
    """Ảnh này theo quy ước phổi SÁNG hay phổi TỐI?

    So độ sáng trung bình của hai vùng phổi (hai bên, khoảng giữa chiều cao)
    với vùng trung thất (cột giữa). Trên X-quang ngực chuẩn, phổi chứa khí nên
    TỐI hơn trung thất. JSRT sau tiền xử lý lại theo quy ước NGƯỢC LẠI.

    Vì sao phải kiểm: ảnh PNG dựng sẵn không có header DICOM nên không biết
    MONOCHROME1 hay MONOCHROME2. Nếu đưa ảnh đảo màu vào mô hình đã học trên
    JSRT thì nó không báo lỗi — chỉ lặng lẽ cho ra số thấp, rồi bị đọc nhầm
    thành 'tổng quát hóa kém'.
    """
    h, w = img.shape
    a = img.astype(np.float32)
    y0, y1 = int(0.35 * h), int(0.66 * h)
    phoi = np.concatenate([a[y0:y1, int(0.18 * w):int(0.37 * w)].ravel(),
                           a[y0:y1, int(0.63 * w):int(0.82 * w)].ravel()])
    trung_that = a[y0:y1, int(0.46 * w):int(0.54 * w)].ravel()
    return float(phoi.mean()) > float(trung_that.mean())


def match_convention(img, want_lung_bright=True, verbose=False):
    """Đảo ảnh nếu quy ước sáng/tối không khớp với tập huấn luyện."""
    if lung_is_bright(img) == want_lung_bright:
        return img, False
    if verbose:
        print("  -> đảo màu cho khớp quy ước của JSRT")
    return (255 - img), True


def iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    ua = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def merge_boxes(boxes, iou_thr=0.3, min_agree=1):
    """Gộp hộp của nhiều bác sĩ cho cùng một tổn thương.
    boxes: list (x1,y1,x2,y2). Trả về list hộp trung bình kèm số bác sĩ đồng ý."""
    used = [False] * len(boxes)
    out = []
    for i in range(len(boxes)):
        if used[i]:
            continue
        group = [boxes[i]]; used[i] = True
        for j in range(i + 1, len(boxes)):
            if not used[j] and iou(boxes[i], boxes[j]) >= iou_thr:
                group.append(boxes[j]); used[j] = True
        if len(group) >= min_agree:
            g = np.array(group, dtype=np.float32)
            out.append((g.mean(axis=0).tolist(), len(group)))
    return out


def main(a):
    os.makedirs(os.path.join(a.out_dir, "images"), exist_ok=True)
    df = pd.read_csv(a.ann)
    need = {"image_id", "class_name", "x_min", "y_min", "x_max", "y_max"}
    missing = need - set(df.columns)
    if missing:
        raise SystemExit(f"CSV thiếu cột: {missing}. Có: {list(df.columns)}")

    has_rad = "rad_id" in df.columns
    print(f"CSV: {len(df)} dòng | có cột rad_id: {has_rad}")
    if has_rad:
        print("  -> đây là tập TRAIN (3 bác sĩ/ảnh), sẽ gộp hộp theo IoU")
        print("     khuyến nghị dùng tập TEST (đồng thuận 5 bác sĩ) nếu có")

    nod = df[df["class_name"].astype(str).str.strip() == NODULE_CLASS].copy()
    nod = nod.dropna(subset=["x_min", "y_min", "x_max", "y_max"])
    print(f"Hộp '{NODULE_CLASS}': {len(nod)} trên {nod['image_id'].nunique()} ảnh")

    by_img = defaultdict(list)
    for _, r in nod.iterrows():
        by_img[str(r["image_id"])].append(
            (float(r["x_min"]), float(r["y_min"]), float(r["x_max"]), float(r["y_max"])))

    # danh sách ảnh cần xử lý: ảnh có nốt + (tùy chọn) ảnh bình thường
    ids = set(by_img)
    if a.include_normal > 0:
        allid = [str(x) for x in df["image_id"].unique()]
        normals = [i for i in allid if i not in ids][:a.include_normal]
        ids |= set(normals)
        print(f"Thêm {len(normals)} ảnh không có nốt để đo dương tính giả")

    # bảng kích thước gốc (tuỳ chọn)
    meta_dims = None
    if a.meta_csv:
        md = pd.read_csv(a.meta_csv)
        need2 = {"image_id", "dim0", "dim1"}
        if not need2 <= set(md.columns):
            raise SystemExit(f"{a.meta_csv} thiếu cột {need2 - set(md.columns)}")
        # quy ước numpy: dim0 = số hàng = CHIỀU CAO, dim1 = số cột = CHIỀU RỘNG
        meta_dims = {str(r.image_id): (float(r.dim0), float(r.dim1))
                     for r in md.itertuples()}
        print(f"Kích thước gốc lấy từ {a.meta_csv}: {len(meta_dims)} ảnh")

    gt, done, miss, n_flip, n_oob = {}, 0, 0, 0, 0
    for iid in sorted(ids):
        src = None
        for ext in (".dicom", ".dcm", ".png", ".jpg"):
            p = os.path.join(a.dicom_dir, iid + ext)
            if os.path.exists(p):
                src = p; break
        if src is None:
            miss += 1; continue
        if src.lower().endswith((".dicom", ".dcm")):
            img = read_dicom(src)
        else:
            img = cv2.imread(src, cv2.IMREAD_GRAYSCALE)
        if img is None:
            miss += 1; continue
        if not a.no_match_convention:
            img, flipped = match_convention(img, want_lung_bright=True)
            n_flip += int(flipped)
        # Kích thước GỐC để quy đổi toạ độ hộp. Nếu ảnh trên đĩa đã bị resize
        # sẵn (ví dụ bộ PNG 512x512 dựng sẵn) thì img.shape KHÔNG phải kích
        # thước gốc, và lấy nó sẽ cho tỉ lệ 1:1 — hộp giữ nguyên toạ độ gốc
        # (có thể tới hơn 3000) trên ảnh 512, tức sai hoàn toàn mà không báo lỗi.
        if meta_dims is not None:
            if iid not in meta_dims:
                miss += 1; continue
            h0, w0 = meta_dims[iid]
        else:
            h0, w0 = img.shape
        out_png = os.path.join(a.out_dir, "images", iid + ".png")
        cv2.imwrite(out_png, cv2.resize(img, (a.size, a.size)))

        sx, sy = a.size / w0, a.size / h0
        merged = merge_boxes(by_img.get(iid, []), a.iou_thr, a.min_agree) \
            if has_rad else [(b, 1) for b in by_img.get(iid, [])]
        boxes_scaled = [{"box": [b[0]*sx, b[1]*sy, b[2]*sx, b[3]*sy],
                         "cx": (b[0]+b[2])/2*sx, "cy": (b[1]+b[3])/2*sy,
                         "n_rad": n} for b, n in merged]
        # Hộp phải nằm trong ảnh. Nếu tràn ra ngoài nhiều thì gần như chắc chắn
        # đã gán nhầm chiều cao/chiều rộng, hoặc dùng sai bảng kích thước.
        for bb in boxes_scaled:
            x1, y1, x2, y2 = bb["box"]
            if x2 > a.size * 1.02 or y2 > a.size * 1.02 or x1 < -1 or y1 < -1:
                n_oob += 1
        gt[iid] = boxes_scaled
        done += 1

    with open(os.path.join(a.out_dir, "vindr_gt.json"), "w") as f:
        json.dump({"size": a.size, "source": os.path.basename(a.ann),
                   "merged": has_rad, "gt": gt}, f)

    n_nod = sum(len(v) for v in gt.values())
    n_pos = sum(1 for v in gt.values() if v)
    print(f"\nXong: {done} ảnh ({miss} ảnh không tìm thấy file)")
    if not a.no_match_convention:
        print(f"  đảo màu cho khớp quy ước JSRT (phổi sáng): {n_flip}/{done} ảnh")
        if 0 < n_flip < done:
            print("  CẢNH BÁO: chỉ MỘT PHẦN ảnh bị đảo — nguồn dữ liệu không đồng nhất,"
                  " nên kiểm tra thủ công vài ảnh trước khi tin kết quả")
    print(f"  ảnh có nốt: {n_pos} | tổng số nốt sau gộp: {n_nod}")
    if n_oob:
        print(f"  CẢNH BÁO: {n_oob}/{n_nod} hộp nằm NGOÀI khung ảnh sau khi quy đổi.")
        print("            Nhiều khả năng dim0/dim1 bị gán ngược, hoặc bảng kích"
              " thước không khớp bộ ảnh. KHÔNG dùng kết quả này.")
    else:
        print(f"  mọi hộp đều nằm trong khung {a.size}x{a.size} — tỉ lệ quy đổi hợp lệ")
    print(f"  -> {a.out_dir}/images/ và {a.out_dir}/vindr_gt.json")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--dicom-dir", required=True, help="thư mục ảnh DICOM hoặc PNG")
    p.add_argument("--ann", required=True, help="annotations_test.csv hoặc annotations_train.csv")
    p.add_argument("--out-dir", default="data/vindr")
    p.add_argument("--size", type=int, default=512)
    p.add_argument("--iou-thr", type=float, default=0.3, help="ngưỡng IoU khi gộp hộp")
    p.add_argument("--min-agree", type=int, default=1,
                   help="số bác sĩ tối thiểu phải đồng ý (chỉ dùng cho tập train)")
    p.add_argument("--meta-csv", default=None,
                   help="CSV có image_id,dim0,dim1 = kích thước GỐC. BẮT BUỘC khi "
                        "ảnh trên đĩa đã được resize sẵn, vì khi đó không suy ra "
                        "được kích thước gốc từ chính ảnh.")
    p.add_argument("--no-match-convention", action="store_true",
                   help="TẮT việc tự đảo màu cho khớp quy ước JSRT. Chỉ dùng khi "
                        "bạn chắc chắn nguồn ảnh đã đúng quy ước.")
    p.add_argument("--include-normal", type=int, default=500,
                   help="số ảnh không có nốt đưa vào để đo dương tính giả")
    main(p.parse_args())
