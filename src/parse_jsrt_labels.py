"""
parse_jsrt_labels.py
--------------------
Chuyển nhãn JSRT (bản Kaggle raddar/nodules-in-chest-xrays-jsrt HOẶC file gốc
CNNDAT_EN.TXT) sang format labels.csv mà pipeline cần:

    filename,x,y,diameter_mm,subtlety,malignant

Thiết kế TỰ THÍCH ỨNG: tự dò cấu trúc file nhãn thực tế thay vì hard-code tên
cột, vì bản Kaggle có thể đóng gói theo nhiều cách. Chạy xong nên KIỂM TRA vài
dòng đầu bằng mắt để chắc ánh xạ đúng.

Dùng:
    python parse_jsrt_labels.py --src /path/to/jsrt_dir --out data/jsrt/labels.csv
"""
import os
import re
import argparse
import glob
import numpy as np
import pandas as pd


# --- Ánh xạ subtlety chữ -> số (thang gốc JSRT, 1 = khó thấy nhất) ---
SUBTLETY_TEXT = {
    "extremely subtle": 1, "extremely-subtle": 1, "extremelysubtle": 1,
    "very subtle": 2, "very-subtle": 2, "verysubtle": 2,
    "subtle": 3, "relatively obvious": 4, "relatively-obvious": 4,
    "obvious": 5, "very obvious": 5,
}

# Từ khoá để nhận diện ý nghĩa từng cột (không phân biệt hoa thường)
COL_HINTS = {
    "filename": ["file", "name", "image", "img", "study"],
    "x":        ["x", "coord_x", "posx", "cx", "x_coord", "x_position"],
    "y":        ["y", "coord_y", "posy", "cy", "y_coord", "y_position"],
    "diameter_mm": ["diam", "size", "mm", "effective"],
    "subtlety": ["subtle", "subtlety", "degree", "difficulty"],
    "malignant": ["malign", "diagnosis", "benign", "patholog", "type"],
}


def guess_column(target, columns):
    """Tìm cột trong 'columns' khớp nhất với 'target' theo từ khoá."""
    hints = COL_HINTS[target]
    low = {c: c.lower() for c in columns}
    # ưu tiên khớp chính xác trước
    for c, cl in low.items():
        if cl in hints:
            return c
    # rồi khớp chứa từ khoá
    for c, cl in low.items():
        if any(h in cl for h in hints):
            return c
    return None


def normalize_subtlety(val):
    """Đưa subtlety về số 1..5. Nhận cả số lẫn chữ."""
    if pd.isna(val):
        return 3
    s = str(val).strip().lower()
    if s in SUBTLETY_TEXT:
        return SUBTLETY_TEXT[s]
    m = re.search(r"[1-5]", s)
    if m:
        return int(m.group())
    return 3  # mặc định trung bình nếu không rõ


def normalize_malignant(val):
    """1 = ác tính (malignant), 0 = lành tính (benign). Mặc định 1 vì JSRT
    tập nốt chủ yếu là ca có nốt cần phát hiện."""
    if pd.isna(val):
        return 1
    s = str(val).strip().lower()
    if "benign" in s:
        return 0
    if "malig" in s:
        return 1
    if s in ("0", "1"):
        return int(s)
    return 1


def parse_original_txt(path):
    """Parse file gốc CNNDAT_EN.TXT của JSRT (dạng bảng ngăn cách khoảng trắng).
    Cấu trúc điển hình mỗi dòng ca có nốt:
      filename  degree_of_subtlety  size(mm)  age  sex  x  y  diagnosis ...
    Vì định dạng gốc có thể lệch, ta parse linh hoạt theo token."""
    rows = []
    with open(path, encoding="latin-1") as f:
        for line in f:
            line = line.strip()
            if not line or line.lower().startswith(("filename", "#", "case")):
                continue
            parts = re.split(r"\s{1,}", line)
            if len(parts) < 6:
                continue
            fn = parts[0]
            if not re.search(r"jpcl|jpcn|\.img|\.png", fn, re.I):
                continue
            # cố gắng rút subtlety (chữ hoặc số) và các số toạ độ/kích thước
            subtlety = normalize_subtlety(
                next((p for p in parts[1:] if p.lower() in SUBTLETY_TEXT
                      or re.fullmatch(r"[1-5]", p)), "3"))
            nums = [float(p) for p in parts if re.fullmatch(r"-?\d+(\.\d+)?", p)]
            # heuristic: hai số lớn nhất ~ toạ độ x,y (ảnh 2048); một số nhỏ ~ đường kính mm
            diam = next((n for n in nums if 3 <= n <= 60), np.nan)
            coords = [n for n in nums if 0 <= n <= 2048]
            x, y = (coords[-2], coords[-1]) if len(coords) >= 2 else (np.nan, np.nan)
            malignant = normalize_malignant(
                "malignant" if "malignant" in line.lower()
                else ("benign" if "benign" in line.lower() else "1"))
            stem = os.path.splitext(os.path.basename(fn))[0]
            rows.append({"filename": stem + ".png", "x": x, "y": y,
                         "diameter_mm": diam, "subtlety": subtlety,
                         "malignant": malignant})
    return pd.DataFrame(rows)


def parse_csv_adaptive(path):
    """Đọc một CSV bất kỳ và ánh xạ cột về format chuẩn bằng dò từ khoá."""
    df = pd.read_csv(path)
    print(f"  [đọc] {os.path.basename(path)} — các cột gốc: {list(df.columns)}")
    mapping = {}
    for target in ["filename", "x", "y", "diameter_mm", "subtlety", "malignant"]:
        col = guess_column(target, df.columns)
        mapping[target] = col
    print(f"  [ánh xạ] {mapping}")

    out = pd.DataFrame()
    # filename bắt buộc
    if mapping["filename"] is None:
        raise ValueError("Không tìm được cột filename — hãy mở CSV kiểm tra thủ công.")
    out["filename"] = df[mapping["filename"]].astype(str).apply(
        lambda s: os.path.splitext(os.path.basename(s))[0] + ".png")
    for col in ["x", "y", "diameter_mm"]:
        out[col] = pd.to_numeric(df[mapping[col]], errors="coerce") \
            if mapping[col] else np.nan
    out["subtlety"] = (df[mapping["subtlety"]].apply(normalize_subtlety)
                       if mapping["subtlety"] else 3)
    out["malignant"] = (df[mapping["malignant"]].apply(normalize_malignant)
                        if mapping["malignant"] else 1)
    # bỏ dòng không có toạ độ (ảnh normal không cần dòng nào trong labels.csv)
    out = out.dropna(subset=["x", "y"]).reset_index(drop=True)
    return out


def find_label_file(src):
    """Tìm file nhãn trong thư mục src: ưu tiên CSV, rồi tới CNNDAT_EN.TXT."""
    csvs = glob.glob(os.path.join(src, "**", "*.csv"), recursive=True)
    # loại các CSV rõ ràng không phải nhãn
    csvs = [c for c in csvs if not re.search(r"sample_submission|readme", c, re.I)]
    if csvs:
        # ưu tiên file có 'nodule'/'label'/'annotation' trong tên
        pref = [c for c in csvs if re.search(r"nodul|label|annot|jsrt", c, re.I)]
        return (pref[0] if pref else csvs[0]), "csv"
    txts = glob.glob(os.path.join(src, "**", "CNNDAT*.TXT"), recursive=True) \
        + glob.glob(os.path.join(src, "**", "*.txt"), recursive=True)
    if txts:
        return txts[0], "txt"
    return None, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="thư mục JSRT đã tải/giải nén")
    ap.add_argument("--out", default="data/jsrt/labels.csv")
    args = ap.parse_args()

    label_file, kind = find_label_file(args.src)
    if label_file is None:
        raise SystemExit(f"Không thấy file nhãn (.csv hay .txt) trong {args.src}")
    print(f"Dùng file nhãn: {label_file}  (loại: {kind})")

    df = parse_csv_adaptive(label_file) if kind == "csv" \
        else parse_original_txt(label_file)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    df.to_csv(args.out, index=False)

    print("\n===== KẾT QUẢ =====")
    print(f"Số nốt: {len(df)} | số ảnh có nốt: {df['filename'].nunique()}")
    print("Phân bố subtlety (1=khó nhất ... 5=rõ):")
    print(df["subtlety"].value_counts().sort_index().to_string())
    print(f"\nĐã ghi -> {args.out}")
    print("\n>>> KIỂM TRA 5 dòng đầu (đối chiếu bằng mắt xem x,y,subtlety hợp lý):")
    print(df.head().to_string(index=False))
    print("\nLƯU Ý: nếu x,y hoặc subtlety trông sai, mở file nhãn gốc xem cột thật")
    print("rồi chỉnh COL_HINTS hoặc SUBTLETY_TEXT ở đầu script cho khớp.")


if __name__ == "__main__":
    main()
