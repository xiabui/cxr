"""
recover_scores.py — Dựng lại runs/scores_<tag>.json từ các fold pickle đã lưu.

Vì sao cần: src/train.py lưu từng fold ra pickle để resume được, rồi mới gộp và
ghi JSON ở cuối. Nếu bước gộp chết giữa chừng thì JSON không có, nhưng PICKLE
VẪN ĐỦ DỮ LIỆU. File này dựng lại mà không phải huấn luyện lại — mỗi lần chạy
ResNet tốn hơn hai giờ nên đây là khác biệt giữa vài giây và vài chục giờ.

    python recover_scores.py --fold-dir runs_resnet_base/resnet_base_s0 \
        --tag resnet_base_s0 --seed 0
"""
import os
import json
import glob
import pickle
import argparse


def main(a):
    files = sorted(glob.glob(os.path.join(a.fold_dir, "fold_*.pkl")))
    if not files:
        raise SystemExit(f"Không thấy fold_*.pkl trong {a.fold_dir}")

    recs, n_nod, n_img = [], 0, 0
    for f in files:
        with open(f, "rb") as fh:
            d = pickle.load(fh)
        n_nod += d["nod"]; n_img += d["n_test"]
        for item in d["matched"]:
            results, unmatched = item[0], item[1]
            name = item[2] if len(item) > 2 else f"img{len(recs):04d}"
            recs.append({
                "image": os.path.splitext(os.path.basename(str(name)))[0],
                "nodules": [{"score": float(nd.get("hit_score") or 0.0),
                             "subtlety": int(nd["subtlety"])} for nd in unmatched],
                "fps": [float(s) for s, tp in results if not tp],
            })

    if len(recs) != n_img:
        print(f"CẢNH BÁO: {len(recs)} bản ghi ảnh nhưng n_test cộng lại là {n_img}")

    os.makedirs("runs", exist_ok=True)
    out = f"runs/scores_{a.tag}.json"
    json.dump({"tag": a.tag, "backbone": a.backbone, "protocol": a.protocol,
               "seed": a.seed, "recovered_from": a.fold_dir,
               "fp_rates": [0.125, 0.25, 0.5, 1.0, 2.0, 4.0],
               "n_images": len(recs), "n_nodules": n_nod,
               "images": recs}, open(out, "w"))
    print(f"{out}: {len(recs)} ảnh, {n_nod} nốt, "
          f"{sum(len(r['fps']) for r in recs)} dương tính giả")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--fold-dir", required=True)
    p.add_argument("--tag", required=True)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--backbone", default="resnet34")
    p.add_argument("--protocol", default="kfold")
    main(p.parse_args())
