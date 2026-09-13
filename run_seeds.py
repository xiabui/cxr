"""
run_seeds.py — Chạy lại cùng một cấu hình với nhiều seed rồi tổng hợp.

Đây là mục 8.3 trong CLAUDE.md: báo cáo trung bình ± độ lệch chuẩn thay vì
một con số đơn lẻ.

Phân biệt rõ với bootstrap_ci.py, hai thứ đo hai loại bất định KHÁC NHAU:
  - bootstrap_ci.py  : bất định do MẪU DỮ LIỆU (247 ảnh này tình cờ là ai).
  - run_seeds.py     : bất định do HUẤN LUYỆN (khởi tạo, thứ tự batch, chia fold).
Bài báo nên có cả hai. Khoảng tin cậy bootstrap thường rộng hơn nhiều.

    python run_seeds.py --config configs/jsrt_10fold.yaml --seeds 0 1 2 3 4
    python run_seeds.py --aggregate-only --seeds 0 1 2 3 4

Mỗi seed sinh runs/results_*_seed<S>.json và runs/scores_*_seed<S>.json.
"""
import os, sys, json, glob, argparse, subprocess

import numpy as np

SUB_NAMES = {1: "1 Cực khó thấy", 2: "2 Rất khó thấy", 3: "3 Trung bình",
             4: "4 Tương đối rõ", 5: "5 Rõ ràng"}


def tag_for(cfg_path, seed):
    import yaml
    proto = yaml.safe_load(open(cfg_path))["train"].get("protocol", "kfold")
    return f"raddino_aug_{proto}_seed{seed}"


def run_one(cfg, seed, epochs, lr, extra):
    cmd = [sys.executable, "raddino_train3.py", "--config", cfg,
           "--seed", str(seed), "--epochs", str(epochs), "--lr", str(lr)] + extra
    print(f"\n{'='*70}\n>>> seed {seed}: {' '.join(cmd)}\n{'='*70}", flush=True)
    r = subprocess.run(cmd, env={**os.environ,
                                 "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"})
    if r.returncode != 0:
        print(f"!!! seed {seed} thất bại (mã {r.returncode}), bỏ qua seed này")
    return r.returncode == 0


def aggregate(tags):
    runs = []
    for t in tags:
        p = f"runs/results_{t}.json"
        if os.path.exists(p):
            runs.append(json.load(open(p)))
        else:
            print(f"  thiếu {p}")
    if not runs:
        raise SystemExit("Không có kết quả nào để tổng hợp.")

    def ms(vals):
        v = np.array([x for x in vals if x is not None], dtype=float)
        return (float(v.mean()), float(v.std(ddof=1)) if v.size > 1 else 0.0, v.size)

    print("\n" + "=" * 66)
    print(f"TỔNG HỢP {len(runs)} SEED — {runs[0].get('protocol')}")
    print("=" * 66)
    m, s, k = ms([r["cpm"] for r in runs])
    print(f"  CPM : {100*m:5.1f}% ± {100*s:.1f}  (n={k} seed)")

    rates = sorted({float(f) for r in runs for f in r.get("sens_at", {})})
    for f in rates:
        m, s, _ = ms([r["sens_at"].get(str(f)) for r in runs])
        print(f"  Sens @ {f:>5} FP/ảnh : {100*m:5.1f}% ± {100*s:.1f}")

    subs = sorted({int(x) for r in runs for x in r.get("subtlety_cpm", {})})
    if subs:
        print("\n  Độ nhạy phân tầng (CPM theo nhóm):")
        for sub in subs:
            m, s, _ = ms([r["subtlety_cpm"].get(str(sub)) for r in runs])
            print(f"    {SUB_NAMES.get(sub, sub):<16} {100*m:5.1f}% ± {100*s:.1f}")

    out = {"n_seeds": len(runs), "seeds": [r.get("seed") for r in runs],
           "cpm_mean": ms([r["cpm"] for r in runs])[0],
           "cpm_sd": ms([r["cpm"] for r in runs])[1],
           "per_seed_cpm": {str(r.get("seed")): r["cpm"] for r in runs}}
    json.dump(out, open("runs/seeds_summary.json", "w"), indent=2)
    print("\nĐã lưu runs/seeds_summary.json")
    print("Nhắc: độ lệch chuẩn giữa các seed KHÔNG phải khoảng tin cậy.")
    print("      Chạy bootstrap_ci.py trên một seed để có khoảng tin cậy dữ liệu.")


def main(a):
    tags = [tag_for(a.config, s) for s in a.seeds]
    if not a.aggregate_only:
        for seed in a.seeds:
            run_one(a.config, seed, a.epochs, a.lr, a.extra)
    aggregate(tags)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/jsrt_10fold.yaml")
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--aggregate-only", action="store_true",
                   help="chỉ tổng hợp từ runs/ đã có, không chạy lại")
    p.add_argument("--extra", nargs=argparse.REMAINDER, default=[],
                   help="tham số truyền thẳng cho raddino_train3.py")
    main(p.parse_args())
