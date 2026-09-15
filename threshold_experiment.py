"""
threshold_experiment.py — Đo xem giữ ngưỡng CỐ ĐỊNH làm khoảng tin cậy hẹp đi
bao nhiêu so với tính lại ngưỡng trong từng lần lấy mẫu bootstrap.

Bài báo khẳng định "dùng ngưỡng cố định của tập gốc cho khoảng hẹp giả tạo".
Đó là lập luận đúng về nguyên tắc, nhưng chưa có số. File này biến nó thành
bằng chứng: chạy cả hai cách trên cùng dữ liệu, cùng hạt giống, chỉ khác đúng
một điều là ngưỡng có được tính lại hay không.

    python threshold_experiment.py runs/scores_unet_kfold.json
"""
import json
import argparse

import numpy as np

from bootstrap_ci import load_scores, thresholds, SUB_NAMES


def metrics_at(packed, idx, rates, subs, thr_fixed=None):
    """Nếu thr_fixed là None thì tính lại ngưỡng từ chính mẫu này."""
    n = len(idx)
    sc = np.concatenate([packed[i][0] for i in idx]) if n else np.zeros(0)
    sb = np.concatenate([packed[i][1] for i in idx]) if n else np.zeros(0, np.int64)
    fp = np.concatenate([packed[i][2] for i in idx]) if n else np.zeros(0)
    thr = thr_fixed if thr_fixed is not None else thresholds(fp, n, rates)
    if sc.size == 0:
        return {}
    hits = {f: (sc >= thr[f]) & (sc > 0) for f in rates}
    out = {f"sens@{f}": float(hits[f].mean()) for f in rates}
    out["cpm"] = float(np.mean([out[f"sens@{f}"] for f in rates]))
    f1 = min(rates, key=lambda z: abs(z - 1.0))
    for s in subs:
        m = sb == s
        out[f"sub{s}@1.0"] = float(hits[f1][m].mean()) if m.sum() else np.nan
    return out


def main(a):
    d = load_scores(a.scores)
    packed, rates = d["packed"], d["fp_rates"]
    n = len(packed)
    subs = sorted({int(s) for _, sb, _ in packed for s in sb if s > 0})
    full = np.arange(n)

    fp_all = np.concatenate([p[2] for p in packed])
    thr_orig = thresholds(fp_all, n, rates)
    obs = metrics_at(packed, full, rates, subs)

    keys = ["cpm"] + [f"sens@{f}" for f in rates] + [f"sub{s}@1.0" for s in subs]
    boot = {k: {"lai": np.empty(a.B), "codinh": np.empty(a.B)} for k in keys}

    rng = np.random.RandomState(a.seed)
    for b in range(a.B):
        idx = rng.randint(0, n, n)                 # CÙNG mẫu cho cả hai cách
        m_lai = metrics_at(packed, idx, rates, subs, thr_fixed=None)
        m_co = metrics_at(packed, idx, rates, subs, thr_fixed=thr_orig)
        for k in keys:
            boot[k]["lai"][b] = m_lai.get(k, np.nan)
            boot[k]["codinh"][b] = m_co.get(k, np.nan)

    def width(v):
        v = v[np.isfinite(v)]
        return (np.percentile(v, 97.5) - np.percentile(v, 2.5)) if v.size > 20 else np.nan

    print("\n" + "=" * 76)
    print(f"NGƯỠNG TÍNH LẠI so với NGƯỠNG CỐ ĐỊNH  —  {d.get('tag')}  (B={a.B})")
    print("=" * 76)
    print("  Cùng dữ liệu, cùng hạt giống, cùng mẫu bootstrap.")
    print("  Khác đúng MỘT điều: ngưỡng FROC có được tính lại trong từng mẫu hay không.\n")
    hdr = f"  {'Chỉ số':<24}{'ước lượng':>10}{'rộng (tính lại)':>18}{'rộng (cố định)':>17}{'hẹp đi':>9}"
    print(hdr); print("  " + "-" * (len(hdr) - 2))
    rows = []
    for k in keys:
        wl, wc = width(boot[k]["lai"]), width(boot[k]["codinh"])
        if not (np.isfinite(wl) and np.isfinite(wc) and wl > 0):
            continue
        giam = 100 * (1 - wc / wl)
        rows.append(giam)
        lab = ("CPM" if k == "cpm" else
               f"Sens @ {k[5:]} FP" if k.startswith("sens@") else
               f"{SUB_NAMES.get(int(k[3]), k)} @1,0")
        print(f"  {lab:<24}{100*obs[k]:9.1f}%{100*wl:16.1f}{100*wc:17.1f}{giam:8.1f}%")
    print(f"\n  Trung bình khoảng tin cậy HẸP ĐI {np.mean(rows):.1f}% khi giữ ngưỡng cố định.")
    print("  Phần hẹp đi đó là độ bất định của chính việc ước lượng ngưỡng, bị bỏ qua.")

    json.dump({"tag": d.get("tag"), "B": a.B,
               "thu_hep_trung_binh_pct": float(np.mean(rows)),
               "thu_hep_max_pct": float(np.max(rows))},
              open("runs/threshold_experiment.json", "w"), indent=2)
    print("\nĐã lưu runs/threshold_experiment.json")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("scores")
    p.add_argument("--B", type=int, default=2000)
    p.add_argument("--seed", type=int, default=0)
    main(p.parse_args())
