"""
bootstrap_ci.py — Khoảng tin cậy bootstrap cho CPM và độ nhạy phân tầng.

Đây là mục 8.1 trong CLAUDE.md: bắt buộc phải có, nếu không phát hiện về sự
đánh đổi giữa các nhóm subtlety sẽ bị phản biện là nhiễu ngẫu nhiên.

BA QUYẾT ĐỊNH PHƯƠNG PHÁP, đọc kỹ trước khi sửa:

  1. LẤY MẪU LẠI THEO ẢNH, KHÔNG THEO NỐT (cluster bootstrap).
     Ngưỡng FROC được định nghĩa theo số dương tính giả TRÊN MỖI ẢNH. Nếu
     lấy mẫu lại từng nốt riêng lẻ thì số ảnh không còn xác định và các
     dương tính giả bị tách khỏi ảnh sinh ra chúng, làm ngưỡng vô nghĩa.
     Một ảnh được chọn lại kéo theo toàn bộ nốt VÀ toàn bộ FP của nó.

  2. NGƯỠNG ĐƯỢC TÍNH LẠI TRONG TỪNG LẦN LẤY MẪU.
     Dùng ngưỡng cố định của tập gốc sẽ cho khoảng tin cậy hẹp giả tạo, vì
     bỏ qua chính độ bất định của việc ước lượng ngưỡng. Đây cũng là biến
     thể của lỗi "ngưỡng FROC và bảng subtlety lệch nhau" đã mắc hai lần
     (CLAUDE.md mục 5) — ở đây mọi chỉ số đều dùng chung một ngưỡng.

  3. ĐỘ NHẠY PHÂN TẦNG DÙNG SỐ NỐT TRONG CHÍNH LẦN LẤY MẪU ĐÓ.
     Nhóm subtlety 5 chỉ có 12 nốt, nhóm 1 có 25. Khoảng tin cậy sẽ rất
     rộng — đó là kết quả đúng, không phải lỗi, và chính là điều cần báo
     cáo để không diễn giải quá mức chênh lệch một vài nốt.

Cách dùng:

    # khoảng tin cậy cho một lần chạy
    python bootstrap_ci.py runs/scores_raddino_aug_loco_seed42.json

    # so sánh CÓ BẮT CẶP giữa hai cấu hình (cùng tập ảnh)
    python bootstrap_ci.py --compare runs/scores_baseline.json \
                                     runs/scores_raddino_aug_loco_seed42.json

Đầu vào là file runs/scores_*.json do raddino_train3.py / external_vindr.py
sinh ra. Không cần GPU, không cần nạp lại đặc trưng.
"""
import os
import json
import argparse
from statistics import NormalDist

import numpy as np

ND = NormalDist()
SUB_NAMES = {1: "1 Cực khó thấy", 2: "2 Rất khó thấy", 3: "3 Trung bình",
             4: "4 Tương đối rõ", 5: "5 Rõ ràng"}


# ---------------------------------------------------------------- nạp dữ liệu
def load_scores(path):
    d = json.load(open(path))
    imgs = d["images"]
    packed = []
    for r in imgs:
        packed.append((
            np.array([nd["score"] for nd in r["nodules"]], dtype=np.float64),
            np.array([nd.get("subtlety", 0) or 0 for nd in r["nodules"]], dtype=np.int64),
            np.array(r["fps"], dtype=np.float64),
        ))
    d["packed"] = packed
    d["ids"] = [r["image"] for r in imgs]
    d.setdefault("fp_rates", [0.125, 0.25, 0.5, 1.0, 2.0, 4.0])
    return d


# ------------------------------------------------------- chỉ số trên một mẫu
def thresholds(fp_all, n_img, rates):
    """Ngưỡng theo mức FP/ảnh. Giữ ĐÚNG công thức của raddino_train3.py."""
    fps = np.sort(fp_all)[::-1] if fp_all.size else fp_all
    thr = {}
    for f in rates:
        k = int(round(n_img * f))
        if fps.size == 0 or k >= fps.size:
            thr[f] = 0.0
        elif k <= 0:
            thr[f] = float(fps[0]) + 1e-9
        else:
            thr[f] = float(fps[k - 1])
    return thr


def evaluate(packed, idx, rates, subs_present):
    """Tính mọi chỉ số trên tập ảnh được chỉ định bởi idx (có thể lặp lại)."""
    n = len(idx)
    sc = np.concatenate([packed[i][0] for i in idx]) if n else np.zeros(0)
    sb = np.concatenate([packed[i][1] for i in idx]) if n else np.zeros(0, np.int64)
    fp = np.concatenate([packed[i][2] for i in idx]) if n else np.zeros(0)

    thr = thresholds(fp, n, rates)
    out = {}
    if sc.size == 0:
        return {k: np.nan for k in metric_names(rates, subs_present)}

    hits = {f: (sc >= thr[f]) & (sc > 0) for f in rates}
    for f in rates:
        out[f"sens@{f}"] = float(hits[f].mean())
    out["cpm"] = float(np.mean([out[f"sens@{f}"] for f in rates]))

    f1 = min(rates, key=lambda z: abs(z - 1.0))
    for s in subs_present:
        m = sb == s
        cnt = int(m.sum())
        if cnt == 0:                       # nhóm vắng mặt trong lần lấy mẫu này
            out[f"sub{s}_cpm"] = np.nan
            out[f"sub{s}@1.0"] = np.nan
        else:
            out[f"sub{s}_cpm"] = float(np.mean([hits[f][m].mean() for f in rates]))
            out[f"sub{s}@1.0"] = float(hits[f1][m].mean())
    return out


def metric_names(rates, subs_present):
    names = ["cpm"] + [f"sens@{f}" for f in rates]
    for s in subs_present:
        names += [f"sub{s}_cpm", f"sub{s}@1.0"]
    return names


# ------------------------------------------------------------------ bootstrap
def bca_interval(boot, obs, jack, alpha):
    """Khoảng BCa. Lùi về percentile khi mẫu suy biến (mọi giá trị bằng nhau)."""
    boot = boot[np.isfinite(boot)]
    if boot.size < 20 or not np.isfinite(obs):
        return (np.nan, np.nan, "n/a")
    lo_p, hi_p = 100 * alpha / 2, 100 * (1 - alpha / 2)

    prop = float((boot < obs).mean())
    jack = jack[np.isfinite(jack)]
    if prop <= 0.0 or prop >= 1.0 or jack.size < 3:
        return (float(np.percentile(boot, lo_p)),
                float(np.percentile(boot, hi_p)), "percentile")

    z0 = ND.inv_cdf(prop)
    d = jack.mean() - jack
    den = 6.0 * (np.sum(d ** 2) ** 1.5)
    acc = float(np.sum(d ** 3) / den) if den > 0 else 0.0

    out = []
    for z in (ND.inv_cdf(alpha / 2), ND.inv_cdf(1 - alpha / 2)):
        adj = z0 + (z0 + z) / (1 - acc * (z0 + z))
        out.append(100 * ND.cdf(adj))
    lo_q, hi_q = min(out), max(out)
    if not (0 < lo_q < hi_q < 100):
        return (float(np.percentile(boot, lo_p)),
                float(np.percentile(boot, hi_p)), "percentile")
    return (float(np.percentile(boot, lo_q)),
            float(np.percentile(boot, hi_q)), "BCa")


def run_single(d, B, alpha, seed):
    packed, rates = d["packed"], d["fp_rates"]
    n = len(packed)
    subs = sorted({int(s) for _, sb, _ in packed for s in sb if s > 0})
    names = metric_names(rates, subs)

    obs = evaluate(packed, np.arange(n), rates, subs)

    rng = np.random.RandomState(seed)
    boot = {k: np.empty(B) for k in names}
    for b in range(B):
        idx = rng.randint(0, n, n)
        m = evaluate(packed, idx, rates, subs)
        for k in names:
            boot[k][b] = m.get(k, np.nan)

    # jackknife theo ảnh, cần cho hệ số gia tốc của BCa
    jack = {k: np.empty(n) for k in names}
    for i in range(n):
        idx = np.delete(np.arange(n), i)
        m = evaluate(packed, idx, rates, subs)
        for k in names:
            jack[k][i] = m.get(k, np.nan)

    res = {}
    for k in names:
        lo, hi, how = bca_interval(boot[k], obs[k], jack[k], alpha)
        res[k] = {"obs": obs[k], "lo": lo, "hi": hi,
                  "se": float(np.nanstd(boot[k], ddof=1)), "method": how}
    return res, subs, rates


# ------------------------------------------------------- so sánh có bắt cặp
def run_compare(da, db, B, alpha, seed):
    """Hiệu số giữa hai cấu hình, lấy mẫu lại CÙNG một tập chỉ số ảnh.

    Bắt cặp theo ảnh là điểm mấu chốt: hai cấu hình được đánh giá trên đúng
    247 ảnh JSRT đó, nên phần lớn biến thiên là chung và sẽ bị triệt tiêu.
    So sánh không bắt cặp sẽ cho khoảng rộng hơn nhiều một cách không cần thiết.
    """
    ids_a, ids_b = da["ids"], db["ids"]
    common = [i for i in ids_a if i in set(ids_b)]
    if len(common) < 2:
        raise SystemExit("Hai file không có ảnh chung — không bắt cặp được.")
    if len(common) != len(ids_a) or len(common) != len(ids_b):
        print(f"CẢNH BÁO: chỉ {len(common)} ảnh chung "
              f"(A={len(ids_a)}, B={len(ids_b)}). Chỉ dùng phần giao.")

    pa = {i: p for i, p in zip(ids_a, da["packed"])}
    pb = {i: p for i, p in zip(ids_b, db["packed"])}
    packed_a = [pa[i] for i in common]
    packed_b = [pb[i] for i in common]

    rates = [f for f in da["fp_rates"] if f in set(db["fp_rates"])]
    subs = sorted({int(s) for _, sb, _ in packed_a for s in sb if s > 0} &
                  {int(s) for _, sb, _ in packed_b for s in sb if s > 0})
    names = metric_names(rates, subs)
    n = len(common)

    oa = evaluate(packed_a, np.arange(n), rates, subs)
    ob = evaluate(packed_b, np.arange(n), rates, subs)

    rng = np.random.RandomState(seed)
    dboot = {k: np.empty(B) for k in names}
    for b in range(B):
        idx = rng.randint(0, n, n)          # CÙNG chỉ số cho cả hai
        ma = evaluate(packed_a, idx, rates, subs)
        mb = evaluate(packed_b, idx, rates, subs)
        for k in names:
            dboot[k][b] = mb.get(k, np.nan) - ma.get(k, np.nan)

    res = {}
    for k in names:
        v = dboot[k][np.isfinite(dboot[k])]
        delta = ob[k] - oa[k]
        if v.size < 20:
            res[k] = {"a": oa[k], "b": ob[k], "delta": delta,
                      "lo": np.nan, "hi": np.nan, "p": np.nan}
            continue
        lo = float(np.percentile(v, 100 * alpha / 2))
        hi = float(np.percentile(v, 100 * (1 - alpha / 2)))
        # p hai phía theo cách đảo dấu quanh 0, kẹp để không bao giờ ra p = 0
        p = 2 * min((v <= 0).mean(), (v >= 0).mean())
        p = float(min(1.0, max(p, 1.0 / (v.size + 1))))
        res[k] = {"a": oa[k], "b": ob[k], "delta": delta, "lo": lo, "hi": hi, "p": p}
    return res, subs, rates, n


# --------------------------------------------------------------------- in ấn
def pct(x):
    return "  n/a " if not np.isfinite(x) else f"{100*x:5.1f}"


def label(k, rates, subs):
    if k == "cpm":
        return "CPM"
    if k.startswith("sens@"):
        return f"Sens @ {k[5:]} FP/ảnh"
    s = int(k[3])
    return (f"{SUB_NAMES.get(s, s)} (CPM)" if k.endswith("_cpm")
            else f"{SUB_NAMES.get(s, s)} @1,0 FP")


def print_single(res, subs, rates, d, B, alpha):
    print("\n" + "=" * 72)
    print(f"KHOẢNG TIN CẬY BOOTSTRAP — {d.get('tag', '?')}")
    print("=" * 72)
    print(f"  Nguồn      : {d.get('backbone','?')} | {d.get('protocol','?')} "
          f"| seed {d.get('seed','?')}")
    print(f"  Dữ liệu    : {d.get('n_images','?')} ảnh, {d.get('n_nodules','?')} nốt")
    print(f"  Bootstrap  : B={B}, lấy mẫu lại theo ẢNH, ngưỡng tính lại mỗi lần")
    print(f"  Khoảng     : {100*(1-alpha):.0f}%\n")
    hdr = f"  {'Chỉ số':<24} {'Ước lượng':>9} {'Khoảng tin cậy':>18} {'SE':>7}  PP"
    print(hdr); print("  " + "-" * (len(hdr) - 2))
    order = ["cpm"] + [f"sens@{f}" for f in rates]
    for k in order:
        r = res[k]
        print(f"  {label(k,rates,subs):<24} {pct(r['obs']):>8}% "
              f"[{pct(r['lo'])}, {pct(r['hi'])}]% {100*r['se']:6.1f}  {r['method']}")
    if not subs:
        print("\n  (Dữ liệu không có nhãn subtlety — bỏ qua phần phân tầng.)")
        return
    print(f"\n  Độ nhạy phân tầng tại 1,0 FP/ảnh:")
    for s in subs:
        r = res[f"sub{s}@1.0"]
        print(f"  {SUB_NAMES.get(s,s):<24} {pct(r['obs']):>8}% "
              f"[{pct(r['lo'])}, {pct(r['hi'])}]% {100*r['se']:6.1f}  {r['method']}")
    print("\n  Nhắc: nhóm 1 và nhóm 5 rất ít nốt nên khoảng sẽ rất rộng.")
    print("  Khoảng rộng chồng lấn nhau nghĩa là KHÔNG kết luận được cấu hình nào hơn.")


def print_compare(res, subs, rates, n, pa, pb, B, alpha):
    print("\n" + "=" * 78)
    print("SO SÁNH CÓ BẮT CẶP (bootstrap trên cùng tập ảnh)")
    print("=" * 78)
    print(f"  A = {os.path.basename(pa)}")
    print(f"  B = {os.path.basename(pb)}")
    print(f"  {n} ảnh chung | B={B} | khoảng {100*(1-alpha):.0f}% | hiệu số = B − A\n")
    hdr = (f"  {'Chỉ số':<24} {'A':>6} {'B':>6} {'Hiệu':>7} "
           f"{'Khoảng tin cậy':>18} {'p':>7}")
    print(hdr); print("  " + "-" * (len(hdr) - 2))
    order = ["cpm"] + [f"sens@{f}" for f in rates] + [f"sub{s}@1.0" for s in subs]
    for k in order:
        r = res[k]
        star = ""
        if np.isfinite(r["lo"]) and (r["lo"] > 0 or r["hi"] < 0):
            star = " *"
        pstr = "  n/a " if not np.isfinite(r["p"]) else f"{r['p']:.3f}"
        print(f"  {label(k,rates,subs):<24} {pct(r['a']):>6} {pct(r['b']):>6} "
              f"{100*r['delta']:+6.1f} [{pct(r['lo'])}, {pct(r['hi'])}]% {pstr:>7}{star}")
    print("\n  * = khoảng tin cậy của hiệu số không chứa 0.")
    print("  p là giá trị bootstrap hai phía, bị chặn dưới bởi 1/(B+1); đây là")
    print("  ước lượng thô, không thay thế cho việc đọc chính khoảng tin cậy.")
    print("  Các chỉ số phân tầng KHÔNG được hiệu chỉnh đa so sánh — coi là thăm dò.")


# ---------------------------------------------------------------------- main
def main(a):
    os.makedirs("runs", exist_ok=True)
    if a.compare:
        pa, pb = a.compare
        da, db = load_scores(pa), load_scores(pb)
        res, subs, rates, n = run_compare(da, db, a.B, a.alpha, a.seed)
        print_compare(res, subs, rates, n, pa, pb, a.B, a.alpha)
        out = a.out or (f"runs/ci_compare_{da.get('tag','A')}_vs_"
                        f"{db.get('tag','B')}.json")
        json.dump({"mode": "compare", "a": pa, "b": pb, "n_images": n,
                   "B": a.B, "alpha": a.alpha, "paired": True, "metrics": res},
                  open(out, "w"), indent=2, default=float)
    else:
        d = load_scores(a.scores)
        res, subs, rates = run_single(d, a.B, a.alpha, a.seed)
        print_single(res, subs, rates, d, a.B, a.alpha)
        out = a.out or f"runs/ci_{d.get('tag','run')}.json"
        json.dump({"mode": "single", "scores": a.scores, "tag": d.get("tag"),
                   "B": a.B, "alpha": a.alpha, "cluster": "image",
                   "metrics": res},
                  open(out, "w"), indent=2, default=float)
    print(f"\nĐã lưu {out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("scores", nargs="?", help="runs/scores_*.json")
    p.add_argument("--compare", nargs=2, metavar=("A", "B"),
                   help="so sánh có bắt cặp giữa hai file scores")
    p.add_argument("--B", type=int, default=2000, help="số lần lấy mẫu bootstrap")
    p.add_argument("--alpha", type=float, default=0.05, help="0,05 cho khoảng 95%%")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=None)
    args = p.parse_args()
    if not args.scores and not args.compare:
        p.error("cần một file scores hoặc --compare A B")
    main(args)
