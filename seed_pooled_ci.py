"""
seed_pooled_ci.py — Gộp nhiều seed và nhiều ảnh trong CÙNG một phân tích.

VÌ SAO CẦN FILE NÀY

Có HAI nguồn bất định, và chúng trả lời HAI câu hỏi khác nhau:

  1. Bất định do HUẤN LUYỆN (khởi tạo ngẫu nhiên). Đo bằng cách chạy nhiều
     seed. Trả lời: "chạy lại có ra kết quả này nữa không?"
  2. Bất định do MẪU ẢNH. Đo bằng bootstrap theo ảnh. Trả lời: "sang bộ 247
     ảnh khác thì kết quả có giữ không?"

Báo cáo riêng lẻ dễ dẫn tới kết luận sai theo cả hai chiều:

  - Chỉ nhìn bootstrap một seed: khoảng tin cậy rộng, chứa 0, kết luận "không
    có hiệu ứng". Nhưng nếu cả năm seed đều cho hiệu số DƯƠNG thì kết luận đó
    sai — nhiễu huấn luyện đang bị tính nhầm thành nhiễu dữ liệu.
  - Chỉ nhìn trung bình qua các seed: độ lệch chuẩn nhỏ, kết luận "hiệu ứng
    chắc chắn". Nhưng cả năm seed đều chạy trên CÙNG 247 ảnh, nên chúng không
    nói gì về việc tổng quát hóa sang ảnh mới.

Ước lượng chính ở đây là HIỆU SỐ TRUNG BÌNH QUA CÁC SEED, lấy khoảng tin cậy
bằng bootstrap theo ảnh. Trong mỗi lần lấy mẫu bootstrap, cùng một tập chỉ số
ảnh được áp cho CẢ MƯỜI lần chạy, rồi mới tính hiệu số từng seed và lấy trung
bình. Cách này vừa giảm nhiễu huấn luyện bằng trung bình, vừa giữ đúng bất
định do mẫu ảnh.

    python seed_pooled_ci.py --a "runs/scores_plain_s*.json" \
                             --b "runs/scores_unet_s*.json"
"""
import os
import glob
import json
import argparse
from statistics import NormalDist

import numpy as np

from bootstrap_ci import load_scores, evaluate

ND = NormalDist()
SUB_NAMES = {1: "1 Cực khó thấy", 2: "2 Rất khó thấy", 3: "3 Trung bình",
             4: "4 Tương đối rõ", 5: "5 Rõ ràng"}


def paired_t(d):
    """t-test một mẫu trên các hiệu số theo seed. n nhỏ nên dùng phân phối t."""
    d = np.asarray(d, float)
    n = d.size
    if n < 2:
        return np.nan, np.nan, (np.nan, np.nan)
    m, sd = d.mean(), d.std(ddof=1)
    se = sd / np.sqrt(n)
    if se == 0:
        return np.inf, 0.0, (m, m)
    t = m / se
    # xấp xỉ p hai phía bằng phân phối chuẩn khi không có scipy; với n=5 đây là
    # ước lượng LẠC QUAN, nên chỉ dùng kèm khoảng tin cậy, đừng dùng một mình
    p = 2 * (1 - ND.cdf(abs(t)))
    tcrit = {2: 12.706, 3: 4.303, 4: 3.182, 5: 2.776, 6: 2.571,
             7: 2.447, 8: 2.365, 9: 2.306, 10: 2.262}.get(n, 1.96)
    return t, p, (m - tcrit * se, m + tcrit * se)


def main(a):
    pa = sorted(glob.glob(a.a))
    pb = sorted(glob.glob(a.b))
    if len(pa) != len(pb) or not pa:
        raise SystemExit(f"Số file không khớp: A={len(pa)} B={len(pb)}")
    print(f"{len(pa)} cặp seed")
    for x, y in zip(pa, pb):
        print(f"  {os.path.basename(x):<28} vs {os.path.basename(y)}")

    DA = [load_scores(p) for p in pa]
    DB = [load_scores(p) for p in pb]

    # mọi lần chạy phải trên cùng tập ảnh, nếu không thì bắt cặp vô nghĩa
    # So sánh theo TẬP HỢP, không theo thứ tự. raddino_train2 và raddino_train3
    # liệt kê ảnh theo thứ tự khác nhau nhưng vẫn là cùng 247 ảnh; cảnh báo theo
    # thứ tự sẽ báo động giả mỗi lần chạy.
    ids = DA[0]["ids"]
    for d in DA + DB:
        if set(d["ids"]) != set(ids):
            common = [i for i in ids if i in set(d["ids"])]
            print(f"CẢNH BÁO: tập ảnh KHÁC NHAU THẬT — chỉ dùng {len(common)} "
                  f"ảnh chung (A có {len(ids)}, file này có {len(d['ids'])})")
            ids = common
    n = len(ids)
    rates = DA[0]["fp_rates"]
    subs = sorted({int(s) for _, sb, _ in DA[0]["packed"] for s in sb if s > 0})

    def repack(d):
        m = {i: p for i, p in zip(d["ids"], d["packed"])}
        return [m[i] for i in ids]

    PA = [repack(d) for d in DA]
    PB = [repack(d) for d in DB]

    # ---------- quan sát trên mẫu gốc ----------
    full = np.arange(n)
    obsA = [evaluate(p, full, rates, subs) for p in PA]
    obsB = [evaluate(p, full, rates, subs) for p in PB]

    keys = ["cpm"] + [f"sub{s}@1.0" for s in subs]
    print("\n" + "=" * 74)
    print("TỪNG SEED")
    print("=" * 74)
    print(f"  {'seed':<6}{'A (%)':>9}{'B (%)':>9}{'hiệu (điểm)':>14}")
    dl = []
    for i, (x, y) in enumerate(zip(obsA, obsB)):
        d = 100 * (y["cpm"] - x["cpm"])
        dl.append(d)
        print(f"  {i:<6}{100*x['cpm']:9.1f}{100*y['cpm']:9.1f}{d:>14.1f}")
    A = np.array([100 * o["cpm"] for o in obsA])
    B = np.array([100 * o["cpm"] for o in obsB])
    dl = np.array(dl)
    print(f"\n  A: {A.mean():.1f} ± {A.std(ddof=1):.1f}   "
          f"B: {B.mean():.1f} ± {B.std(ddof=1):.1f}")
    print(f"  Hiệu số: {dl.mean():+.1f} ± {dl.std(ddof=1):.1f} điểm")
    npos = int((dl > 0).sum())
    print(f"  Dương ở {npos}/{len(dl)} seed", end="")
    if npos == len(dl):
        print(f"  (kiểm định dấu hai phía: p = {2*0.5**len(dl):.4f})")
    else:
        print()
    t, p, ci = paired_t(dl)
    print(f"  t-test trên hiệu số theo seed: t={t:.2f}, "
          f"khoảng tin cậy 95% [{ci[0]:+.1f}, {ci[1]:+.1f}] điểm")
    print("  (chỉ nói về bất định HUẤN LUYỆN, KHÔNG nói gì về ảnh mới)")

    # ---------- bootstrap theo ảnh trên hiệu số trung bình qua seed ----------
    rng = np.random.RandomState(a.seed)
    boot = {k: np.empty(a.B) for k in keys}
    for b in range(a.B):
        idx = rng.randint(0, n, n)                 # CÙNG chỉ số cho cả 10 lần chạy
        for k in keys:
            ds = []
            for p_a, p_b in zip(PA, PB):
                va = evaluate(p_a, idx, rates, subs).get(k, np.nan)
                vb = evaluate(p_b, idx, rates, subs).get(k, np.nan)
                ds.append(vb - va)
            boot[k][b] = np.nanmean(ds)

    print("\n" + "=" * 74)
    print(f"HIỆU SỐ TRUNG BÌNH QUA {len(PA)} SEED — bootstrap theo ẢNH (B={a.B})")
    print("=" * 74)
    print(f"  {'Chỉ số':<26}{'Hiệu':>9}{'Khoảng tin cậy 95%':>24}{'p':>9}")
    print("  " + "-" * 66)
    out = {}
    for k in keys:
        obs = float(np.nanmean([
            obsB[i].get(k, np.nan) - obsA[i].get(k, np.nan) for i in range(len(PA))]))
        v = boot[k][np.isfinite(boot[k])]
        if v.size < 20:
            continue
        lo = float(np.percentile(v, 2.5)); hi = float(np.percentile(v, 97.5))
        pb_ = 2 * min((v <= 0).mean(), (v >= 0).mean())
        pb_ = float(min(1.0, max(pb_, 1.0 / (v.size + 1))))
        star = " *" if (lo > 0 or hi < 0) else ""
        lab = "CPM" if k == "cpm" else SUB_NAMES.get(int(k[3]), k) + " @1,0"
        print(f"  {lab:<26}{100*obs:+9.1f}"
              f"{f'[{100*lo:+.1f}, {100*hi:+.1f}]':>24}{pb_:>9.3f}{star}")
        out[k] = {"delta": obs, "lo": lo, "hi": hi, "p": pb_}

    print("\n  * = khoảng tin cậy không chứa 0.")
    print("  Khoảng này bao gồm CẢ bất định do mẫu ảnh; nó là con số nên đưa")
    print("  vào bài báo, vì bài báo nói về ảnh nói chung chứ không riêng 247")
    print("  ảnh này.")

    os.makedirs("runs", exist_ok=True)
    json.dump({"n_seeds": len(PA), "n_images": n, "B": a.B,
               "per_seed_A": A.tolist(), "per_seed_B": B.tolist(),
               "per_seed_delta": dl.tolist(),
               "delta_mean": float(dl.mean()), "delta_sd": float(dl.std(ddof=1)),
               "seed_t_ci": [float(ci[0]), float(ci[1])],
               "pooled": out}, open("runs/seed_pooled_ci.json", "w"), indent=2)
    print("\nĐã lưu runs/seed_pooled_ci.json")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--a", required=True, help="glob cho nhánh A")
    p.add_argument("--b", required=True, help="glob cho nhánh B")
    p.add_argument("--B", type=int, default=2000)
    p.add_argument("--seed", type=int, default=0)
    main(p.parse_args())
