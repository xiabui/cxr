"""
plot_figures.py — Sinh hình cho bài báo từ runs/scores_*.json.

Không cần GPU, không cần huấn luyện lại. Mọi khoảng tin cậy được tính lại từ
điểm thô bằng đúng quy trình của bootstrap_ci.py (lấy mẫu lại THEO ẢNH, ngưỡng
tính lại trong từng lần lấy mẫu).

    python plot_figures.py --out paper/fig

Sinh ba hình:
  fig_froc.pdf    đường cong FROC ba cấu hình, có dải tin cậy
  fig_strat.pdf   độ nhạy theo mức subtlety, có thanh sai số
  fig_forest.pdf  hiệu số do khử xương theo từng mức (biểu đồ rừng)

BẢNG MÀU: Okabe–Ito, chuẩn an toàn cho người mù màu. Đã chạy qua validator:
mọi cặp kề nhau đều đạt ngưỡng phân biệt. Mỗi cấu hình còn được gán THÊM một
dạng nét và một dấu riêng, để hình vẫn đọc được khi in đen trắng — bài báo in
ra giấy thì màu là thứ mất trước tiên.
"""
import os
import argparse
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from bootstrap_ci import load_scores, evaluate

# Okabe–Ito: xanh dương / cam đỏ / xanh lục
COLORS = {"plain": "#0072B2", "unet": "#D55E00", "aug": "#009E73"}
MARKS = {"plain": "o", "unet": "s", "aug": "^"}
LINES = {"plain": "-", "unet": "--", "aug": ":"}
SUB_SHORT = {1: "1\nrất kín đáo", 2: "2", 3: "3", 4: "4", 5: "5\nrõ ràng"}
SUB_EN = {1: "1\nextremely\nsubtle", 2: "2\nvery\nsubtle", 3: "3\nmoderate",
          4: "4\nfairly\nobvious", 5: "5\nobvious"}


def style():
    plt.rcParams.update({
        "font.family": "serif", "font.size": 8,
        "axes.linewidth": 0.6, "axes.edgecolor": "#444444",
        "axes.grid": True, "grid.color": "#DDDDDD", "grid.linewidth": 0.4,
        "axes.axisbelow": True,                 # lưới nằm DƯỚI dữ liệu
        "xtick.major.width": 0.6, "ytick.major.width": 0.6,
        "legend.frameon": False, "figure.dpi": 200,
    })


def boot_curves(d, B, rng):
    """Trả về (quan sát, mảng bootstrap) cho độ nhạy tại từng mức FP."""
    packed, rates = d["packed"], d["fp_rates"]
    n = len(packed)
    subs = sorted({int(s) for _, sb, _ in packed for s in sb if s > 0})
    obs = evaluate(packed, np.arange(n), rates, subs)
    out = {k: np.empty(B) for k in [f"sens@{f}" for f in rates]
           + [f"sub{s}@1.0" for s in subs]}
    for b in range(B):
        idx = rng.randint(0, n, n)
        m = evaluate(packed, idx, rates, subs)
        for k in out:
            out[k][b] = m.get(k, np.nan)
    return obs, out, rates, subs


def band(arr):
    a = arr[np.isfinite(arr)]
    return (np.percentile(a, 2.5), np.percentile(a, 97.5)) if a.size > 20 else (np.nan, np.nan)


# ------------------------------------------------------------------ hình 1
def fig_froc(runs, out, B, seed):
    fig, ax = plt.subplots(figsize=(3.4, 2.7))
    rng = np.random.RandomState(seed)
    for key, (label, d) in runs.items():
        obs, bt, rates, _ = boot_curves(d, B, rng)
        y = [100 * obs[f"sens@{f}"] for f in rates]
        lo, hi = zip(*[band(bt[f"sens@{f}"]) for f in rates])
        ax.fill_between(rates, [100*v for v in lo], [100*v for v in hi],
                        color=COLORS[key], alpha=0.13, linewidth=0)
        ax.plot(rates, y, LINES[key], color=COLORS[key], linewidth=1.3,
                marker=MARKS[key], markersize=3.4, label=label, zorder=3)
    ax.set_xscale("log")
    # Trên trục log matplotlib tự thêm nhãn tick PHỤ (2×10⁻¹, 4×10⁻¹ ...) và
    # chúng đè lên nhãn mình đặt. Phải tắt cả locator lẫn formatter phụ.
    from matplotlib.ticker import NullFormatter, NullLocator, FixedLocator
    ax.xaxis.set_minor_locator(NullLocator())
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.xaxis.set_major_locator(FixedLocator([0.125, 0.25, 0.5, 1, 2, 4]))
    ax.set_xticklabels(["0.125", "0.25", "0.5", "1", "2", "4"])
    ax.set_xlabel("False positives per image")
    ax.set_ylabel("Sensitivity (%)")
    ax.set_ylim(0, 80)
    ax.legend(loc="upper left", fontsize=7)
    fig.tight_layout(pad=0.3)
    p = os.path.join(out, "fig_froc.pdf")
    fig.savefig(p); plt.close(fig)
    print("  ", p)


# ------------------------------------------------------------------ hình 2
def fig_strat(runs, out, B, seed):
    fig, ax = plt.subplots(figsize=(3.4, 2.7))
    rng = np.random.RandomState(seed)
    offs = {"plain": -0.18, "unet": 0.0, "aug": 0.18}
    for key, (label, d) in runs.items():
        obs, bt, rates, subs = boot_curves(d, B, rng)
        xs = np.array(subs, float) + offs[key]
        y = np.array([100 * obs[f"sub{s}@1.0"] for s in subs])
        lo, hi = zip(*[band(bt[f"sub{s}@1.0"]) for s in subs])
        err = np.vstack([y - 100*np.array(lo), 100*np.array(hi) - y])
        err = np.clip(err, 0, None)
        ax.errorbar(xs, y, yerr=err, fmt=MARKS[key], color=COLORS[key],
                    markersize=3.6, linewidth=0, elinewidth=0.9,
                    capsize=1.8, capthick=0.9, label=label, zorder=3)
    ax.set_xticks(list(SUB_EN))
    ax.set_xticklabels([SUB_EN[s] for s in SUB_EN], fontsize=6)
    ax.set_xlabel("Subtlety level")
    ax.set_ylabel("Sensitivity at 1.0 FP/image (%)")
    ax.set_ylim(-5, 105)
    ax.legend(loc="upper left", fontsize=7)
    fig.tight_layout(pad=0.3)
    p = os.path.join(out, "fig_strat.pdf")
    fig.savefig(p); plt.close(fig)
    print("  ", p)


# ------------------------------------------------------------------ hình 3
def fig_forest(pairs, out, B, seed):
    """Hiệu số do khử xương, trung bình qua các seed, bootstrap theo ảnh."""
    ids = pairs[0][0]["ids"]
    def repack(d):
        m = dict(zip(d["ids"], d["packed"]))
        return [m[i] for i in ids]
    PA = [repack(a) for a, _ in pairs]
    PB = [repack(b) for _, b in pairs]
    n = len(ids)
    rates = pairs[0][0]["fp_rates"]
    subs = sorted({int(s) for _, sb, _ in PA[0] for s in sb if s > 0})
    keys = [f"sub{s}@1.0" for s in subs] + ["cpm"]

    full = np.arange(n)
    oa = [evaluate(p, full, rates, subs) for p in PA]
    ob = [evaluate(p, full, rates, subs) for p in PB]
    obs = {k: float(np.nanmean([ob[i][k] - oa[i][k] for i in range(len(PA))]))
           for k in keys}

    rng = np.random.RandomState(seed)
    bt = {k: np.empty(B) for k in keys}
    for b in range(B):
        idx = rng.randint(0, n, n)
        ev_a = [evaluate(p, idx, rates, subs) for p in PA]
        ev_b = [evaluate(p, idx, rates, subs) for p in PB]
        for k in keys:
            bt[k][b] = np.nanmean([ev_b[i].get(k, np.nan) - ev_a[i].get(k, np.nan)
                                   for i in range(len(PA))])

    labels = [f"Subtlety {s}" for s in subs] + ["CPM (aggregate)"]
    ys = np.arange(len(keys))[::-1]
    fig, ax = plt.subplots(figsize=(3.6, 2.5))
    ax.axvline(0, color="#888888", linewidth=0.8, zorder=1)
    for y, k in zip(ys, keys):
        lo, hi = band(bt[k])
        d = 100 * obs[k]
        sig = (lo > 0 or hi < 0)
        c = "#D55E00" if sig else "#555555"
        ax.plot([100*lo, 100*hi], [y, y], color=c, linewidth=1.1, zorder=2)
        ax.plot([100*lo, 100*hi], [y, y], "|", color=c, markersize=3.5, zorder=2)
        ax.plot(d, y, "D" if k == "cpm" else "o", color=c,
                markersize=4.2 if k == "cpm" else 3.6,
                markeredgecolor="white", markeredgewidth=0.5, zorder=3)
    ax.set_yticks(ys); ax.set_yticklabels(labels, fontsize=7)
    ax.set_xlabel("Change from bone suppression (points)")
    ax.grid(axis="y", visible=False)
    fig.tight_layout(pad=0.3)
    p = os.path.join(out, "fig_forest.pdf")
    fig.savefig(p); plt.close(fig)
    print("  ", p)
    for k, l in zip(keys, labels):
        lo, hi = band(bt[k])
        print(f"     {l:<20} {100*obs[k]:+6.1f}  [{100*lo:+6.1f}, {100*hi:+6.1f}]")


def main(a):
    os.makedirs(a.out, exist_ok=True)
    style()
    # CẢ BA phải cùng một seed. Bản trước lấy plain/unet ở seed 0 nhưng
    # augmentation ở seed 42 — trộn seed trong cùng một hình thì phần chênh
    # lệch giữa các đường lẫn cả nhiễu khởi tạo, không còn đọc được.
    runs = {
        "plain": ("RAD-DINO, 3-channel", load_scores("runs/scores_raddino_plain.json")),
        "unet":  ("+ bone suppression",  load_scores("runs/scores_unet_kfold.json")),
        "aug":   ("+ augmentation",      load_scores("runs/scores_unetaug_kfold.json")),
    }
    seeds = {k: d.get("seed") for k, (_, d) in runs.items()}
    assert len(set(seeds.values())) == 1, f"LỆCH SEED giữa các cấu hình: {seeds}"
    print(f"Tất cả cấu hình dùng seed {next(iter(seeds.values()))}")
    print("Đang vẽ:")
    fig_froc(runs, a.out, a.B, a.seed)
    fig_strat(runs, a.out, a.B, a.seed)
    pairs = [(load_scores(f"runs/scores_plain_s{i}.json"),
              load_scores(f"runs/scores_unet_s{i}.json")) for i in range(5)]
    fig_forest(pairs, a.out, a.B, a.seed)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="paper/fig")
    p.add_argument("--B", type=int, default=1000)
    p.add_argument("--seed", type=int, default=0)
    main(p.parse_args())
