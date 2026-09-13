"""FROC + CPM + độ nhạy phân tầng theo subtlety.

Đây là phần LÕI đánh giá của đề tài (Hướng B). Khác biệt so với nhiều nghiên
cứu cũ: không chỉ báo cáo một AUC tổng, mà tách độ nhạy theo từng mức subtlety
(1=cực khó thấy ... 5=rõ) — đúng chỗ mà tài liệu hiện tại còn thiếu.

Quy ước:
- Một "hit" = tâm dự đoán nằm trong hit_radius_mm của tâm nốt thật.
- Mỗi nốt thật chỉ được ghi nhận trúng bởi 1 dự đoán (tránh đếm trùng).
- Dự đoán không khớp nốt nào = false positive.
- FROC: độ nhạy theo số FP trung bình mỗi ảnh, lấy tại các mốc fp_rates.
- CPM (Competition Performance Metric): trung bình độ nhạy tại các mốc FP chuẩn.
"""
import numpy as np
from scipy.ndimage import maximum_filter


def extract_peaks(heatmap, min_distance=10, thresh=0.0):
    """Tìm cực đại cục bộ trên heatmap -> danh sách (x,y,score) ở toạ độ heatmap."""
    mx = maximum_filter(heatmap, size=min_distance)
    mask = (heatmap == mx) & (heatmap > thresh)
    ys, xs = np.where(mask)
    scores = heatmap[ys, xs]
    order = np.argsort(-scores)
    return [(int(xs[i]), int(ys[i]), float(scores[i])) for i in order]


def match_detections(preds, nodules, stride, mm_per_px, hit_radius_mm):
    """preds: [(x,y,score)] toạ độ heatmap. nodules: [{x,y,subtlety}] toạ độ ảnh.
    Trả về danh sách (score, is_tp) và tập subtlety của nốt đã trúng."""
    radius_px = hit_radius_mm / mm_per_px
    unmatched = [{"x": n["x"], "y": n["y"], "subtlety": n["subtlety"],
                  "hit": False, "hit_score": None}
                 for n in nodules]
    results, hit_subtlety = [], {}
    for (px, py, score) in preds:
        ix, iy = px * stride, py * stride  # về toạ độ ảnh gốc đã xử lý
        best, best_d = -1, 1e9
        for j, nd in enumerate(unmatched):
            if nd["hit"]:
                continue
            d = np.hypot(ix - nd["x"], iy - nd["y"])
            if d < best_d:
                best_d, best = d, j
        if best >= 0 and best_d <= radius_px:
            unmatched[best]["hit"] = True
            unmatched[best]["hit_score"] = score   # LƯU score để áp ngưỡng sau
            hit_subtlety[best] = unmatched[best]["subtlety"]
            results.append((score, True))
        else:
            results.append((score, False))
    return results, unmatched


def compute_froc(all_results, n_images, n_total_nodules, fp_rates):
    """all_results: list các (score, is_tp) gộp toàn bộ ảnh.
    Trả về (fppi_curve, sens_curve, sens_at_fp{rate:val}, cpm)."""
    if not all_results or n_total_nodules == 0:
        return np.array([]), np.array([]), {r: 0.0 for r in fp_rates}, 0.0
    arr = sorted(all_results, key=lambda x: -x[0])
    tp = fp = 0
    fppi, sens = [], []
    for score, is_tp in arr:
        if is_tp:
            tp += 1
        else:
            fp += 1
        fppi.append(fp / n_images)
        sens.append(tp / n_total_nodules)
    fppi, sens = np.array(fppi), np.array(sens)

    sens_at = {}
    for r in fp_rates:
        idx = np.where(fppi <= r)[0]
        sens_at[r] = float(sens[idx[-1]]) if len(idx) else 0.0
    # CPM chuẩn: trung bình tại 1/8,1/4,1/2,1,2,4 (giao với fp_rates cung cấp)
    cpm_pts = [r for r in (0.125, 0.25, 0.5, 1, 2, 4) if r in sens_at]
    cpm = float(np.mean([sens_at[r] for r in cpm_pts])) if cpm_pts else 0.0
    return fppi, sens, sens_at, cpm


def sensitivity_by_subtlety(matched_per_image, target_fp_per_image, n_images):
    """Độ nhạy tách theo mức subtlety, tại một ngưỡng điểm sao cho FP/ảnh ~ target.
    matched_per_image: list theo ảnh, mỗi phần tử (results, unmatched_nodules).
    Trả về {subtlety: (n_hit, n_total, sensitivity)}.
    Nốt được coi TRÚNG nếu detection trúng nó có score >= ngưỡng (nhất quán FROC)."""
    # gộp tất cả FP để tìm ngưỡng đạt target FP/ảnh
    all_fp_scores = sorted(
        [s for (res, _) in matched_per_image for (s, tp) in res if not tp],
        reverse=True)
    budget = int(round(target_fp_per_image * n_images))
    thresh = all_fp_scores[budget - 1] if 0 < budget <= len(all_fp_scores) \
        else (all_fp_scores[-1] if all_fp_scores else 0.0)

    from collections import defaultdict
    hit = defaultdict(int); tot = defaultdict(int)
    for (res, unmatched) in matched_per_image:
        for nd in unmatched:
            tot[nd["subtlety"]] += 1
            # trúng chỉ khi có detection trúng nốt VÀ score >= ngưỡng
            if nd["hit"] and nd["hit_score"] is not None \
                    and nd["hit_score"] >= thresh:
                hit[nd["subtlety"]] += 1
    out = {}
    for s in sorted(tot):
        out[s] = (hit[s], tot[s], hit[s] / tot[s] if tot[s] else 0.0)
    return out


def format_subtlety_table(by_sub):
    labels = {1: "1 Cực khó thấy", 2: "2 Rất khó thấy", 3: "3 Trung bình",
              4: "4 Tương đối rõ", 5: "5 Rõ ràng"}
    lines = ["  Subtlety            Hit/Total   Sensitivity",
             "  " + "-" * 44]
    for s in sorted(by_sub):
        hit, tot, sens = by_sub[s]
        lines.append(f"  {labels.get(s, s):<18} {hit:>4}/{tot:<5}   {sens*100:5.1f}%")
    return "\n".join(lines)
