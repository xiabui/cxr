# Gói mã nguồn — CXR Nodule Detection

Đọc `CLAUDE.md` trước. Đó là file bối cảnh đầy đủ của dự án.

## Nội dung gói

| File | Vai trò |
|---|---|
| `CLAUDE.md` | **Bối cảnh dự án** — đọc đầu tiên |
| `vindr_prepare.py` | Chuẩn bị VinDr: DICOM sang PNG, gộp hộp của 3 bác sĩ |
| `external_vindr.py` | External validation trên VinDr (bản viết lại) |
| `raddino_export_head.py` | Train decoder head trên toàn bộ JSRT rồi lưu |
| `raddino_extract2.py` | Trích xuất đặc trưng RAD-DINO có augmentation + kênh khử xương |
| `raddino_train3.py` | Train head, ngưỡng nhất quán, chia fold theo ảnh |
| `raddino_extract.py` | Bản trích xuất đơn giản (3 kênh lặp) |
| `raddino_train2.py` | Bản train đơn giản, dùng cho cấu hình 36,5% |
| `bootstrap_ci.py` | **Khoảng tin cậy bootstrap** (mục 8.1) — lấy mẫu lại theo ảnh, có chế độ so sánh bắt cặp |
| `run_seeds.py` | Chạy nhiều seed rồi tổng hợp trung bình ± độ lệch chuẩn (mục 8.3) |

## Quy trình thống kê

`bootstrap_ci.py` đọc `runs/scores_*.json` (điểm thô gom theo từng ảnh) mà
`raddino_train3.py` và `external_vindr.py` sinh ra. Không cần GPU, chạy vài giây.

```bash
# khoảng tin cậy cho một lần chạy
python bootstrap_ci.py runs/scores_raddino_aug_loco_seed42.json

# hiệu số giữa hai cấu hình, bắt cặp trên cùng tập ảnh
python bootstrap_ci.py --compare runs/scores_A.json runs/scores_B.json

# nhiều seed
python run_seeds.py --config configs/jsrt_10fold.yaml --seeds 0 1 2 3 4
```

Thư mục `src/` và dữ liệu **không nằm trong gói này**, chúng ở trên máy tại
`/home/bvxia/workspace/cancer-detection/cxr_nodule/`.

## Cài đặt

```bash
conda activate ai_env
cd ~/workspace/cancer-detection/cxr_nodule
# giải nén gói này vào đây, các script nằm cùng cấp với src/
pip install transformers pydicom
```

## Dùng với Claude Code

```bash
cd ~/workspace/cancer-detection/cxr_nodule
claude
```

Claude Code sẽ tự đọc `CLAUDE.md`. Nếu không, nhắc nó đọc file đó trước.

## Lưu ý chung khi chạy

Luôn đặt biến môi trường chống phân mảnh bộ nhớ, vì card chỉ có 6GB:

```bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

Khi chạy dài như LOOCV, dùng tmux và tăng giới hạn file descriptor:

```bash
ulimit -n 65535
tmux new -s run
```

---

## Bài báo

| | |
|---|---|
| Bản thảo tiếng Anh | `paper/main.tex` → `paper/main.pdf` |
| Bản thảo tiếng Việt | `paper/main_vi.tex` → `paper/main_vi.pdf` |
| Việc còn lại + hướng dẫn | `paper/README_PAPER.md` |

**Nộp bài:** <https://www.editorialmanager.com/jdim/default2.aspx>
(Journal of Imaging Informatics in Medicine, Springer, Q1)

Khi hệ thống hỏi mô hình xuất bản, **chọn subscription** — miễn phí. Chọn open
access là mất $4.890.

Biên dịch:

```bash
cd paper
pdflatex main.tex && bibtex main && pdflatex main.tex && pdflatex main.tex
xelatex main_vi.tex && bibtex main_vi && xelatex main_vi.tex && xelatex main_vi.tex
```

Bản tiếng Việt phải dùng **xelatex**, không dùng pdflatex: pdflatex nuốt mất các
chữ như ầ và ị mà không báo lỗi.
