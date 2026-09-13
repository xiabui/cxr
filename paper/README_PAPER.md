# Bản thảo bài báo — việc còn phải làm

## Trạng thái

Khung bài đã đủ: Introduction, Related Work, Methods, Results, Discussion,
Limitations, Conclusion. Phần **Methods viết xong hẳn** vì phương pháp đã chốt.
Phần **Results còn chờ số** — xem bảng dưới.

Biên dịch thử:

```bash
cd paper && pdflatex main.tex && bibtex main && pdflatex main.tex && pdflatex main.tex
```

Mọi chỗ chưa xong hiện **màu đỏ** (`\todo`) hoặc **màu cam** (`\pending`) ngay
trên bản PDF. Còn thấy màu là chưa nộp được.

---

## 1. Trích dẫn — việc nặng nhất, và mình KHÔNG làm thay được

`refs.bib` chia hai nhóm.

**Nhóm 1 (6 mục):** mình điền sẵn từ hiểu biết chung — JSRT, ChestX-ray14,
CBAM, RAD-DINO, NODE21, VinDr. Vẫn **phải đối chiếu** số tập, số trang, năm.
Mình không tra cứu được bản gốc nên không đảm bảo từng chi tiết thư mục.

Ưu tiên đối chiếu hai mục:
- `perezgarcia2024raddino` — con số **838.000 ảnh** được viết trong **abstract**.
  Sai con số trong abstract là lỗi thấy ngay.
- `sogancioglu2024node21` — cần xác nhận **JSRT thật sự nằm trong tập huấn
  luyện NODE21**, vì bài dùng điều đó để từ chối so sánh.

**Nhóm 2 (8 mục):** mình để **trống có chủ ý**. Đây là các bài bạn đã đọc mà
mình không biết là bài nào. Bịa tên tác giả và năm sẽ tạo trích dẫn giả — lỗi
nặng nhất trong một bài báo, và người phản biện phát hiện ngay. Bạn tự điền.

Hai mục **bắt buộc**, thiếu là bài mất chỗ đứng:

| Khoá | Là bài nào | Vì sao bắt buộc |
|---|---|---|
| `TODO-subtlety-prior` | Bài dùng encoder-decoder full-resolution phân tích JSRT theo subtlety, **đề xuất khử xương** làm hướng khắc phục | Toàn bộ định vị của bài ta là *kiểm chứng đề xuất họ để ngỏ*. Không có nó, ta thành người tự nhận nghĩ ra đánh giá phân tầng — đúng điều `CLAUDE.md` mục 9 cấm |
| `TODO-frozen-foundation-2026` | Bài 2026 công bố độc lập việc mô hình nền đóng băng làm mất tín hiệu tổn thương nhỏ | Ta **tái lập** kết quả này. Không dẫn thì thành tuyên bố phát hiện mới |

---

## 2. Số còn thiếu

| Chỗ | Cần gì | Lấy ở đâu |
|---|---|---|
| Bảng 1 (LOOCV) | khoảng tin cậy cho 3 dòng | chạy LOOCV có ghi scores, rồi `bootstrap_ci.py` |
| §4.4 | bảng trung bình ± độ lệch chuẩn qua 5 seed | `run_seeds.py` — **đang chạy trên homeserver** |
| §4.5 | kết quả VinDr-CXR | `vindr_prepare.py` + `external_vindr.py` |
| §5 "Frozen backbones" | kết quả tinh chỉnh một phần | notebook Colab, nhánh A và B |

Bảng 1 hiện là số LOOCV cũ **chưa có khoảng tin cậy**. Bảng 3 (so sánh bắt cặp)
là số 10-fold **đã chạy và đã kiểm chứng hôm nay**.

---

## 3. Quyết định quan trọng: bài đã được định khung lại

**Phải đọc mục này trước khi sửa bất cứ đâu.**

`CLAUDE.md` mục 4 ghi ba phát hiện, trong đó phát hiện 1 là *"khử xương nhất
quán qua hai kiến trúc, cộng 7,9 điểm"*.

Chạy lại hôm nay: **+2,8 điểm, khoảng tin cậy 95% là [−0,9; +6,3]** — chứa số 0.
Thêm nữa, chạy lại đúng cấu hình đã cho 36,5% thì ra **40,5%**, chênh 4 điểm chỉ
do khởi tạo ngẫu nhiên (`raddino_train2.py` trước đây không gieo hạt cho torch).

Hiệu ứng 7,9 điểm nhiều khả năng là khoảng 2,8 điểm thật cộng với nhiễu khởi
tạo thuận chiều ở cả hai đầu.

Cho nên bản thảo **không** viết "khử xương cộng 7,9 điểm". Nó viết:

> Phân tầng làm lộ ra sự đánh đổi mà chỉ số trung bình che mất; nhưng khi tính
> khoảng tin cậy thì phần lớn chênh lệch giữa các cấu hình trên bộ 247 ảnh
> **không tách được khỏi nhiễu** — trong khi thất bại ở nhóm khó nhất thì
> **vững**, tồn tại qua mọi cấu hình.

Vì sao chọn khung này:

1. **Nó đúng.** Khung cũ không chống được một người phản biện chỉ cần chạy thêm
   một seed.
2. **Phát hiện 3 không hề suy suyển** — và nó vốn là phát hiện mạnh nhất. Nhóm
   subtlety 1 nằm ở 0–8% qua mọi cấu hình, kể cả mô hình học từ 838.000 ảnh.
3. **Nó hợp với luận điểm trung tâm ở mục 1 của `CLAUDE.md`**: chỉ số trung
   bình che giấu thất bại ở nhóm khó. Bài chỉ đơn giản đi thêm một bước —
   khoảng tin cậy còn che giấu nhiều hơn thế.
4. Một bài cảnh báo về phương pháp, có công cụ kèm theo, **rất hợp workshop
   MICCAI / EMBC** — đúng đích đã đặt.

Nếu bạn muốn quay lại khung cũ, cần chạy nhiều seed và cho ra khoảng tin cậy
không chứa 0. Kết quả 5 seed đang chạy sẽ trả lời. **Đừng viết lại theo khung
cũ trước khi có số đó.**

---

## 4. Đổi định dạng khi nộp

Thân bài không phụ thuộc lớp tài liệu. Chỉ đổi dòng `\documentclass`:

| Đích | Lớp |
|---|---|
| MICCAI workshop | `\documentclass{llncs}` (cần `llncs.cls`) |
| IEEE EMBC | `\documentclass[conference]{IEEEtran}` |
| Tạp chí Q2 | theo mẫu tạp chí |

EMBC giới hạn 4 trang — sẽ phải cắt mạnh; cắt Related Work và Limitations
trước, giữ nguyên Results.

---

## 5. Chưa có hình

Bài hiện **không có hình nào**. Tối thiểu nên có hai:

1. **Đường cong FROC** ba cấu hình, có dải tin cậy bootstrap.
2. **Độ nhạy theo mức subtlety**, dạng cột có thanh sai số — đây là hình mang
   luận điểm chính, thanh sai số rộng chính là thông điệp.

Cả hai vẽ được từ `runs/scores_*.json` mà không phải huấn luyện lại. Bảo mình
khi cần.
