# Bản thảo bài báo — việc còn phải làm

## NƠI NỘP: Journal of Imaging Informatics in Medicine (Springer)

**Q1, và MIỄN PHÍ nếu chọn mô hình thuê bao** khi bài được nhận. Chỉ mất tiền
nếu chọn open access ($4.890) — đừng chọn.

Vì sao chọn nơi này: chính tạp chí đã đăng bài khử bóng xương Ibrahim 2025 mà
ta trích. Biên tập viên và người phản biện đã quen mảng này, và bài của ta
chính là phần hạ nguồn mà bài đó bỏ ngỏ (họ đo SSIM/PSNR, ta đo ảnh hưởng tới
phát hiện).

Trượt thì hạ xuống **IJCARS** (Springer, Q2, cũng miễn phí) — không phải viết
lại. Phương án nhanh: **MELBA** (chỉ $10, phản biện nhanh, nhưng chưa xếp hạng Q).

**Tránh:** IEEE Access ($1.995), Scientific Reports (~$2.490), MDPI, Frontiers,
PLOS — đều thu phí bắt buộc.

Miễn giảm APC gần như không dùng được: hệ thống kiểm tự động qua đơn vị công
tác, mà ta không có đơn vị. Nên đường thuê bao an toàn hơn hẳn.

---

## HAI VIỆC CHỈ HAI TÁC GIẢ LÀM ĐƯỢC

1. **Mục Author contributions** — lời khai ai làm gì, không ai viết hộ được.
   Trong `main.tex` đã có bản nháp gợi ý, sửa cho đúng thực tế rồi xoá `\todo`.
2. **Đổi repo `github.com/xiabui/cxr` sang PUBLIC trước khi nộp.** Mục Data
   availability đang ghi code có sẵn ở đó — repo còn riêng tư thì câu đó SAI.
   Không muốn công khai thì sửa thành "available from the corresponding author
   on reasonable request".

---

## Hai con số nên tự đối chiếu bản gốc

- **838.000 ảnh** của RAD-DINO — nằm trong abstract nên sai là thấy ngay.
- `gusarev2017bonesup` có đúng là nguồn bộ 241 cặp ảnh trên Kaggle không.

---

## Trạng thái

Bài 14 trang, biên dịch sạch, không còn `PENDING`. Abstract 256 từ (giới hạn
Springer thường là 250 — sát, nhưng nếu tạp chí đòi chặt thì cắt đoạn cuối).
Đã có: 12 trích dẫn, 3 hình, kiểm định ngoại có khoảng tin cậy, mục Declarations
theo đúng yêu cầu Springer.

Springer nhận nộp tự do định dạng ở vòng đầu, nên KHÔNG cần đổi sang `sn-jnl.cls`
lúc này. Chỉ đổi khi bài được nhận.

## Còn chạy trên máy chủ

LOOCV có gieo hạt, để thay Bảng 1 (hiện là số chạy đơn lẻ, đã ghi rõ cảnh báo
trong chú thích bảng). Không chặn việc nộp.

---

## Ghi chú cũ

## 1. Trích dẫn — ĐÃ XONG (15/09/2026)

Đã tra cứu và xác minh, không còn mục nào trống. 12 mục trong `refs.bib`.

Hai mục bắt buộc đều tìm được:

| Khoá | Bài | Cách xác minh |
|---|---|---|
| `horry2023fullres` | Horry và cs. 2023, IEEE Access 11 | tải PDF, đọc toàn văn: có bảng phân tầng subtlety, có câu đề xuất khử xương |
| `muthyala2026frozen` | Muthyala và cs. 2026, arXiv 2606.11606 | đọc abstract đầy đủ: có thử RAD-DINO, nhưng là phân loại chứ không phải phát hiện |

Còn hai mục nên đối chiếu lại bản gốc trước khi nộp:
- `gusarev2017bonesup` — xác nhận đây đúng là nguồn 241 cặp ảnh trên Kaggle.
- `perezgarcia2024raddino` — con số **838.000 ảnh** nằm trong abstract, sai là thấy ngay.

### Kết luận tra cứu trùng lặp: KHÔNG TRÙNG

Horry và cs. **đề xuất** khử bóng xương và ghi rõ trong Kết luận rằng nhóm họ
"đang làm" — nhưng tra cứu cho thấy **chưa công bố**. Bài khử xương 2025 tìm
được là của nhóm Ibrahim, và chỉ đo chất lượng ảnh (SSIM/PSNR), không đo ảnh
hưởng tới phát hiện. Khoảng trống vẫn còn nguyên.

Muthyala và cs. 2026 có thử chính RAD-DINO, nhưng là **phân loại (AUC)** trên
NIH/MIMIC/Emory/ChestX-Det10 — không phải phát hiện trên JSRT, không dùng
subtlety, không động tới khử xương hay tăng cường, không có khoảng tin cậy.

Đọc bài đó còn được thêm một luận điểm: họ kết luận tín hiệu mất ở bước **gộp
toàn cục** và giữ được ở **patch token**. Decoder của ta dùng patch token, nên
thất bại ở nhóm subtlety 1 **không** giải thích được bằng gộp toàn cục. Đã đưa
vào Related Work.

## 1b. Ghi chú cũ về trích dẫn

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
