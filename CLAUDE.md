# CXR Nodule Detection — Project Context

Tài liệu này là bộ nhớ dự án cho Claude Code. Đọc file này trước khi làm bất cứ việc gì.

**Trạng thái:** luận văn đã bảo vệ. Giai đoạn hiện tại là **viết bài báo quốc tế**.
**Mục tiêu công bố:** tạp chí Q2 (trần thực tế), hoặc workshop MICCAI / IEEE EMBC.

---

## 1. Bài toán

Phát hiện nốt phổi **khó thấy** trên ảnh X-quang ngực, đánh giá **phân tầng theo
subtlety** thay vì chỉ báo cáo chỉ số trung bình.

Luận điểm trung tâm: chỉ số trung bình bị nhóm nốt dễ (chiếm đa số) kéo lên, che
giấu việc mô hình thất bại ở nhóm khó, mà nhóm khó mới là nhóm có giá trị lâm sàng
vì nốt rõ thì bác sĩ vốn đã thấy.

---

## 2. Môi trường và đường dẫn

```
Máy:        homeserver, user bvxia, RTX 3060 Mobile 6GB VRAM, driver 595
Conda env:  ai_env  (~/miniconda3/envs/ai_env)  — KHÔNG phải cxr_env
Project:    /home/bvxia/workspace/cancer-detection/cxr_nodule/
```

Mỗi phiên SSH mới mở ở `base`, phải `conda activate ai_env` trước.

Ràng buộc lặp lại nhiều lần: **6GB VRAM**. Luôn chạy với
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`. Nếu OOM thì giảm batch.

---

## 3. Dữ liệu

| Tập | Vai trò | Ghi chú |
|---|---|---|
| JSRT (247 ảnh) | fine-tune + đánh giá | 154 nốt + 93 ảnh bình thường; nhãn subtlety 1-5 từ ROC của 20 bác sĩ; nốt xác nhận bằng CT |
| NIH ChestX-ray14 (112.120 ảnh) | pre-train backbone | nhãn khai thác tự động từ báo cáo nên **có nhiễu**; chỉ dùng học biểu diễn |
| bone_shadow (241 cặp) | train U-Net khử xương | Kaggle `hmchuong/xray-bone-shadow-supression` |
| VinDr-CXR | external validation (ĐANG LÀM) | xem mục 7 |

Phân bố nốt theo subtlety: **25 / 29 / 50 / 38 / 12** (mức 1→5).
Hai nhóm biên rất ít mẫu, mọi chênh lệch phải diễn giải thận trọng.

Cấu trúc thư mục JSRT lồng hai lớp: ảnh nằm ở `data/jsrt/images/images/`, nên
config phải đặt `images_dir: images/images`.

---

## 4. Kết quả (con số chính thức)

### LOOCV — dùng cho bài báo

| Cấu hình | Thuật toán | CPM |
|---|---|---|
| Baseline | ResNet34 | **29,5%** |
| + bone suppression + attention | ResNet34 + U-Net + CBAM | **34,2%** |
| Foundation model | RAD-DINO (frozen) + U-Net | **42,5%** |

Độ nhạy phân tầng tại 1,0 FP/ảnh:

| Mức | Baseline | ResNet34+U-Net+CBAM | RAD-DINO+U-Net |
|---|---|---|---|
| 1 Cực khó | 0,0% | 8,0% | 4,0% |
| 2 Rất khó | 6,9% | 20,7% | 27,6% |
| 3 Trung bình | 20,0% | 32,0% | 48,0% |
| 4 Tương đối rõ | 68,4% | 57,9% | 73,7% |
| 5 Rõ ràng | 83,3% | 58,3% | 75,0% |

### Ablation 10-fold

ResNet34: baseline 29,3% | +U-Net 29,7% | +U-Net+CBAM 37,6% | +U-Net+weighted loss 33,3%

RAD-DINO (frozen): 3 kênh lặp 36,5% | +U-Net **44,4%** | +U-Net+augmentation **23,9%**

### CẬP NHẬT 13/09/2026 — đã có khoảng tin cậy, phát hiện 1 KHÔNG đứng vững

Chạy lại 10-fold seed 42 kèm bootstrap bắt cặp (`bootstrap_ci.py`):

| Đối chiếu | Hiệu số CPM | Khoảng tin cậy 95% | Kết luận |
|---|---|---|---|
| Khử xương (3 kênh → +U-Net) | **+2,8** | **[−0,9; +6,3]** | CHỨA 0 — không kết luận được |
| Tăng cường khi backbone đóng băng | **−22,1** | **[−28,2; −16,1]** | vững chắc |

Hai điều buộc phải ghi nhớ:

1. **Số 7,9 điểm của phát hiện 1 không tái lập được.** Chạy lại đúng cấu hình
   đã cho 36,5% thì ra **40,5%** — chênh 4 điểm chỉ do khởi tạo ngẫu nhiên, vì
   `raddino_train2.py` trước đây KHÔNG gieo hạt cho torch (đã sửa). Hiệu ứng
   7,9 nhiều khả năng là ~2,8 thật cộng nhiễu thuận chiều ở cả hai đầu.
2. **Phát hiện 2 và 3 vững.** Tăng cường làm sụt 22 điểm, mọi mức FP đều có
   khoảng tin cậy không chứa 0. Số đo mới: B=43,3% → C=21,2%.

**Quan sát mới, đáng đưa vào bài:** tăng cường làm sụt 22 điểm tổng thể và 42
điểm ở nhóm 5, nhưng nhóm 1 **không đổi** (+0,0 [−11,5; +11,5]). Nhóm 1 đã ở
mức sàn 4% (1/25) nên không còn gì để mất. Nhóm khó nhất trơ với cả thay đổi
có lợi lẫn thay đổi có hại.

Bản thảo trong `paper/` đã được định khung lại theo bằng chứng này. Đọc
`paper/README_PAPER.md` mục 3 trước khi sửa khung bài.

### Ba phát hiện chính của bài báo (KHUNG CŨ — xem cập nhật ở trên)

1. **Bone suppression nhất quán qua hai kiến trúc.** Cộng 7,9 điểm ngay cả với
   backbone đóng băng. Bằng chứng chéo kiến trúc mạnh hơn quan sát đơn lẻ.
2. **Augmentation phản tác dụng khi backbone frozen.** Trừ 20,5 điểm. Vì backbone
   không thích nghi được, ảnh biến đổi lệch khỏi phân bố pre-training.
3. **Nhóm subtlety 1 kháng lại mọi cấu hình**, dao động 0-8% kể cả với mô hình
   học từ 882.775 ảnh. Quy mô pre-training không tự giải quyết được nốt khó nhất.

Lưu ý: chênh lệch nhóm 1 giữa 8,0% và 4,0% chỉ là **một nốt trên 25**, nằm trong
dao động ngẫu nhiên. Không được diễn giải là cấu hình nào tốt hơn.

---

## 5. Các lỗi đã mắc (đừng lặp lại)

| Lỗi | Triệu chứng | Cách sửa |
|---|---|---|
| Tâm Gaussian không đúng 1.0 | model không học được gì | ép `hm[iy,ix]=1.0` sau khi làm tròn |
| Mất cân bằng dương/âm | model xuất giá trị thấp khắp nơi | `pos_weight` 40x |
| **Ngưỡng FROC vs subtlety lệch nhau** | bảng subtlety cho số vô lý (84%) | dùng CÙNG ngưỡng theo mức FP. **Đã mắc HAI lần** |
| Bùng nổ gradient | loss thành inf rồi NaN | clip norm 5.0 + clamp 1e-4 + giảm LR |
| **AMP gây tràn số** | loss bằng đúng 0, không học | `amp: false`. Đây là breakthrough lớn nhất |
| Chạy nhầm code cũ | kết luận sai rằng attention vô dụng | kiểm tra phiên bản trước khi kết luận |
| Eval trên chính tập train | độ nhạy 100% mọi nhóm | model `train_full` CHỈ để vẽ hình |
| Đổi 2 yếu tố cùng lúc | không quy kết được nguyên nhân | ablation phải đổi từng cái một |
| fd leak khi LOOCV | crash ở fold 119, "Too many open files" | `persistent_workers=False`, `ulimit -n 65535` |

---

## 6. Cấu trúc mã nguồn

```
src/                          (nằm trên máy, không có trong gói này)
  dataset.py       JSRTNoduleDataset — trả về (img, heatmap, wmap, meta) 4-tuple
  model.py         HeatmapDetector, CBAM, focal_mse_loss(weight_map=...)
  train.py         train_one_fold, train_full, load_full_model, list_images
  froc.py          extract_peaks, match_detections, compute_froc
  bone_suppression.py  UNet + train_bone_suppression
  gen_bs_channel.py    sinh images_bs/ từ U-Net
  preprocess.py, parse_jsrt_labels.py, pretrain_nih.py, analysis.py

Scripts RAD-DINO (trong gói này)
  raddino_extract.py      trích xuất đặc trưng đơn giản (3 kênh lặp)
  raddino_extract2.py     trích xuất CÓ augmentation + kênh khử xương
  raddino_train3.py       train head, ngưỡng nhất quán, chia fold theo ẢNH
  raddino_export_head.py  train head trên toàn bộ JSRT rồi lưu (cho external)

External validation (trong gói này)
  vindr_prepare.py        DICOM→PNG, gộp hộp của 3 bác sĩ
  external_vindr.py       inference + FROC trên VinDr

Thống kê (trong gói này)
  bootstrap_ci.py         khoảng tin cậy BCa, lấy mẫu lại THEO ẢNH; có chế độ
                          --compare so sánh hai cấu hình CÓ BẮT CẶP
  run_seeds.py            chạy nhiều seed rồi tổng hợp trung bình ± độ lệch chuẩn
```

`raddino_train3.py` và `external_vindr.py` nay ghi thêm `runs/scores_*.json`
chứa điểm thô **gom theo từng ảnh**. Đây là đầu vào bắt buộc của `bootstrap_ci.py`:
không có điểm thô thì không tính lại được ngưỡng trong mỗi lần lấy mẫu.

Ba tham số bật/tắt đóng góp trong config:
`in_channels: 2` (U-Net) · `use_attention: true` (CBAM) · `subtlety_weighted_loss: true`

---

## 7. External validation trên VinDr-CXR (việc đang làm)

### Điều cần biết về VinDr

- 18.000 ảnh PA, train 15.000 / test 3.000, định dạng **DICOM**
- **Tập train: 3 bác sĩ gán nhãn độc lập** → cùng một nốt có tới 3 hộp trùng.
  Không gộp thì đếm sai số nốt. Script `vindr_prepare.py` gộp theo IoU.
- **Tập test: đồng thuận 5 bác sĩ** → sạch hơn, **nên dùng tập này**.
- Cột: `image_id, rad_id, class_name, x_min, y_min, x_max, y_max`
  (tập test **không có** `rad_id`)
- Nhãn cần lọc: `class_name == "Nodule/Mass"`
- DICOM có thể là MONOCHROME1 (đảo màu), phải xử lý + apply VOI LUT

### Quy trình

```bash
conda activate ai_env
cd ~/workspace/cancer-detection/cxr_nodule

# 1. chuẩn bị dữ liệu (ưu tiên tập test)
python vindr_prepare.py --dicom-dir /data/vindr/test \
    --ann /data/vindr/annotations_test.csv \
    --out-dir data/vindr --size 512 --include-normal 500

# 2. train head trên toàn bộ JSRT rồi lưu
python raddino_export_head.py --config configs/jsrt_10fold.yaml

# 3. chạy external validation
python external_vindr.py --vindr-dir data/vindr --backbone raddino \
    --bs-ckpt checkpoints/bone_suppression_unet.pt \
    --head checkpoints/raddino_head.pt
```

### Quyết định phương pháp quan trọng

Tiêu chí hit trên VinDr là **đỉnh dự đoán nằm trong hộp thật**, không dùng bán
kính mm. Lý do: nhãn VinDr là bounding box, và pixel spacing của DICOM khác nhau
giữa các ảnh nên quy đổi mm không tin cậy.

Hệ quả: **CPM trên VinDr KHÔNG so sánh trực tiếp được với CPM trên JSRT** vì hai
tiêu chí hit khác nhau. Phải nêu rõ điều này trong bài báo. Chỉ dùng để đánh giá
mức độ tổng quát hóa một cách định tính.

---

## 8. Việc còn lại trước khi nộp bài

Xếp theo tỷ lệ lợi ích trên công sức:

1. ~~**Bootstrap khoảng tin cậy**~~ — CÔNG CỤ ĐÃ XONG, còn phải CHẠY trên máy.
   `bootstrap_ci.py` đã viết và đã kiểm thử trên dữ liệu giả. Chạy lại
   `raddino_train3.py` một lần để sinh `runs/scores_*.json`, rồi:
   ```bash
   python bootstrap_ci.py runs/scores_raddino_aug_loco_seed42.json
   python bootstrap_ci.py --compare runs/scores_<baseline>.json \
                                    runs/scores_<raddino>.json
   ```
   Chế độ `--compare` mới là thứ trả lời được phản biện "7,9 điểm có phải nhiễu
   không": nó cho khoảng tin cậy của HIỆU SỐ, bắt cặp trên cùng 247 ảnh.
2. **External validation VinDr** (mục 7). Đang làm. Script đã sẵn sàng và nay
   cũng ghi `runs/scores_vindr.json` để chạy bootstrap được.
3. **Chạy nhiều seed** — `run_seeds.py` đã viết, `raddino_train3.py` đã có `--seed`.
   Còn phải chạy trên máy.
4. Tinh chỉnh một phần tầng cuối RAD-DINO — tốt nhưng khó trên 6GB.

**Ba quy tắc thống kê không được vi phạm** (đã mã hóa trong `bootstrap_ci.py`):

- Lấy mẫu lại theo **ẢNH**, không theo nốt. Ngưỡng FROC định nghĩa theo số FP
  trên mỗi ảnh, tách nốt khỏi ảnh của nó là làm ngưỡng vô nghĩa.
- **Tính lại ngưỡng trong từng lần lấy mẫu.** Dùng ngưỡng cố định của tập gốc
  cho khoảng hẹp giả tạo. Đây là biến thể của lỗi ngưỡng ở mục 5.
- So sánh hai cấu hình phải **bắt cặp** (cùng chỉ số ảnh cho cả hai), nếu không
  khoảng tin cậy rộng một cách vô ích vì không triệt tiêu biến thiên chung.

## 9. Cần biết khi viết bài

Không được tuyên bố là người đầu tiên làm bất cứ điều gì. Cụ thể:

- **Đánh giá phân tầng theo subtlety đã có công trình làm.** Có bài dùng kiến trúc
  encoder-decoder full-resolution trên JSRT đã phân tích theo subtlety, và còn
  **đề xuất bone suppression như hướng khắc phục**. Định vị đúng: luận văn kiểm
  chứng chính đề xuất mà họ để ngỏ.
- **Phát hiện về mô hình nền đóng băng làm mất tín hiệu tổn thương nhỏ đã được
  công bố độc lập** trong một bài 2026. Coi đây là tái lập độc lập, phải trích dẫn.
- JSRT đã nằm trong tập huấn luyện của NODE21, benchmark chuẩn hiện nay.

Hạn chế phải nêu rõ: dữ liệu nhỏ, một nguồn duy nhất, backbone đóng băng chứ chưa
tinh chỉnh, và hiệu năng còn dưới các công bố khác trên cùng JSRT.
