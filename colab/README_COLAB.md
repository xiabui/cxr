# Chạy trên Colab Pro

## Một lần duy nhất

**1. Tạo repo riêng trên GitHub** (tên tuỳ ý, ví dụ `cxr-nodule`), để **Private**.

**2. Đẩy mã nguồn lên** — chạy trên máy Mac, trong thư mục `~/Papers/CXR`:

```bash
git remote add origin https://github.com/<tên-tài-khoản>/cxr-nodule.git
git branch -M main
git push -u origin main
```

**3. Tạo GitHub token** tại <https://github.com/settings/tokens> →
*Fine-grained tokens* → chỉ cấp quyền **Contents: Read** cho đúng repo này.

**4. Lưu token vào Colab Secrets** — biểu tượng chìa khoá ở thanh bên trái,
tên biến đúng là `GH_TOKEN`, bật *Notebook access*.
Đừng dán token vào ô code: notebook có thể bị chia sẻ kèm token.

**5. Tải dữ liệu lên Drive.** File `colab/cxr_colab_data.tar.gz` (83 MB) đã có
sẵn trên máy Mac. Kéo thả vào thư mục gốc **My Drive**. Gói gồm:

| Nội dung | Số lượng |
|---|---|
| Ảnh JSRT đã xử lý | 247 |
| Ảnh đã khử xương | 247 |
| `labels_processed.csv`, `transforms.json` | nhãn nốt + subtlety |
| `bone_suppression_unet.pt` | U-Net khử xương |

Dữ liệu không nằm trong repo vì `.gitignore` loại toàn bộ `data/` —
GitHub không phải chỗ để dữ liệu y tế.

## Mỗi phiên làm việc

Mở `colab/raddino_finetune_colab.ipynb` bằng Colab, **bật GPU**
(Runtime → Change runtime type → A100 hoặc L4), rồi chạy lần lượt từ trên xuống.

Khi mình sửa code, chỉ cần chạy lại **ô số 2** — nó `git pull`, không clone lại.

## Sau khi chạy xong

Ô cuối chép `runs/*.json` về `MyDrive/cxr_runs/`. Colab xoá `/content` khi ngắt
phiên nên bước này bắt buộc, nếu không mất hết kết quả.

## Vì sao mỗi fold dựng lại backbone

Tinh chỉnh làm thay đổi trọng số backbone. Nếu fold sau dùng lại backbone đã
tinh chỉnh ở fold trước thì backbone đã "nhìn thấy" ảnh kiểm tra của fold sau —
**rò rỉ dữ liệu**, và độ nhạy sẽ cao giả tạo. Vì vậy chậm là có chủ đích.

`CLAUDE.md` mục 5 đã ghi một lỗi cùng họ: *"Eval trên chính tập train → độ nhạy
100% mọi nhóm"*.
