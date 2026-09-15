#!/bin/bash
# Quet muc noi bien hop, de TACH hai nguyen nhan cua muc sut giam:
#   (a) tieu chi hit khat khe hon that su, va
#   (b) mo hinh tong quat hoa kem.
# Dung sai tren VinDr (nam trong hop) trung vi chi 7,6 px, trong khi JSRT cho
# phep 15 mm = 21,4 px. Noi bien 14 px dua dung sai ve xap xi muc cua JSRT.
set -u
source ~/miniconda3/etc/profile.d/conda.sh
conda activate ai_env
cd ~/workspace/cancer-detection/cxr_nodule
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# Khong dem stdout. Neu khong, tien do fold nam trong bo dem va log rong
# suot nhieu gio — khong biet chay toi dau, tuong nhu treo.
export PYTHONUNBUFFERED=1
rm -f runs/MARGIN_DONE
for m in 7 14 21; do
  echo "### margin $m px  $(date +%H:%M:%S)"
  python external_vindr.py --vindr-dir data/vindr --backbone raddino \
      --bs-ckpt checkpoints/bone_suppression_unet.pt \
      --head checkpoints/raddino_head.pt --margin $m --tag vindr_m$m 2>&1 | grep -E "CPM|Sensitivity @   1"
done
date +%H:%M:%S > runs/MARGIN_DONE
echo "### HET"
