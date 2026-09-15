#!/bin/bash
# LOOCV co GIEO HAT va co LUU DIEM THO, de thay cho Bang 1 hien dang la
# ket qua chay don le khong tai lap duoc.
#
# Hai nhanh, chi khac --max-variants (tuc co/khong kenh khu xuong qua U-Net):
#   plain = 3 kenh lap      (features_raddino, raddino_train2)
#   unet  = + khu xuong     (features_aug v0, raddino_train3 --max-variants 1)
#
# 247 fold moi nhanh. Uoc tinh ~2-3 gio moi nhanh.
set -u
source ~/miniconda3/etc/profile.d/conda.sh
conda activate ai_env
cd ~/workspace/cancer-detection/cxr_nodule
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# Khong dem stdout. Neu khong, tien do fold nam trong bo dem va log rong
# suot nhieu gio — khong biet chay toi dau, tuong nhu treo.
export PYTHONUNBUFFERED=1
ulimit -n 65535
rm -f runs/LOOCV_DONE

echo "### LOOCV nhanh plain (3 kenh lap)  $(date +%H:%M:%S)"
time python raddino_train2.py --config configs/jsrt_loocv.yaml --epochs 60 --seed 42 --tag loocv_plain
echo "### plain xong $(date +%H:%M:%S)"

echo "### LOOCV nhanh unet (+ khu xuong)  $(date +%H:%M:%S)"
time python raddino_train3.py --config configs/jsrt_loocv.yaml --epochs 60 --seed 42 --max-variants 1 --tag loocv_unet
echo "### unet xong $(date +%H:%M:%S)"

echo "### KHOANG TIN CAY TUNG NHANH"
python bootstrap_ci.py runs/scores_loocv_plain.json --B 2000
python bootstrap_ci.py runs/scores_loocv_unet.json  --B 2000
echo "### SO SANH BAT CAP"
python bootstrap_ci.py --compare runs/scores_loocv_plain.json runs/scores_loocv_unet.json --B 3000

date +%H:%M:%S > runs/LOOCV_DONE
echo "### HET"
