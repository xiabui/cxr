#!/bin/bash
# Doi head duoc xuat xong roi chay kiem dinh ngoai tren VinDr.
set -u
source ~/miniconda3/etc/profile.d/conda.sh
conda activate ai_env
cd ~/workspace/cancer-detection/cxr_nodule
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# Khong dem stdout. Neu khong, tien do fold nam trong bo dem va log rong
# suot nhieu gio — khong biet chay toi dau, tuong nhu treo.
export PYTHONUNBUFFERED=1
rm -f runs/VINDR_DONE

# cho file head xuat hien va on dinh kich thuoc
while [ ! -f checkpoints/raddino_head.pt ]; do sleep 20; done
s1=0; while :; do s2=$(stat -c%s checkpoints/raddino_head.pt); [ "$s1" = "$s2" ] && break; s1=$s2; sleep 10; done
echo "### head san sang $(date +%H:%M:%S)"

echo "### KIEM DINH NGOAI TREN VinDr  $(date +%H:%M:%S)"
time python external_vindr.py --vindr-dir data/vindr --backbone raddino --bs-ckpt checkpoints/bone_suppression_unet.pt --head checkpoints/raddino_head.pt

echo "### KHOANG TIN CAY"
python bootstrap_ci.py runs/scores_vindr.json --B 2000

date +%H:%M:%S > runs/VINDR_DONE
echo "### HET"
