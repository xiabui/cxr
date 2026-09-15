#!/bin/bash
# ResNet-34, 10-fold, NAM hat giong, hai cau hinh chi khac DUNG MOT yeu to:
#   resnet_base : in_channels 1 (khong khu xuong)
#   resnet_bs   : in_channels 2 (co khu xuong)
# CBAM TAT o ca hai — bat CBAM la doi hai yeu to cung luc.
#
# Muc dich: tai lap hieu ung khu xuong tren KIEN TRUC THU HAI, co gieo hat va
# co khoang tin cay. Neu duong, do la bang chung cheo kien truc that su.
#
# KHONG SUA FILE NAY TRONG KHI NO DANG CHAY (bash doc theo vi tri byte).
set -u
source ~/miniconda3/etc/profile.d/conda.sh
conda activate ai_env
cd ~/workspace/cancer-detection/cxr_nodule
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1
ulimit -n 65535
rm -f runs/RESNET_DONE

for s in 0 1 2 3 4; do
  for cfg in base bs; do
    T=resnet_${cfg}_s${s}
    if [ -f runs/scores_${T}.json ]; then echo "### $T da co, bo qua"; continue; fi
    echo "### $T  $(date +%H:%M:%S)"
    python -m src.train --config configs/resnet_${cfg}_10fold.yaml --seed $s --tag $T 2>&1 | grep -E "^Seed|CPM |Sensitivity @   1"
    echo "### $T xong $(date +%H:%M:%S)"
  done
done

echo "### SO SANH BAT CAP TUNG SEED"
for s in 0 1 2 3 4; do
  echo "--- seed $s ---"
  python bootstrap_ci.py --compare runs/scores_resnet_base_s${s}.json runs/scores_resnet_bs_s${s}.json --B 2000 2>&1 | grep -E "^  CPM"
done
echo "### GOP NAM SEED"
python seed_pooled_ci.py --a "runs/scores_resnet_base_s*.json" --b "runs/scores_resnet_bs_s*.json" --B 2000

date +%H:%M:%S > runs/RESNET_DONE
echo "### HET"
