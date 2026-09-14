#!/bin/bash
# Tinh chinh mot phan RAD-DINO tren homeserver (thay cho Colab).
#
# HAI NHANH CHI KHAC NHAU O --variants. Moi thu khac giong het:
#   --unfreeze 4 --epochs 20 --folds 5 --batch 2 --seed 42
# Doi them bat cu tham so nao la doi hai yeu to cung luc, va chenh lech se
# khong quy ket duoc cho tang cuong du lieu (CLAUDE.md muc 5).
#
# KHONG dung pgrep de cho trong file nay: mau pgrep tung khop chinh dong lenh
# cha chua no, lam vong doi khong bao gio thoat.
set -u
source ~/miniconda3/etc/profile.d/conda.sh
conda activate ai_env
cd ~/workspace/cancer-detection/cxr_nodule
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
ulimit -n 65535
rm -f runs/FT_DONE

CFG=configs/jsrt_ft_homeserver.yaml
COMMON="--config $CFG --unfreeze 4 --epochs 20 --folds 5 --batch 2 --seed 42"

echo "### NHANH A: tinh chinh, KHONG tang cuong   $(date +%H:%M:%S)"
time python raddino_finetune.py $COMMON --variants 1 --tag ft_noaug
echo "### A xong $(date +%H:%M:%S)"

echo "### NHANH B: tinh chinh, CO tang cuong (8 bien the)   $(date +%H:%M:%S)"
time python raddino_finetune.py $COMMON --variants 8 --tag ft_aug
echo "### B xong $(date +%H:%M:%S)"

echo "### SO SANH BAT CAP A vs B"
python bootstrap_ci.py --compare runs/scores_ft_noaug.json runs/scores_ft_aug.json --B 3000

echo "### DOI CHIEU: hai nhanh co that su chi khac mot yeu to khong?"
python - <<PYEND
import json
a=json.load(open("runs/results_ft_noaug.json")); b=json.load(open("runs/results_ft_aug.json"))
keys=["unfreeze","epochs","folds","batch","seed","protocol"]
print("  tham so   A / B")
bad=[]
for k in keys:
    same = a.get(k)==b.get(k)
    if not same: bad.append(k)
    print(f"  {k:<10} {a.get(k)} / {b.get(k)}   {\"OK\" if same else \"LECH !!\"}")
print(f"  variants   {a.get(\"variants\")} / {b.get(\"variants\")}   (day la yeu to DUY NHAT duoc phep khac)")
print("KET LUAN:", "hop le" if not bad else f"HONG - lech o {bad}")
PYEND

date +%H:%M:%S > runs/FT_DONE
echo "### HET"
