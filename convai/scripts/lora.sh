#!/bin/bash

YOUR_CACHE_DIR="" # set this

RANK=8
METHOD="standard"
SEEDS=(0 1 2 3 4)
for SEED in ${SEEDS[@]}; do
python -u lora.py \
    --model roberta-base \
    --lora_dim $RANK \
    --from_pretrained_base "results/${METHOD}/seed${SEED}" \
    --save "results/${METHOD}/lora-ft/rank${RANK}/seed${SEED}" \
    --persona_nums_rng 0 9 \
    --num_valid 2 \
    --batchsz 8 \
    --num_epochs 10 \
    --lr 1e-4 \
    --sm 4.0 \
    --valid_file data/raw_download/valid_self_original.txt \
    --cache_dir $YOUR_CACHE_DIR \
    --persona_map_path data/ \
    --seed $SEED \
    --device cuda
done

echo ""
echo "################################################################"
echo "@@@@@@@@@@@@@@@@@@@@ Run completed at:- @@@@@@@@@@@@@@@@@@@@@@@@@"
date
echo "@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@"
echo "################################################################"
