#!/bin/bash

YOUR_CACHE_DIR="" # set this

SEEDS=(0 1 2 3 4)
for SEED in ${SEEDS[@]}; do
    python -u standard.py \
        --model roberta-base \
        --persona_nums_rng 0 9 \
        --num_valid 3 \
        --batchsz 8 \
        --num_epochs 10 \
        --lr 5e-5 \
        --sm 6.67 \
        --train_file data/raw_download/train_self_original.txt \
        --cache_dir $YOUR_CACHE_DIR \
        --persona_map_path data/ \
        --seed $SEED \
        --save results/standard/seed$SEED \
        --device cuda
done
