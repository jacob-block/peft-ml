#!/bin/bash

PEFT_METHOD="last-layer"
RT_METHOD="${PEFT_METHOD}-ml"

python -u run_cifar.py \
    --retrain-method $RT_METHOD \
    --peft-methods $PEFT_METHOD \
    --num-tasks 10 \
    --lora-dim 1 \
    --batchsz 256 \
    --lr 1e-4 \
    --lr-ft 5e-4 \
    --weight-decay 1e-5 \
    --num-epochs 100 \
    --num-epochs-ft 100 \
    --seed-start 1 \
    --seed-end 5 \
    --save "results/${RT_METHOD}"
