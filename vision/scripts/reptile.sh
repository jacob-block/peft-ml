#!/bin/bash

RT_METHOD="reptile"
PEFT_METHODS="lora last-layer"

python -u run_cifar.py \
    --retrain-method $RT_METHOD \
    --peft-methods $PEFT_METHODS \
    --num-tasks 10 \
    --lora-dim 1 \
    --batchsz 256 \
    --lr 1e-5 \
    --lr-ft 5e-4 \
    --inner-lr 1e-2 \
    --weight-decay 1e-5 \
    --num-epochs 5 \
    --num-epochs-inner 20 \
    --num-epochs-ft 100 \
    --seed-start 1 \
    --seed-end 5 \
    --save "paper-results/${RT_METHOD}/"
