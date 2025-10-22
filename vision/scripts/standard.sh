#!/bin/bash

RT_METHOD="standard"
PEFT_METHODS="lora last-layer"

python -u run_cifar.py \
    --retrain-method $RT_METHOD \
    --peft-methods $PEFT_METHODS \
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
    --save "paper-results/${RT_METHOD}/"
