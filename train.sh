#!/bin/bash

set -e

echo "Starting training with Wav2Vec encoder..."

python scripts/train_classifier.py \
    --train-labels data/train_labels.txt \
    --val-labels data/val_labels.txt \
    --batch-size 32 \
    --epochs 100 \
    --learning-rate 0.001 \
    --save-dir exp/models \
    --tensorboard-dir exp/runs \
    --sample-rate 16000 \
    --noise-factor 0.15 \
    --background-noise-dir data/musan \
    --augmentation-prob 0.6 \
    --num-workers 4 \
    --checkpoint exp/models/checkpoint_epoch_2.pt

echo "Training completed!"