#!/bin/bash
# Inference script for gender classifier

python scripts/inference.py \
    --audio_path data/VoxCeleb_gender/males/1861.wav \
    --model_path exp/models/best_model.pt
