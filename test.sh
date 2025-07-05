#!/bin/bash

python scripts/test_classifier.py \
    --test-labels data/test_labels.txt \
    --model-path exp/models/best_model.pt \
    --sample-rate 16000 \
    --batch-size 32 \
    --threshold 0.5

echo "Testing completed! Check test_results/ directory for results."