#!/bin/bash
# Data preparation script for gender classifier

set -e

echo "Starting data preparation..."

python scripts/data_preparation.py

echo "Data preparation completed!"