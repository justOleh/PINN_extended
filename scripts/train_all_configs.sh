#!/bin/bash

MAIN_SCRIPT="train.py"
CONFIG_DIR="configs/training"

for CONFIG in "$CONFIG_DIR"/*.yaml; do
    echo "Running $MAIN_SCRIPT with config: $CONFIG"
    python "$MAIN_SCRIPT" --config_path "$CONFIG"
done
