#!/bin/bash

MAIN_SCRIPT="eval.py"
CONFIG_DIR="weights/trained_models"

for CONFIG in "$CONFIG_DIR"/*.pth; do
    echo "Running $MAIN_SCRIPT with config: $CONFIG"
    python "$MAIN_SCRIPT" --weights_path "$CONFIG"
done
