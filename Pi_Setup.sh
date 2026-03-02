#!/bin/bash
# Pi-Pylon-Controller Modern Stack Setup (TF 2.16+ / NumPy 2.x)

echo "--- Removing old environment ---"
rm -rf .venv

echo "--- Installing System Dependencies ---"
sudo apt-get update
sudo apt-get install -y python3-pyqt6 libatlas-base-dev libqt5gui5 libqt5core5a libopenjp2-7

echo "--- Creating 64-bit .venv ---"
python3 -m venv .venv --system-site-packages
source .venv/bin/activate

echo "--- Installing Latest AI Stack ---"
# Installing these together ensures they are compiled against the same NumPy version
# 1. Upgrade pip
pip install --upgrade pip

# 2. Install UI and Vision Core
pip install PyQt6 "opencv-contrib-python-headless>=4.10"

# 3. Install Hardware and Communication protocols
pip install pycomm3 flask python-periphery pypylon

# 4. Install YOLO and force the NumPy downgrade simultaneously 
pip install ultralytics "numpy<2"

echo "--- Verification ---"
python3 -c "import tensorflow as tf; import numpy as np; print(f'TF: {tf.__version__} | NumPy: {np.__version__}')"