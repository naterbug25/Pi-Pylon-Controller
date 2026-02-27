#!/bin/bash

# --- Pi-Pylon-Controller Environment Setup ---
# Designed for Raspberry Pi 4/5 running Raspberry Pi OS (64-bit)

echo "Starting environment setup for Pi-Pylon-Controller..."

# 1. Update System Packages
sudo apt update
sudo apt upgrade -y

# 2. Install System Dependencies for OpenCV, PyQt6, and Basler
# These are required for the GUI and Camera interface to function on Linux
sudo apt install -y \
    python3-venv \
    python3-pip \
    libqt6gui6 \
    libqt6widgets6 \
    libqt6core6 \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libatlas-base-dev \
    libusb-1.0-0-dev

# 3. Create Virtual Environment
echo "Creating virtual environment..."
python3 -m venv .venv

# 4. Activate Environment
source .venv/bin/activate

# 5. Upgrade Build Tools
pip install --upgrade pip setuptools wheel

# 6. Install Python Dependencies
# NOTE: NumPy is pinned below 2.0.0 to prevent TensorFlow ABI crashes.
echo "Installing Python packages..."
pip install "numpy<2.0.0" \
    tensorflow \
    opencv-python \
    PyQt6 \
    pycomm3 \
    pypylon \
    flask \
    python-periphery

echo "------------------------------------------------"
echo "Setup Complete."
echo "To activate the environment in the future, run: source .venv/bin/activate"
echo "------------------------------------------------"