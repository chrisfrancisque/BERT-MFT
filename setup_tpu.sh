#!/bin/bash
# TPU Setup Script for Gradient Analysis Experiment

set -e  # Exit on error

echo "=================================="
echo "TPU Environment Setup"
echo "=================================="

# Check if we're on a TPU VM
if [ ! -d "/dev/accel0" ] && [ ! -c "/dev/accel0" ]; then
    echo "Warning: TPU device not detected. Are you on a TPU VM?"
fi

# Kill any stuck TPU processes
echo "Cleaning up any stuck TPU processes..."
sudo pkill -f libtpu || true
sudo pkill -f torch_xla || true

# Setup Python environment
echo "Setting up Python 3.10 environment..."
if [ ! -d "$HOME/tpu-env" ]; then
    python3.10 -m venv $HOME/tpu-env
    echo "Created new virtual environment at $HOME/tpu-env"
else
    echo "Using existing environment at $HOME/tpu-env"
fi

# Activate environment
source $HOME/tpu-env/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install PyTorch and XLA
echo "Installing PyTorch 2.7.0 and XLA..."
pip install torch==2.7.0

# Install torch_xla with TPU support
pip install 'torch_xla[tpu]==2.7.0' \
    -f https://storage.googleapis.com/libtpu-wheels/index.html \
    -f https://storage.googleapis.com/libtpu-releases/index.html

# Install other dependencies
echo "Installing other dependencies..."
pip install transformers==4.36.0
pip install datasets==2.16.1
pip install numpy==1.24.3
pip install scikit-learn==1.3.0
pip install matplotlib==3.5.0
pip install seaborn==0.12.0
pip install tqdm
pip install pyarrow==11.0.0
pip install fsspec==2023.5.0
pip install pandas==2.0.3

# Verify TPU access
echo "Verifying TPU access..."
python3 -c "
import torch_xla.core.xla_model as xm
devices = xm.xla_real_devices()
print(f'TPU devices available: {devices}')
print(f'Number of TPU cores: {len(devices)}')
"

echo "=================================="
echo "Setup complete!"
echo "=================================="
echo ""
echo "To run the experiment:"
echo "  1. Test mode (100 samples): python main_tpu.py --test"
echo "  2. Full mode (1000 samples): python main_tpu.py"
echo ""
echo "Environment variables will be set automatically by the script."