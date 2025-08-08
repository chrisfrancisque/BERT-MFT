#!/bin/bash
# Run script with TPU optimizations and fallbacks

set -e

echo "Starting TPU Gradient Analysis Experiment"
echo "=========================================="

# Activate environment
source $HOME/tpu-env/bin/activate

# Clean up any stuck processes
echo "Cleaning up stuck processes..."
sudo pkill -f libtpu || true
sleep 2

# Try PJRT first (recommended)
echo "Attempting to run with PJRT runtime..."
export PJRT_DEVICE=TPU
export XLA_USE_BF16=1
export TORCH_COMPILE_DISABLE=1

# Run with timeout to catch hangs
timeout 30 python main_tpu.py --test 2>&1 | tee test_pjrt.log

if [ $? -ne 0 ]; then
    echo "PJRT runtime failed or timed out. Switching to XRT..."
    
    # Kill any stuck processes
    sudo pkill -f python || true
    sudo pkill -f libtpu || true
    sleep 3
    
    # Switch to XRT runtime
    unset PJRT_DEVICE
    export TPU_NUM_DEVICES=8
    export XRT_TPU_CONFIG="localservice;0;localhost:51011"
    export XLA_USE_BF16=1
    
    echo "Retrying with XRT runtime..."
    python main_tpu.py --test 2>&1 | tee test_xrt.log
    
    if [ $? -eq 0 ]; then
        echo "XRT runtime successful! Using XRT for full run."
        RUNTIME="XRT"
    else
        echo "Both runtimes failed. Check logs for details."
        exit 1
    fi
else
    echo "PJRT runtime successful!"
    RUNTIME="PJRT"
fi

# Ask if user wants to run full experiment
echo ""
echo "Test run complete with $RUNTIME runtime."
read -p "Run full experiment (1000 samples)? (y/n) " -n 1 -r
echo ""

if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "Starting full experiment..."
    
    if [ "$RUNTIME" = "XRT" ]; then
        # Use XRT settings
        unset PJRT_DEVICE
        export TPU_NUM_DEVICES=8
        export XRT_TPU_CONFIG="localservice;0;localhost:51011"
        export XLA_USE_BF16=1
    else
        # Use PJRT settings
        export PJRT_DEVICE=TPU
        export XLA_USE_BF16=1
    fi
    
    # Run full experiment
    python main_tpu.py 2>&1 | tee full_experiment.log
    
    echo "Experiment complete! Check results/ directory for outputs."
else
    echo "Skipping full experiment."
fi