import os
import sys
os.environ["PJRT_DEVICE"] = "TPU"
os.environ["XLA_USE_BF16"] = "1"
os.environ["TORCH_COMPILE_DISABLE"] = "1"

# Configure for small test
from config import config
config.train_samples = 100
config.batch_size = 8

# Use the 75% checkpoint
checkpoint_path = 'models/fft_checkpoints/checkpoint_75pct'

# Check it exists
if not os.path.exists(checkpoint_path):
    print(f"ERROR: Checkpoint not found at {checkpoint_path}")
    sys.exit(1)

print(f"Testing with 75% accuracy checkpoint")

# Update config to use this model
config.model_name = checkpoint_path

# Run the single-core test
from main_tpu_fixed import run_on_tpu_single_core
run_on_tpu_single_core()
