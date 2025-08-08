# BERT-MFT: Mask Fine-Tuning for BERT

## Overview

BERT-MFT (Mask Fine-Tuning) is a novel approach to analyzing and modifying BERT models by identifying and selectively zeroing parameters that naturally trend toward zero during training. This technique preemptively removes parameters that would decay during fine-tuning, potentially improving model efficiency without traditional training.

## Methodology

### Core Concept

Instead of performing full fine-tuning, this approach:

1. Collects gradients from one epoch of training data without updating weights
2. Identifies "detrimental parameters" that would move toward zero
3. Zeros the top 10% of these parameters based on movement magnitude
4. Evaluates the modified model's performance

### Detrimental Parameter Definition

A parameter is considered "detrimental" if applying the gradient update would reduce its absolute value:

* Parameter at 0.6 that would update to 0.1: detrimental (moving toward 0)
* Parameter at -0.8 that would update to -0.1: detrimental (moving toward 0)  
* Parameter at 0.2 that would update to 0.5: NOT detrimental (moving away from 0)

## Current Results

### Local Experiments (MacBook Pro M2)

Experiments conducted on SST-2 sentiment analysis dataset with 1000 training samples:

**Configuration:**
* Model: bert-base-uncased (109.5M parameters)
* Training samples: 1000
* Batch size: 32
* Learning rate: 2e-5 (for gradient calculation only)
* Device: Apple Silicon GPU (MPS)

**Key Findings:**
* Total detrimental parameters identified: 39,991,457 (36.53% of model)
* Parameters zeroed: 3,999,146 (3.65% of total model)
* Affected layers: 187 out of 201 (classifier layers excluded)
* Gradient collection time: ~5.5 minutes
* Parameter zeroing time: ~12 minutes

**Performance Impact:**

The baseline model exhibited strong bias (100% recall, predicting all positive), while the modified model flipped to the opposite bias (0% recall, predicting all negative). This dramatic shift demonstrates that the approach has significant impact on model behavior, though further refinement is needed for balanced performance.

## Project Structure

```
BERT-MFT/
├── config.py                 # Experiment configuration
├── data_utils.py            # Dataset loading and preprocessing
├── gradient_analyzer.py     # Gradient collection and analysis
├── detrimental_params.py    # Parameter identification and zeroing
├── evaluation.py            # Model evaluation utilities
├── main_local.py            # Local execution script
├── main_tpu.py              # TPU-optimized execution
├── setup_tpu.sh             # TPU environment setup
├── run_tpu.sh               # TPU execution with fallbacks
├── requirements.txt         # Python dependencies
└── README.md                # This file
```

## Installation

### Prerequisites

* Python 3.10 or 3.11 (not compatible with Python 3.13)
* 8GB+ RAM for gradient storage
* CUDA GPU (optional) or Apple Silicon

### Setup

1. Clone the repository:
```bash
git clone https://github.com/chrisfrancisque/BERT-MFT.git
cd BERT-MFT
```

2. Create virtual environment:
```bash
python3.10 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

## Usage

### Local Execution

Test run with 100 samples:
```bash
python main_local.py --test
```

Full experiment with 1000 samples:
```bash
python main_local.py
```

### TPU Execution

Setup TPU environment:
```bash
chmod +x setup_tpu.sh
./setup_tpu.sh
```

Run experiment:
```bash
./run_tpu.sh
```

## Technical Implementation

### Gradient Collection

The system collects actual gradient tensors matching the exact shape of model parameters, accumulating them across all training batches without performing weight updates. This provides a complete gradient snapshot of the model's training dynamics.

### Memory Efficient Processing

To handle 39M+ detrimental parameters, the implementation uses a heap-based approach that maintains only the top 10% of parameters in memory, preventing memory overflow issues encountered in initial versions.

### Dataset Loading Fixes

The implementation includes robust handling of HuggingFace dataset cache issues:
* Forced fresh downloads to prevent cache corruption
* Fallback to alternative dataset namespaces
* Proper column renaming for BERT compatibility

## Key Insights

1. **Parameter Distribution**: Approximately 35% of BERT parameters show detrimental behavior (trending toward zero) after just one epoch of gradient accumulation.

2. **Layer Impact**: The classifier head shows the strongest detrimental signals, necessitating its exclusion to prevent model collapse.

3. **Sensitivity**: Zeroing just 3.65% of parameters can completely flip model behavior, indicating high sensitivity to parameter modification.

## Future Work

* Implement adaptive thresholding instead of fixed percentile selection
* Explore layer-wise zeroing strategies
* Test on additional datasets and tasks
* Investigate relationship between training epochs and detrimental parameter identification
* Optimize zeroing process for faster execution

## Output Structure

Each experiment creates a timestamped directory containing:
```
results/experiment_YYYYMMDD_HHMMSS/
├── logs/                           # Detailed execution logs
├── training_loss.png              # Loss curve visualization
├── results.json                   # Complete metrics
├── summary.txt                    # Human-readable summary
└── gradient_statistics.json      # Per-parameter gradient statistics
```

## Citation

If you use this work in your research, please cite:
```
@software{bert-mft,
  title = {BERT-MFT: Mask Fine-Tuning for BERT},
  author = {Chris Francisque},
  year = {2025},
  url = {https://github.com/chrisfrancisque/BERT-MFT}
}
```
