# CNN_Cls_miniNAS

A lightweight Neural Architecture Search (NAS) framework for Convolutional Neural Network (CNN) classification tasks. This project enables automated search and evaluation of CNN architectures on datasets like MNIST, with a focus on modularity and extensibility.

## Features
- Modular NAS pipeline with configurable search and validation strategies
- Easily extensible search space for CNN architectures
- YAML-based configuration for experiments
- Utilities for data handling and experiment management
- Output tracking for model accuracy and loss histories

## Project Structure
```
nas.py                  # Main entry point for running NAS experiments
requirements.txt        # Python dependencies
configs/                # YAML configuration files for experiments
  config1.yaml
  config2.yaml
modules/                # Core NAS modules
  search_space.py       # Defines the CNN search space
  search_strategy.py    # Search algorithms (e.g., random, evolutionary)
  validation_strategy.py# Validation and evaluation logic
outputs/                # Experiment results
  accuracies.txt        # Accuracy logs
  winning_model.py      # Best model found
  loss_histories/       # Loss history per run
utils/                  # Utility functions
  io_ops.py             # I/O operations
  ...
data/                   # Dataset storage (e.g., MNIST)
  MNIST/
    raw/
      train-images-idx3-ubyte
      train-labels-idx1-ubyte
      t10k-images-idx3-ubyte
      t10k-labels-idx1-ubyte
tests/                  # Unit tests and reports
  test_search_space.py
  test_report.txt
```

## Getting Started

### Prerequisites
- Python 3.8+
- See `requirements.txt` for required packages

### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/Convexers-HyperOptCNN/CNN_Cls_miniNAS.git
   cd CNN_Cls_miniNAS
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Usage
1. Prepare your dataset (default: MNIST in `data/MNIST/raw/`).
2. Configure your experiment in `configs/config1.yaml` or `configs/config2.yaml`.
3. Run the NAS pipeline:
   ```bash
   python nas.py --config configs/config1.yaml
   ```
4. Results will be saved in the `outputs/` directory.

### Example Command
```bash
python nas.py --config configs/config1.yaml
```

## Testing
Run unit tests with:
```bash
python -m unittest discover tests
```
