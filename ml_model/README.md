# Peregrine ML Model

This folder contains the ML Model and associated scripts for the Peregrine Neural stage. The model is based in PyTorch, and uses the AWS Neuron SDK to run on Trainium hardware.

### Contents

- **`model.py`**: Defines the `PeregrineMLModel` feed‑forward neural network (simple MLP with configurable hidden sizes).
- **`plot_train_test_loss.py`**: Plots the loss and error curves from the training metrics JSON files.
- **`train.py`**: Trains `PeregrineMLModel` on the provided dataset using PyTorch with AWS Neuron.
- **`training_metrics/`** (created at runtime): Stores JSON files containing the train/eval loss and percent error at each epoch saved by `train.py`.
- **`checkpoints/`** (created at runtime): Stores model checkpoints (e.g. `checkpoint.pt`) saved by `train.py`.
- **`plots/`** (created at runtime): Stores png plots of epoch vs loss and epoch vs error, created by `plot_train_test_loss.py`.
- **`training_data/`**: Place csv files containing the analytical model outputs for each cpu configuration. NOTE: suggested location, not required.

### Requirements

- Trainium EC2 instance with the Neuron Deep Learning AMI. 

## Usage

### 0. Source the Neuron Environment
```bash
source /opt/aws_neuronx_venv_pytorch_2_8/bin/activate
```
Note that the exact path may be different depending on the AMI version you are using.

### 1. Train the model

```bash
python train.py --dataset-path <csv path>
```

Training logs and loss metrics will be printed to stdout, and a checkpoint will be written to `checkpoints/` upon completion.
Per epoch loss and error values will be stored in `training_metrics/`.

### 2. Plot the training curves

```bash
python plot_train_test_loss.py --training-metrics <json path>
```

Plots epoch vs train/eval loss curve and epoch vs error curve to `plots/`.

## Notes 

### Clearing Neuron Compilation Cache
You may notice Neuron uses cached neffs to avoid recompilation:
```bash
INFO ||NEURON_CC_WRAPPER||: Using a cached neff at /var/tmp/neuron-compile-cache/neuronxcc-2.21.33363.0+82129205/MODULE_16389337710168549518+e30acd
3a/model.neff
```

In case you make changes to the files and Neuron uses the cached neffs instead of recompiling, you can clear the cached files with the following command:
```bash
rm -rf /var/tmp/neuron-compile-cache/
```
