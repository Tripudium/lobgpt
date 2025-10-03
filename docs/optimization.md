# Hyperparameter Search Setup Guide

This guide explains how to set up and run hyperparameter optimization for the LOB Transformer.

## Setup

1. **Install additional dependencies:**
```bash
uv add hydra-core optuna wandb plotly
```

2. **Set up Weights & Biases (optional but recommended):**
```bash
wandb login
```

## Configuration System

The project uses **Hydra** for configuration management, enabling:
- Modular configuration with composition
- Easy hyperparameter sweeps
- Automatic experiment tracking

### Configuration Structure

```
configs/
├── config.yaml              # Main configuration
├── model/
│   ├── lob_transformer_small.yaml
│   └── lob_transformer_base.yaml
├── data/
│   └── btcusdt.yaml
├── training/
│   └── default.yaml
└── sweep/
    ├── optuna_sweep.yaml     # Optuna optimization config
    └── grid_search.yaml      # Grid search config
```

## Running Experiments

### 1. Single Training Run

```bash
cd scripts
python train_with_config.py
```

### 2. Grid Search (Hydra Multirun)

```bash
# Run predefined grid search
python run_grid_search.py

# Or run custom grid search
python train_with_config.py --config-path ../configs/sweep --config-name grid_search --multirun
```

### 3. Bayesian Optimization (Optuna)

```bash
# Run hyperparameter optimization
python hyperparameter_search.py

# Run with custom config
python hyperparameter_search.py --config-path ../configs/sweep --config-name optuna_sweep
```

## Customizing Search Spaces

### Grid Search

Edit `configs/sweep/grid_search.yaml`:

```yaml
# Grid search parameters using Hydra syntax
training.learning_rate: 1e-4,3e-4,6e-4,1e-3
data.batch_size: 16,32,64
model.d_model: 256,512
model.n_layers: 4,6
model.dropout: 0.1,0.2
```

### Optuna Search

Edit `configs/sweep/optuna_sweep.yaml`:

```yaml
search_space:
  # Model architecture
  model.d_model:
    type: categorical
    choices: [256, 512, 768]

  model.n_layers:
    type: categorical
    choices: [4, 6, 8, 12]

  # Training hyperparameters
  training.learning_rate:
    type: loguniform
    low: 1e-5
    high: 1e-3

  training.weight_decay:
    type: loguniform
    low: 1e-4
    high: 1e-1
```

### Supported Parameter Types

- **categorical**: Choose from discrete options
- **uniform**: Continuous range [low, high]
- **loguniform**: Log-scale continuous range
- **int**: Integer range with optional step
- **discrete_uniform**: Discrete values with step size

## Analyzing Results

### 1. Optuna Results

```bash
# Analyze Optuna study
python analyze_results.py --optuna-db optuna_study.db --output-dir ./analysis

# View Optuna dashboard
optuna-dashboard optuna_study.db
```

### 2. Grid Search Results

```bash
# Analyze Hydra multirun results
python analyze_results.py --results-dir ./outputs/grid_search --output-dir ./analysis
```

### 3. Generated Analysis

The analysis script creates:
- Parameter correlation plots
- Performance distributions
- Summary report with best configurations
- Detailed CSV with all results

## Advanced Configuration

### Custom Model Architectures

Create new model configs:

```yaml
# configs/model/lob_transformer_custom.yaml
_target_: lobgpt.models.lob_transformer.LOBTransformerConfig

d_model: 1024
n_layers: 8
n_heads: 16
d_ff: 4096
dropout: 0.15
use_time_encoding: true
regime_conditioning: true
```

### Data Configuration

Modify data loading:

```yaml
# configs/data/custom_data.yaml
product: "ETHUSDT"
depth: 20
train_times: ['250912.000000', '250912.060000']
val_times: ['250912.060000', '250912.090000']

sequence_length: 512
horizon: 20
threshold: 0.001
```

### Training Configuration

Adjust training parameters:

```yaml
# configs/training/aggressive.yaml
max_epochs: 100
learning_rate: 1e-3
weight_decay: 0.05
warmup_steps: 5000
scheduler_type: "onecycle"
gradient_clip_val: 2.0
```

## Example Workflows

### 1. Quick Grid Search

```bash
# Small grid search for fast iteration
python train_with_config.py \
    training.learning_rate=1e-4,3e-4 \
    data.batch_size=16,32 \
    model.d_model=256,512 \
    training.max_epochs=10 \
    --multirun
```

### 2. Production Hyperparameter Search

```bash
# Comprehensive Optuna search
python hyperparameter_search.py \
    optuna.n_trials=100 \
    training.max_epochs=50 \
    wandb.enabled=true
```

### 3. Architecture Search

```bash
# Focus on model architecture
python train_with_config.py \
    model.d_model=256,512,768,1024 \
    model.n_layers=4,6,8,12,16 \
    model.n_heads=4,8,12,16 \
    training.max_epochs=20 \
    --multirun
```

## Monitoring and Debugging

### 1. Weights & Biases Integration

- Automatic experiment tracking
- Real-time metrics visualization
- Hyperparameter importance analysis
- Model artifact storage

### 2. Local Monitoring

```bash
# TensorBoard (if W&B disabled)
tensorboard --logdir ./outputs

# Check experiment status
ls ./outputs/grid_search/
```

### 3. Resource Management

```yaml
# Limit resource usage
training:
  max_epochs: 20  # Shorter for search
data:
  batch_size: 16  # Smaller batches
  num_workers: 2  # Fewer workers
```

## Tips and Best Practices

### 1. Search Strategy

1. **Start small**: Begin with a coarse grid search
2. **Iterative refinement**: Use best results to guide next search
3. **Focus on important parameters**: Learning rate, model size, batch size
4. **Consider resource constraints**: Balance thoroughness with time/compute

### 2. Configuration Management

- Use descriptive experiment names
- Version control your configs
- Document important findings
- Save best configurations for production

### 3. Performance Optimization

- Use mixed precision training (`precision: "16-mixed"`)
- Enable gradient checkpointing for large models
- Tune batch size for your hardware
- Consider distributed training for large searches

### 4. Early Stopping

```yaml
# Aggressive early stopping for search
early_stopping:
  monitor: "val_loss"
  patience: 5  # Shorter patience
  mode: "min"

# Optuna pruning
optuna:
  pruner:
    _target_: optuna.pruners.MedianPruner
    n_startup_trials: 3
    n_warmup_steps: 20
```

## Troubleshooting

### Common Issues

1. **Out of memory**: Reduce batch size or model size
2. **Slow convergence**: Increase learning rate or reduce model complexity
3. **NaN losses**: Check learning rate (too high) or gradient clipping
4. **Poor performance**: Verify data preprocessing and tokenization

### Debug Commands

```bash
# Test single configuration
python train_with_config.py training.max_epochs=1 data.batch_size=4

# Check configuration resolution
python train_with_config.py --cfg job

# Validate search space
python hyperparameter_search.py optuna.n_trials=1
```