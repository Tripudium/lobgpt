"""
Hyperparameter optimization using Optuna.
"""

import os
import sys
from pathlib import Path
from typing import Dict, Any

import hydra
import optuna
import pytorch_lightning as pl
from omegaconf import DictConfig, OmegaConf

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from scripts.train_with_config import train_model


def create_trial_config(cfg: DictConfig, trial: optuna.Trial) -> DictConfig:
    """Create configuration for current trial."""
    trial_cfg = cfg.copy()

    # Sample hyperparameters based on search space
    for param_name, param_config in cfg.search_space.items():
        param_type = param_config.type

        if param_type == "categorical":
            value = trial.suggest_categorical(param_name, param_config.choices)
        elif param_type == "uniform":
            value = trial.suggest_uniform(param_name, param_config.low, param_config.high)
        elif param_type == "loguniform":
            value = trial.suggest_loguniform(param_name, param_config.low, param_config.high)
        elif param_type == "int":
            step = param_config.get("step", 1)
            value = trial.suggest_int(param_name, param_config.low, param_config.high, step=step)
        elif param_type == "discrete_uniform":
            value = trial.suggest_discrete_uniform(
                param_name, param_config.low, param_config.high, param_config.q
            )
        else:
            raise ValueError(f"Unknown parameter type: {param_type}")

        # Set the value in the config using dot notation
        keys = param_name.split(".")
        config_section = trial_cfg
        for key in keys[:-1]:
            config_section = config_section[key]
        config_section[keys[-1]] = value

    # Handle dependent parameters
    if trial_cfg.model.d_ff is None:
        trial_cfg.model.d_ff = 4 * trial_cfg.model.d_model

    # Ensure n_heads divides d_model
    while trial_cfg.model.d_model % trial_cfg.model.n_heads != 0:
        trial_cfg.model.n_heads = trial.suggest_categorical(
            f"model.n_heads_adjusted_{trial.number}",
            [h for h in cfg.search_space["model.n_heads"].choices
             if trial_cfg.model.d_model % h == 0]
        )

    # Update experiment name for this trial
    trial_cfg.experiment_name = f"{cfg.experiment_name}_trial_{trial.number}"

    return trial_cfg


def objective(trial: optuna.Trial, base_cfg: DictConfig) -> float:
    """Optuna objective function."""
    # Create trial-specific configuration
    trial_cfg = create_trial_config(base_cfg, trial)

    # Set up trial-specific paths
    trial_cfg.output_dir = f"{base_cfg.output_dir}/trial_{trial.number}"
    trial_cfg.checkpoint_dir = f"{trial_cfg.output_dir}/checkpoints"

    # Create directories
    Path(trial_cfg.output_dir).mkdir(parents=True, exist_ok=True)
    Path(trial_cfg.checkpoint_dir).mkdir(parents=True, exist_ok=True)

    # Log trial parameters
    print(f"\n{'='*60}")
    print(f"TRIAL {trial.number}")
    print(f"{'='*60}")
    print("Parameters:")
    for param_name, param_config in base_cfg.search_space.items():
        keys = param_name.split(".")
        value = trial_cfg
        for key in keys:
            value = value[key]
        print(f"  {param_name}: {value}")

    try:
        # Train model
        results = train_model(trial_cfg)
        val_loss = results["val_loss"]

        # Report intermediate values for pruning
        trial.report(val_loss, step=base_cfg.training.max_epochs)

        # Handle pruning
        if trial.should_prune():
            raise optuna.TrialPruned()

        print(f"Trial {trial.number} completed. Val loss: {val_loss:.4f}")
        return val_loss

    except Exception as e:
        print(f"Trial {trial.number} failed: {e}")
        return float('inf')


def run_study(cfg: DictConfig):
    """Run Optuna study."""
    # Create study
    storage = cfg.optuna.storage if cfg.optuna.storage else None

    study = optuna.create_study(
        study_name=cfg.optuna.study_name,
        direction=cfg.optuna.direction,
        storage=storage,
        load_if_exists=True,
        pruner=hydra.utils.instantiate(cfg.optuna.pruner),
        sampler=hydra.utils.instantiate(cfg.optuna.sampler)
    )

    # Define objective with config
    objective_with_config = lambda trial: objective(trial, cfg)

    # Optimize
    study.optimize(
        objective_with_config,
        n_trials=cfg.optuna.n_trials,
        timeout=cfg.optuna.timeout,
        n_jobs=1  # Sequential execution for GPU memory management
    )

    # Print results
    print("\n" + "="*60)
    print("OPTIMIZATION COMPLETED")
    print("="*60)

    print(f"Number of finished trials: {len(study.trials)}")
    print(f"Best trial: {study.best_trial.number}")
    print(f"Best value: {study.best_value:.4f}")

    print("\nBest parameters:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")

    # Save results
    results_path = Path(cfg.output_dir) / "optimization_results.txt"
    with open(results_path, "w") as f:
        f.write(f"Best trial: {study.best_trial.number}\n")
        f.write(f"Best value: {study.best_value:.4f}\n")
        f.write("\nBest parameters:\n")
        for key, value in study.best_params.items():
            f.write(f"  {key}: {value}\n")

    print(f"\nResults saved to: {results_path}")

    # Optionally create visualization
    try:
        import plotly
        fig = optuna.visualization.plot_optimization_history(study)
        fig.write_html(Path(cfg.output_dir) / "optimization_history.html")

        fig = optuna.visualization.plot_param_importances(study)
        fig.write_html(Path(cfg.output_dir) / "param_importances.html")

        print("Visualizations saved to optimization_history.html and param_importances.html")
    except ImportError:
        print("Plotly not available. Skipping visualizations.")

    return study


@hydra.main(version_base=None, config_path="../configs/sweep", config_name="optuna_sweep")
def main(cfg: DictConfig):
    """Main hyperparameter search function."""
    print("="*60)
    print("LOB Transformer Hyperparameter Optimization")
    print("="*60)

    # Create output directory
    Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)

    # Run optimization
    study = run_study(cfg)

    return study.best_value


if __name__ == "__main__":
    main()