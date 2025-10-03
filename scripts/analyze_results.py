"""
Analyze and visualize hyperparameter search results.
"""

import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import List, Dict, Any
import argparse


def load_grid_search_results(results_dir: Path) -> pd.DataFrame:
    """Load results from Hydra grid search."""
    results = []

    # Find all experiment directories
    for exp_dir in results_dir.rglob("*"):
        if exp_dir.is_dir() and (exp_dir / ".hydra").exists():
            try:
                # Load config
                config_file = exp_dir / ".hydra" / "config.yaml"
                if config_file.exists():
                    import yaml
                    with open(config_file) as f:
                        config = yaml.safe_load(f)

                # Try to find metrics file
                metrics_files = list(exp_dir.glob("**/metrics.json"))
                if not metrics_files:
                    # Look for wandb logs or other metric sources
                    continue

                with open(metrics_files[0]) as f:
                    metrics = json.load(f)

                # Combine config and metrics
                result = {
                    "experiment_dir": str(exp_dir),
                    "learning_rate": config.get("training", {}).get("learning_rate"),
                    "batch_size": config.get("data", {}).get("batch_size"),
                    "d_model": config.get("model", {}).get("d_model"),
                    "n_layers": config.get("model", {}).get("n_layers"),
                    "dropout": config.get("model", {}).get("dropout"),
                    "val_loss": metrics.get("val_loss"),
                    "val_accuracy": metrics.get("val_accuracy"),
                }
                results.append(result)

            except Exception as e:
                print(f"Error loading {exp_dir}: {e}")
                continue

    return pd.DataFrame(results)


def load_optuna_results(db_path: Path) -> pd.DataFrame:
    """Load results from Optuna study."""
    import optuna

    study = optuna.load_study(
        study_name="lob_transformer_optimization",
        storage=f"sqlite:///{db_path}"
    )

    results = []
    for trial in study.trials:
        result = trial.params.copy()
        result.update({
            "trial_number": trial.number,
            "value": trial.value,
            "state": trial.state.name,
            "duration": trial.duration.total_seconds() if trial.duration else None
        })
        results.append(result)

    return pd.DataFrame(results)


def plot_parameter_analysis(df: pd.DataFrame, target_col: str = "val_loss", save_dir: Path = None):
    """Create parameter analysis plots."""
    # Identify numeric and categorical columns
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
    numeric_cols = [col for col in numeric_cols if col != target_col and not col.startswith('trial')]

    categorical_cols = df.select_dtypes(include=['object']).columns
    categorical_cols = [col for col in categorical_cols if col not in ['experiment_dir', 'state']]

    n_numeric = len(numeric_cols)
    n_categorical = len(categorical_cols)
    n_plots = n_numeric + n_categorical

    if n_plots == 0:
        print("No parameters to plot")
        return

    # Calculate subplot layout
    n_cols = min(3, n_plots)
    n_rows = (n_plots + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
    if n_plots == 1:
        axes = [axes]
    elif n_rows == 1:
        axes = axes.reshape(1, -1)

    plot_idx = 0

    # Plot numeric parameters
    for col in numeric_cols:
        if plot_idx >= len(axes.flat):
            break

        ax = axes.flat[plot_idx]
        ax.scatter(df[col], df[target_col], alpha=0.6)
        ax.set_xlabel(col)
        ax.set_ylabel(target_col)
        ax.set_title(f'{target_col} vs {col}')

        # Add correlation
        corr = df[col].corr(df[target_col])
        ax.text(0.05, 0.95, f'r = {corr:.3f}', transform=ax.transAxes,
                bbox=dict(boxstyle="round", facecolor='wheat', alpha=0.5))

        plot_idx += 1

    # Plot categorical parameters
    for col in categorical_cols:
        if plot_idx >= len(axes.flat):
            break

        ax = axes.flat[plot_idx]
        df.boxplot(column=target_col, by=col, ax=ax)
        ax.set_xlabel(col)
        ax.set_ylabel(target_col)
        ax.set_title(f'{target_col} by {col}')
        plt.suptitle('')  # Remove automatic title

        plot_idx += 1

    # Hide unused subplots
    for i in range(plot_idx, len(axes.flat)):
        axes.flat[i].set_visible(False)

    plt.tight_layout()

    if save_dir:
        plt.savefig(save_dir / "parameter_analysis.png", dpi=300, bbox_inches='tight')
        print(f"Parameter analysis saved to {save_dir / 'parameter_analysis.png'}")
    else:
        plt.show()


def plot_learning_curves(results_dir: Path, save_dir: Path = None):
    """Plot learning curves from multiple experiments."""
    # This would require parsing training logs
    # Implementation depends on your logging setup
    print("Learning curves plotting not implemented yet")
    print("This would require parsing training logs from each experiment")


def create_summary_report(df: pd.DataFrame, target_col: str = "val_loss") -> str:
    """Create a summary report."""
    # Filter out failed trials
    successful_df = df[df[target_col].notna()]

    if len(successful_df) == 0:
        return "No successful trials found."

    # Best trial
    best_idx = successful_df[target_col].idxmin() if target_col.endswith('loss') else successful_df[target_col].idxmax()
    best_trial = successful_df.loc[best_idx]

    # Summary statistics
    report = f"""
Hyperparameter Search Results Summary
====================================

Total Trials: {len(df)}
Successful Trials: {len(successful_df)}
Success Rate: {len(successful_df) / len(df) * 100:.1f}%

Best Trial Results:
------------------
{target_col}: {best_trial[target_col]:.4f}
"""

    # Add best parameters
    param_cols = [col for col in df.columns
                  if col not in ['experiment_dir', 'trial_number', 'value', 'state', 'duration']]

    report += "\nBest Parameters:\n"
    for col in param_cols:
        if col != target_col:
            report += f"{col}: {best_trial[col]}\n"

    # Parameter importance (correlation analysis)
    numeric_cols = successful_df.select_dtypes(include=['float64', 'int64']).columns
    numeric_cols = [col for col in numeric_cols if col != target_col]

    if len(numeric_cols) > 0:
        report += f"\nParameter Correlations with {target_col}:\n"
        report += "-" * 40 + "\n"

        correlations = []
        for col in numeric_cols:
            corr = successful_df[col].corr(successful_df[target_col])
            correlations.append((col, abs(corr), corr))

        # Sort by absolute correlation
        correlations.sort(key=lambda x: x[1], reverse=True)

        for col, abs_corr, corr in correlations[:10]:  # Top 10
            report += f"{col:20s}: {corr:6.3f} (|{abs_corr:.3f}|)\n"

    # Performance statistics
    report += f"\n{target_col} Statistics:\n"
    report += "-" * 20 + "\n"
    report += f"Mean: {successful_df[target_col].mean():.4f}\n"
    report += f"Std:  {successful_df[target_col].std():.4f}\n"
    report += f"Min:  {successful_df[target_col].min():.4f}\n"
    report += f"Max:  {successful_df[target_col].max():.4f}\n"

    return report


def main():
    parser = argparse.ArgumentParser(description="Analyze hyperparameter search results")
    parser.add_argument("--results-dir", type=Path, required=True,
                       help="Directory containing experiment results")
    parser.add_argument("--optuna-db", type=Path,
                       help="Path to Optuna SQLite database")
    parser.add_argument("--output-dir", type=Path, default="./analysis_results",
                       help="Directory to save analysis results")
    parser.add_argument("--target", type=str, default="val_loss",
                       help="Target metric to analyze")

    args = parser.parse_args()

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Load results
    df = None

    if args.optuna_db and args.optuna_db.exists():
        print(f"Loading Optuna results from {args.optuna_db}")
        df = load_optuna_results(args.optuna_db)
        target_col = "value"
    elif args.results_dir.exists():
        print(f"Loading grid search results from {args.results_dir}")
        df = load_grid_search_results(args.results_dir)
        target_col = args.target
    else:
        print("No valid results found")
        return

    if df is None or len(df) == 0:
        print("No results loaded")
        return

    print(f"Loaded {len(df)} trials")

    # Create analysis plots
    plot_parameter_analysis(df, target_col, args.output_dir)

    # Generate summary report
    report = create_summary_report(df, target_col)
    print(report)

    # Save report
    with open(args.output_dir / "summary_report.txt", "w") as f:
        f.write(report)

    # Save detailed results
    df.to_csv(args.output_dir / "detailed_results.csv", index=False)

    print(f"\nAnalysis complete! Results saved to {args.output_dir}")


if __name__ == "__main__":
    main()