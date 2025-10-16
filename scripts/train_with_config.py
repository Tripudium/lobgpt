"""
Configurable training script for LOB Transformer using Hydra.
"""

from typing import Any, Dict

import hydra
import pytorch_lightning as pl
import wandb
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
    RichProgressBar,
)
from pytorch_lightning.loggers import WandbLogger

from lobgpt.book_state import reconstruct_from_tardis
from lobgpt.hdb import get_dataset
from lobgpt.models.lob_transformer import LOBTransformer, LOBTransformerConfig
from lobgpt.dataset import TokenBookDataset
from lobgpt.tokenizer import LOBTokenizer
from lobgpt.training import LOBDataModule, LOBTransformerLightning


def create_datasets(cfg: DictConfig):
    """Create datasets from configuration."""
    print("Loading and preparing data...")

    # Load data
    dl = get_dataset("tardis")
    product = cfg.data.product

    # Training data
    train_book_states = reconstruct_from_tardis(
        product, cfg.data.train_times, depth=cfg.data.depth
    )
    train_inc_df = dl.load_inc(product, cfg.data.train_times)

    # Validation data
    val_book_states = reconstruct_from_tardis(
        product, cfg.data.val_times, depth=cfg.data.depth
    )
    val_inc_df = dl.load_inc(product, cfg.data.val_times)

    # Test data (optional)
    test_book_states = None
    test_inc_df = None
    if hasattr(cfg.data, 'test_times') and cfg.data.test_times:
        test_book_states = reconstruct_from_tardis(
            product, cfg.data.test_times, depth=cfg.data.depth
        )
        test_inc_df = dl.load_inc(product, cfg.data.test_times)

    print(f"Train book states: {len(train_book_states)}")
    print(f"Train inc events: {len(train_inc_df)}")
    print(f"Val book states: {len(val_book_states)}")
    print(f"Val inc events: {len(val_inc_df)}")

    # Create tokenizer
    tokenizer = LOBTokenizer(
        tick_size=cfg.data.tokenizer.tick_size,
        ref_size=cfg.data.tokenizer.ref_size
    )

    # Create datasets
    train_dataset = TokenBookDataset(
        inc_df=train_inc_df,
        book_states=train_book_states,
        tokenizer=tokenizer,
        sequence_length=cfg.data.sequence_length,
        horizon=cfg.data.horizon,
        threshold=cfg.data.threshold,
        stride=cfg.data.train_stride
    )

    val_dataset = TokenBookDataset(
        inc_df=val_inc_df,
        book_states=val_book_states,
        tokenizer=tokenizer,
        sequence_length=cfg.data.sequence_length,
        horizon=cfg.data.horizon,
        threshold=cfg.data.threshold,
        stride=cfg.data.val_stride
    )

    test_dataset = None
    if test_inc_df is not None:
        test_dataset = TokenBookDataset(
            inc_df=test_inc_df,
            book_states=test_book_states,
            tokenizer=tokenizer,
            sequence_length=cfg.data.sequence_length,
            horizon=cfg.data.horizon,
            threshold=cfg.data.threshold,
            stride=cfg.data.test_stride
        )

    print(f"Train dataset: {train_dataset.get_stats()}")
    print(f"Val dataset: {val_dataset.get_stats()}")

    return train_dataset, val_dataset, test_dataset, tokenizer


def create_model_config(cfg: DictConfig, vocab_size: int) -> LOBTransformerConfig:
    """Create model configuration from Hydra config."""
    model_cfg = cfg.model.copy()

    # Override vocab size
    model_cfg.vocab_size = vocab_size

    # Set d_ff if not specified (common practice: 4 * d_model)
    if model_cfg.d_ff is None:
        model_cfg.d_ff = 4 * model_cfg.d_model

    # Handle constraints
    if model_cfg.n_heads > 0:
        assert model_cfg.d_model % model_cfg.n_heads == 0, \
            f"d_model ({model_cfg.d_model}) must be divisible by n_heads ({model_cfg.n_heads})"

    return LOBTransformerConfig(**model_cfg)


def create_callbacks(cfg: DictConfig) -> list:
    """Create training callbacks."""
    callbacks = []

    # Model checkpointing
    checkpoint_callback = ModelCheckpoint(
        dirpath=cfg.checkpoint_dir,
        filename=cfg.checkpoint.filename,
        monitor=cfg.checkpoint.monitor,
        mode=cfg.checkpoint.mode,
        save_top_k=cfg.checkpoint.save_top_k,
        save_last=cfg.checkpoint.save_last,
        verbose=True
    )
    callbacks.append(checkpoint_callback)

    # Early stopping
    if cfg.early_stopping.patience > 0:
        early_stop_callback = EarlyStopping(
            monitor=cfg.early_stopping.monitor,
            patience=cfg.early_stopping.patience,
            mode=cfg.early_stopping.mode,
            verbose=True
        )
        callbacks.append(early_stop_callback)

    # Learning rate monitoring
    lr_monitor = LearningRateMonitor(logging_interval="step")
    callbacks.append(lr_monitor)

    # Progress bar
    progress_bar = RichProgressBar()
    callbacks.append(progress_bar)

    return callbacks


def create_logger(cfg: DictConfig) -> pl.loggers.Logger:
    """Create logger."""
    if cfg.wandb.enabled:
        logger = WandbLogger(
            project=cfg.wandb.project,
            name=cfg.wandb.name,
            tags=cfg.wandb.tags,
            log_model=True
        )
        # Log configuration
        logger.experiment.config.update(OmegaConf.to_container(cfg, resolve=True))
    else:
        logger = pl.loggers.TensorBoardLogger(
            save_dir=cfg.output_dir,
            name=cfg.experiment_name
        )

    return logger


def train_model(cfg: DictConfig) -> Dict[str, Any]:
    """Train the model and return metrics."""
    # Set random seed
    pl.seed_everything(cfg.seed, workers=True)

    # Create datasets
    train_dataset, val_dataset, test_dataset, tokenizer = create_datasets(cfg)

    # Create model configuration
    model_config = create_model_config(cfg, len(tokenizer.vocab))

    print("\nModel configuration:")
    print(f"  Vocab size: {model_config.vocab_size}")
    print(f"  Model dim: {model_config.d_model}")
    print(f"  Layers: {model_config.n_layers}")
    print(f"  Heads: {model_config.n_heads}")
    print(f"  Feed forward: {model_config.d_ff}")

    # Calculate model parameters
    model = LOBTransformer(model_config)
    n_params = model.get_num_params()
    print(f"  Parameters: {n_params:,} ({n_params/1e6:.1f}M)")

    # Create data module
    data_module = LOBDataModule(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        test_dataset=test_dataset,
        batch_size=cfg.data.batch_size,
        num_workers=cfg.data.num_workers,
        pin_memory=cfg.data.pin_memory
    )

    # Create Lightning module
    lightning_model = LOBTransformerLightning(
        config=model_config,
        learning_rate=cfg.training.learning_rate,
        weight_decay=cfg.training.weight_decay,
        warmup_steps=cfg.training.warmup_steps,
        max_steps=cfg.training.max_epochs * len(train_dataset) // cfg.data.batch_size,
        scheduler_type=cfg.training.scheduler_type,
        tokenizer=tokenizer
    )

    # Create callbacks and logger
    callbacks = create_callbacks(cfg)
    logger = create_logger(cfg)

    # Create trainer
    trainer = pl.Trainer(
        max_epochs=cfg.training.max_epochs,
        accelerator=cfg.accelerator,
        devices=cfg.devices,
        precision=cfg.precision,
        gradient_clip_val=cfg.training.gradient_clip_val,
        accumulate_grad_batches=cfg.training.accumulate_grad_batches,
        callbacks=callbacks,
        logger=logger,
        val_check_interval=cfg.val_check_interval,
        log_every_n_steps=cfg.log_every_n_steps,
        deterministic=True
    )

    # Train
    print("\nStarting training...")
    trainer.fit(lightning_model, data_module)

    # Test if test dataset available
    test_results = {}
    if test_dataset is not None:
        print("\nRunning test...")
        test_results = trainer.test(lightning_model, data_module)

    # Get best metrics
    best_val_loss = trainer.callback_metrics.get("val_loss", float('inf'))
    best_val_accuracy = trainer.callback_metrics.get("val_accuracy", 0.0)

    results = {
        "val_loss": float(best_val_loss),
        "val_accuracy": float(best_val_accuracy),
        "test_results": test_results,
        "n_params": n_params,
        "model_config": OmegaConf.to_container(model_config, resolve=True)
    }

    # Clean up wandb
    if cfg.wandb.enabled:
        wandb.finish()

    return results


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> float:
    """Main training function."""
    print("="*60)
    print("LOB Transformer Training")
    print("="*60)
    print(f"Config: {OmegaConf.to_yaml(cfg)}")

    try:
        results = train_model(cfg)

        print("\n" + "="*60)
        print("Training completed successfully!")
        print(f"Best validation loss: {results['val_loss']:.4f}")
        print(f"Best validation accuracy: {results['val_accuracy']:.4f}")
        print(f"Model parameters: {results['n_params']:,}")
        print("="*60)

        # Return metric for hyperparameter optimization
        return results["val_loss"]

    except Exception as e:
        print(f"\nTraining failed: {e}")
        import traceback
        traceback.print_exc()
        return float('inf')


if __name__ == "__main__":
    main()
