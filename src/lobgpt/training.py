"""
PyTorch Lightning training infrastructure for LOB Transformer.
"""

from typing import Any, Dict, Optional, Tuple

import pytorch_lightning as pl
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, OneCycleLR

from lobgpt.models.lob_transformer import LOBTransformer, LOBTransformerConfig
from lobgpt.tokenizer import LOBTokenizer


class LOBTransformerLightning(pl.LightningModule):
    """
    PyTorch Lightning module for training LOB Transformer.
    """

    def __init__(
        self,
        config: LOBTransformerConfig,
        learning_rate: float = 6e-4,
        weight_decay: float = 0.01,
        warmup_steps: int = 2000,
        max_steps: int = 100000,
        scheduler_type: str = "cosine",
        tokenizer: Optional[LOBTokenizer] = None
    ):
        super().__init__()
        self.save_hyperparameters()

        # Model
        self.model = LOBTransformer(config)
        self.config = config
        self.tokenizer = tokenizer

        # Training hyperparameters
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.warmup_steps = warmup_steps
        self.max_steps = max_steps
        self.scheduler_type = scheduler_type

        # Loss function
        self.loss_fn = torch.nn.CrossEntropyLoss(ignore_index=-100)

        # Metrics
        self.train_accuracy = pl.metrics.Accuracy(task="multiclass", num_classes=config.vocab_size)
        self.val_accuracy = pl.metrics.Accuracy(task="multiclass", num_classes=config.vocab_size)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs
    ) -> torch.Tensor:
        """Forward pass."""
        logits, _ = self.model(input_ids, attention_mask, **kwargs)
        return logits

    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Training step."""
        input_ids = batch["input_ids"]
        time_buckets = batch.get("time_buckets")

        # Shift for next-token prediction
        if input_ids.size(1) <= 1:
            return None  # Skip if sequence too short

        inputs = input_ids[:, :-1]
        targets = input_ids[:, 1:]

        # Shift time buckets if available
        time_inputs = None
        if time_buckets is not None:
            time_inputs = time_buckets[:, :-1]

        # Forward pass
        logits = self(inputs, time_buckets=time_inputs)

        # Calculate loss
        loss = self.loss_fn(
            logits.reshape(-1, self.config.vocab_size),
            targets.reshape(-1)
        )

        # Calculate perplexity
        perplexity = torch.exp(loss)

        # Calculate accuracy
        predictions = torch.argmax(logits, dim=-1)
        accuracy = (predictions == targets).float().mean()

        # Log metrics
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train_perplexity", perplexity, on_step=True, on_epoch=True)
        self.log("train_accuracy", accuracy, on_step=True, on_epoch=True)
        self.log("learning_rate", self.trainer.optimizers[0].param_groups[0]["lr"], on_step=True)

        return loss

    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Validation step."""
        input_ids = batch["input_ids"]
        time_buckets = batch.get("time_buckets")

        if input_ids.size(1) <= 1:
            return None

        inputs = input_ids[:, :-1]
        targets = input_ids[:, 1:]

        # Shift time buckets if available
        time_inputs = None
        if time_buckets is not None:
            time_inputs = time_buckets[:, :-1]

        # Forward pass
        logits = self(inputs, time_buckets=time_inputs)

        # Calculate loss
        loss = self.loss_fn(
            logits.reshape(-1, self.config.vocab_size),
            targets.reshape(-1)
        )

        # Calculate metrics
        perplexity = torch.exp(loss)
        predictions = torch.argmax(logits, dim=-1)
        accuracy = (predictions == targets).float().mean()

        # Log metrics
        self.log("val_loss", loss, on_epoch=True, prog_bar=True)
        self.log("val_perplexity", perplexity, on_epoch=True, prog_bar=True)
        self.log("val_accuracy", accuracy, on_epoch=True, prog_bar=True)

        return loss

    def test_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Test step."""
        input_ids = batch["input_ids"]
        time_buckets = batch.get("time_buckets")

        if input_ids.size(1) <= 1:
            return None

        inputs = input_ids[:, :-1]
        targets = input_ids[:, 1:]

        # Shift time buckets if available
        time_inputs = None
        if time_buckets is not None:
            time_inputs = time_buckets[:, :-1]

        # Forward pass
        logits = self(inputs, time_buckets=time_inputs)

        # Calculate loss
        loss = self.loss_fn(
            logits.reshape(-1, self.config.vocab_size),
            targets.reshape(-1)
        )

        # Calculate metrics
        perplexity = torch.exp(loss)
        predictions = torch.argmax(logits, dim=-1)
        accuracy = (predictions == targets).float().mean()

        # Event-type accuracy (extract event type from token)
        pred_events = predictions // 1000  # Assuming event type encoded in thousands
        true_events = targets // 1000
        event_accuracy = (pred_events == true_events).float().mean()

        # Log metrics
        self.log("test_loss", loss, on_epoch=True)
        self.log("test_perplexity", perplexity, on_epoch=True)
        self.log("test_accuracy", accuracy, on_epoch=True)
        self.log("test_event_accuracy", event_accuracy, on_epoch=True)

        return loss

    def on_validation_epoch_end(self) -> None:
        """End of validation epoch."""
        if self.current_epoch % 10 == 0:  # Generate every 10 epochs
            self._generate_samples()

    def _generate_samples(self, num_samples: int = 3, max_length: int = 50):
        """Generate and log sample sequences."""
        if self.tokenizer is None:
            return

        self.model.eval()
        with torch.no_grad():
            # Create a simple prompt (first few tokens from validation set)
            prompt = torch.randint(1, 100, (num_samples, 10), device=self.device)

            # Generate
            generated = self.model.generate(
                prompt,
                max_length=max_length,
                temperature=1.0,
                top_k=50,
                do_sample=True
            )

            # Decode and log
            for i, seq in enumerate(generated):
                try:
                    decoded = self.tokenizer.decode_tokens(seq.cpu().numpy())
                    preview = [token.to_string() for token in decoded[:10]]
                    self.logger.experiment.log({
                        f"sample_{i}": "\n".join(preview)
                    })
                except Exception as e:
                    self.logger.experiment.log({f"sample_{i}": f"Decode error: {e}"})

    def configure_optimizers(self) -> Dict[str, Any]:
        """Configure optimizers and schedulers."""
        # Separate decay parameters
        decay_params = []
        no_decay_params = []

        for name, param in self.model.named_parameters():
            if param.requires_grad:
                if "bias" in name or "norm" in name or "embedding" in name:
                    no_decay_params.append(param)
                else:
                    decay_params.append(param)

        # Create optimizer
        optimizer = AdamW([
            {"params": decay_params, "weight_decay": self.weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0}
        ], lr=self.learning_rate, betas=(0.9, 0.95))

        # Create scheduler
        if self.scheduler_type == "cosine":
            scheduler = CosineAnnealingLR(
                optimizer,
                T_max=self.max_steps,
                eta_min=self.learning_rate * 0.1
            )
        elif self.scheduler_type == "onecycle":
            scheduler = OneCycleLR(
                optimizer,
                max_lr=self.learning_rate,
                total_steps=self.max_steps,
                pct_start=self.warmup_steps / self.max_steps
            )
        else:
            raise ValueError(f"Unknown scheduler type: {self.scheduler_type}")

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            }
        }

    def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        """Save additional information with checkpoint."""
        checkpoint["config"] = self.config.__dict__
        checkpoint["tokenizer"] = self.tokenizer


class LOBDataModule(pl.LightningDataModule):
    """
    PyTorch Lightning data module for LOB token data.
    """

    def __init__(
        self,
        train_dataset,
        val_dataset,
        test_dataset=None,
        batch_size: int = 32,
        num_workers: int = 4,
        pin_memory: bool = True
    ):
        super().__init__()
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=True,
            collate_fn=self._collate_fn
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=self._collate_fn
        )

    def test_dataloader(self):
        if self.test_dataset is None:
            return None
        return torch.utils.data.DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=self._collate_fn
        )

    def _collate_fn(self, batch):
        """Custom collate function for variable-length sequences."""
        # Handle new dictionary format from TokenBookDataset
        if isinstance(batch[0], dict):
            # Extract components
            input_ids = [item["input_ids"] for item in batch]
            time_buckets = [item.get("time_buckets") for item in batch]
            labels = [item.get("labels") for item in batch]

            # Pad sequences to same length
            max_len = max(len(seq) for seq in input_ids)
            padded_ids = []
            padded_time_buckets = []

            for i, seq in enumerate(input_ids):
                if len(seq) < max_len:
                    # Pad with zeros
                    padded = torch.cat([seq, torch.zeros(max_len - len(seq), dtype=seq.dtype)])
                else:
                    padded = seq[:max_len]
                padded_ids.append(padded)

                # Pad time buckets if available
                if time_buckets[i] is not None:
                    time_seq = time_buckets[i]
                    if len(time_seq) < max_len:
                        padded_time = torch.cat([time_seq, torch.zeros(max_len - len(time_seq), dtype=time_seq.dtype)])
                    else:
                        padded_time = time_seq[:max_len]
                    padded_time_buckets.append(padded_time)

            result = {
                "input_ids": torch.stack(padded_ids)
            }

            if padded_time_buckets:
                result["time_buckets"] = torch.stack(padded_time_buckets)

            if labels and labels[0] is not None:
                result["labels"] = torch.stack(labels)

            return result

        else:
            # Legacy format (tuple)
            input_ids = [item[0] for item in batch]

            # Pad sequences to same length
            max_len = max(len(seq) for seq in input_ids)
            padded_ids = []

            for seq in input_ids:
                if len(seq) < max_len:
                    # Pad with zeros (or special padding token)
                    padded = torch.cat([seq, torch.zeros(max_len - len(seq), dtype=seq.dtype)])
                else:
                    padded = seq[:max_len]
                padded_ids.append(padded)

            return {
                "input_ids": torch.stack(padded_ids)
            }


def create_trainer(
    config: LOBTransformerConfig,
    train_dataset,
    val_dataset,
    test_dataset=None,
    max_epochs: int = 100,
    batch_size: int = 32,
    learning_rate: float = 6e-4,
    project_name: str = "lob-transformer",
    experiment_name: Optional[str] = None,
    checkpoint_dir: str = "./checkpoints",
    **trainer_kwargs
) -> Tuple[pl.Trainer, LOBTransformerLightning, LOBDataModule]:
    """
    Create trainer, model, and data module.

    Args:
        config: Model configuration
        train_dataset: Training dataset
        val_dataset: Validation dataset
        test_dataset: Test dataset (optional)
        max_epochs: Maximum training epochs
        batch_size: Batch size
        learning_rate: Learning rate
        project_name: Wandb project name
        experiment_name: Experiment name
        checkpoint_dir: Checkpoint directory
        **trainer_kwargs: Additional trainer arguments

    Returns:
        trainer: Lightning trainer
        model: Lightning model
        data_module: Lightning data module
    """
    # Data module
    data_module = LOBDataModule(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        test_dataset=test_dataset,
        batch_size=batch_size
    )

    # Model
    model = LOBTransformerLightning(
        config=config,
        learning_rate=learning_rate,
        max_steps=max_epochs * len(train_dataset) // batch_size
    )

    # Callbacks
    callbacks = [
        pl.callbacks.ModelCheckpoint(
            dirpath=checkpoint_dir,
            filename="{epoch}-{val_loss:.3f}",
            monitor="val_loss",
            mode="min",
            save_top_k=3,
            save_last=True
        ),
        pl.callbacks.EarlyStopping(
            monitor="val_loss",
            mode="min",
            patience=10,
            verbose=True
        ),
        pl.callbacks.LearningRateMonitor(logging_interval="step"),
        pl.callbacks.RichProgressBar()
    ]

    # Logger
    logger = pl.loggers.WandbLogger(
        project=project_name,
        name=experiment_name,
        log_model=True
    )

    # Trainer
    trainer = pl.Trainer(
        max_epochs=max_epochs,
        accelerator="auto",
        devices="auto",
        strategy="auto",
        precision="16-mixed",
        gradient_clip_val=1.0,
        accumulate_grad_batches=1,
        callbacks=callbacks,
        logger=logger,
        val_check_interval=0.25,  # Validate 4 times per epoch
        log_every_n_steps=50,
        **trainer_kwargs
    )

    return trainer, model, data_module


def train_lob_transformer(
    config: LOBTransformerConfig,
    train_dataset,
    val_dataset,
    test_dataset=None,
    **kwargs
) -> LOBTransformerLightning:
    """
    Full training pipeline.

    Args:
        config: Model configuration
        train_dataset: Training dataset
        val_dataset: Validation dataset
        test_dataset: Test dataset (optional)
        **kwargs: Additional arguments for create_trainer

    Returns:
        Trained model
    """
    # Create trainer
    trainer, model, data_module = create_trainer(
        config, train_dataset, val_dataset, test_dataset, **kwargs
    )

    # Train
    trainer.fit(model, data_module)

    # Test if test dataset provided
    if test_dataset is not None:
        trainer.test(model, data_module)

    return model
