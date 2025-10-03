"""
Native PyTorch Dataset classes for Limit Order Book data.
"""

from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
import polars as pl
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset


class L2BookDataset(Dataset):
    """
    PyTorch Dataset for L2 order book data in DeepLOB format.

    Generates CNN-compatible tensors of shape (1, seq_len, features)
    where features = 4 * depth (price and volume for each level on both sides).
    """

    def __init__(
        self,
        book_states: pl.DataFrame,
        sequence_length: int = 100,
        depth: int = 10,
        horizon: int = 10,
        threshold: float = 0.002,
        normalize: bool = True,
        label_type: str = "classification",
        stride: int = 1,
        scaler: Optional[StandardScaler] = None,
        cache: bool = False
    ):
        """
        Initialize L2 Book Dataset for DeepLOB format.

        Args:
            book_states: DataFrame from reconstruct_book_states()
            sequence_length: Length of sequences
            depth: Number of price levels
            horizon: Prediction horizon for labels
            threshold: Price movement threshold for classification
            normalize: Whether to normalize features
            label_type: "classification" or "regression"
            stride: Step size between sequences
            scaler: Pre-fitted StandardScaler (if None, will fit on data)
            cache: Whether to cache processed sequences in memory
        """
        self.book_states = book_states
        self.sequence_length = sequence_length
        self.depth = depth
        self.horizon = horizon
        self.threshold = threshold
        self.normalize = normalize
        self.label_type = label_type
        self.stride = stride
        self.external_scaler = scaler
        self.cache = cache

        # Prepare data
        self._prepare_data()

        # Cache for processed sequences
        self._cache = {} if cache else None

    def _prepare_data(self):
        """Prepare data in DeepLOB format."""

        # Extract features in DeepLOB order
        features = []
        for level in range(self.depth):
            features.extend([
                f'asks[{level}].price',
                f'asks[{level}].amount',
                f'bids[{level}].price',
                f'bids[{level}].amount'
            ])

        # Convert to numpy
        self.data = self.book_states.select(features).to_numpy()

        # Handle infinite values
        self.data = np.where(np.isinf(self.data), np.nan, self.data)

        # Forward/backward fill NaN values
        df_temp = pd.DataFrame(self.data)
        df_temp = df_temp.ffill().bfill()
        self.data = df_temp.values

        # Normalize if requested
        self.scaler = self.external_scaler
        if self.normalize:
            if self.scaler is None:
                self.scaler = StandardScaler()
                data_reshaped = self.data.reshape(-1, self.data.shape[-1])
                data_reshaped = self.scaler.fit_transform(data_reshaped)
            else:
                data_reshaped = self.data.reshape(-1, self.data.shape[-1])
                data_reshaped = self.scaler.transform(data_reshaped)
            self.data = data_reshaped.reshape(self.data.shape)

        # Prepare labels
        self._prepare_labels()

        # Calculate number of sequences
        self.n_sequences = max(0, (len(self.data) - self.sequence_length - self.horizon + 1) // self.stride)


    def _prepare_labels(self):
        """Prepare labels based on mid-price movement."""

        # Calculate mid prices
        bid_prices = self.book_states['bids[0].price'].to_numpy()
        ask_prices = self.book_states['asks[0].price'].to_numpy()

        # Handle infinite values
        bid_prices = np.where(np.isinf(bid_prices), np.nan, bid_prices)
        ask_prices = np.where(np.isinf(ask_prices), np.nan, ask_prices)

        mid_prices = (bid_prices + ask_prices) / 2
        mid_prices = pd.Series(mid_prices).ffill().bfill().values

        self.labels = []

        for i in range(0, len(mid_prices) - self.sequence_length - self.horizon + 1, self.stride):
            current_mid = mid_prices[i + self.sequence_length - 1]
            future_mid = mid_prices[i + self.sequence_length + self.horizon - 1]

            if np.isnan(current_mid) or np.isnan(future_mid) or current_mid == 0:
                label = 1  # Default to stationary
            elif self.label_type == "classification":
                # 3-class: 0=down, 1=stationary, 2=up
                price_change = (future_mid - current_mid) / current_mid

                if price_change < -self.threshold:
                    label = 0  # Down
                elif price_change > self.threshold:
                    label = 2  # Up
                else:
                    label = 1  # Stationary

            elif self.label_type == "regression":
                label = (future_mid - current_mid) / current_mid if current_mid != 0 else 0.0

            self.labels.append(label)

        self.labels = np.array(self.labels)

    def __len__(self) -> int:
        """Return number of sequences."""
        return self.n_sequences

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a single sequence and label.

        Returns:
            x: Input sequence
            y: Target label
        """

        # Check cache
        if self._cache is not None and idx in self._cache:
            return self._cache[idx]

        # Calculate actual index based on stride
        start_idx = idx * self.stride
        end_idx = start_idx + self.sequence_length

        # Get sequence - Shape: (seq_len, features)
        sequence = self.data[start_idx:end_idx]
        # Convert to tensor and add channel dimension for CNN
        x = torch.tensor(sequence, dtype=torch.float32).unsqueeze(0)

        # Get label
        if idx < len(self.labels):
            if self.label_type == "classification":
                y = torch.tensor(self.labels[idx], dtype=torch.long)
            else:
                y = torch.tensor(self.labels[idx], dtype=torch.float32)
        else:
            # Default label if index is out of range
            y = torch.tensor(1 if self.label_type == "classification" else 0.0,
                           dtype=torch.long if self.label_type == "classification" else torch.float32)

        # Cache if enabled
        if self._cache is not None:
            self._cache[idx] = (x, y)

        return x, y

    def get_scaler(self) -> Optional[StandardScaler]:
        """Return the scaler used for normalization."""
        return self.scaler

    def get_stats(self) -> Dict[str, Any]:
        """Return dataset statistics."""

        if len(self.labels) == 0:
            return {
                "n_sequences": 0,
                "error": "No valid sequences found"
            }

        stats = {
            "n_sequences": self.n_sequences,
            "sequence_length": self.sequence_length,
            "depth": self.depth,
            "format": "deeplob",
            "normalized": self.normalize,
            "label_type": self.label_type,
            "horizon": self.horizon,
            "threshold": self.threshold
        }

        if self.label_type == "classification":
            unique_labels, counts = np.unique(self.labels, return_counts=True)
            stats["label_distribution"] = {
                int(label): int(count) for label, count in zip(unique_labels, counts)
            }
        else:
            stats["label_stats"] = {
                "mean": float(np.mean(self.labels)),
                "std": float(np.std(self.labels)),
                "min": float(np.min(self.labels)),
                "max": float(np.max(self.labels))
            }

        return stats


class TokenBookDataset(Dataset):
    """
    PyTorch Dataset for tokenized order book events.

    Generates sequences of token IDs for transformer models.
    """

    def __init__(
        self,
        inc_df: pl.DataFrame,
        book_states: pl.DataFrame,
        tokenizer: Any,
        sequence_length: int = 100,
        horizon: int = 10,
        threshold: float = 0.002,
        label_type: str = "classification",
        stride: int = 1,
        cache: bool = False
    ):
        """
        Initialize Tokenized Book Dataset.

        Args:
            inc_df: Incremental updates from load_inc()
            book_states: DataFrame from reconstruct_book_states()
            tokenizer: LOBTokenizer instance
            sequence_length: Length of sequences
            horizon: Prediction horizon for labels
            threshold: Price movement threshold for classification
            label_type: "classification" or "regression"
            stride: Step size between sequences
            cache: Whether to cache processed sequences in memory
        """
        self.inc_df = inc_df
        self.book_states = book_states
        self.tokenizer = tokenizer
        self.sequence_length = sequence_length
        self.horizon = horizon
        self.threshold = threshold
        self.label_type = label_type
        self.stride = stride
        self.cache = cache

        # Prepare data
        self._prepare_data()

        # Cache for processed sequences
        self._cache = {} if cache else None

    def _prepare_data(self):
        """Prepare tokenized data."""

        # Tokenize events
        self.tokens = self.tokenizer.tokenize_events(self.inc_df, self.book_states)

        # Prepare labels from book states
        self._prepare_labels()

        # Calculate number of sequences
        self.n_sequences = max(0, (len(self.tokens) - self.sequence_length - self.horizon + 1) // self.stride)

    def _prepare_labels(self):
        """Prepare labels based on mid-price movement."""

        # Calculate mid prices
        bid_prices = self.book_states['bids[0].price'].to_numpy()
        ask_prices = self.book_states['asks[0].price'].to_numpy()

        # Handle infinite values
        bid_prices = np.where(np.isinf(bid_prices), np.nan, bid_prices)
        ask_prices = np.where(np.isinf(ask_prices), np.nan, ask_prices)

        mid_prices = (bid_prices + ask_prices) / 2
        mid_prices = pd.Series(mid_prices).ffill().bfill().values

        self.labels = []

        for i in range(0, min(len(self.tokens), len(mid_prices)) - self.sequence_length - self.horizon + 1, self.stride):
            current_mid = mid_prices[min(i + self.sequence_length - 1, len(mid_prices) - 1)]
            future_idx = min(i + self.sequence_length + self.horizon - 1, len(mid_prices) - 1)
            future_mid = mid_prices[future_idx]

            if np.isnan(current_mid) or np.isnan(future_mid) or current_mid == 0:
                label = 1  # Default to stationary
            elif self.label_type == "classification":
                price_change = (future_mid - current_mid) / current_mid

                if price_change < -self.threshold:
                    label = 0  # Down
                elif price_change > self.threshold:
                    label = 2  # Up
                else:
                    label = 1  # Stationary

            elif self.label_type == "regression":
                label = (future_mid - current_mid) / current_mid if current_mid != 0 else 0.0

            self.labels.append(label)

        self.labels = np.array(self.labels)

    def __len__(self) -> int:
        """Return number of sequences."""
        return self.n_sequences

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get a single sequence and label."""

        # Check cache
        if self._cache is not None and idx in self._cache:
            return self._cache[idx]

        # Calculate actual index based on stride
        start_idx = idx * self.stride
        end_idx = start_idx + self.sequence_length

        # Get token sequence
        sequence = self.tokens[start_idx:end_idx]
        x = torch.tensor(sequence, dtype=torch.long)

        # Get label
        if idx < len(self.labels):
            if self.label_type == "classification":
                y = torch.tensor(self.labels[idx], dtype=torch.long)
            else:
                y = torch.tensor(self.labels[idx], dtype=torch.float32)
        else:
            y = torch.tensor(1 if self.label_type == "classification" else 0.0,
                           dtype=torch.long if self.label_type == "classification" else torch.float32)

        # Cache if enabled
        if self._cache is not None:
            self._cache[idx] = (x, y)

        return x, y

    def get_vocab_size(self) -> int:
        """Return vocabulary size from tokenizer."""
        return len(self.tokenizer.vocab)

    def get_stats(self) -> Dict[str, Any]:
        """Return dataset statistics."""

        stats = {
            "n_sequences": self.n_sequences,
            "sequence_length": self.sequence_length,
            "vocab_size": self.get_vocab_size(),
            "label_type": self.label_type,
            "horizon": self.horizon,
            "threshold": self.threshold
        }

        if self.label_type == "classification" and len(self.labels) > 0:
            unique_labels, counts = np.unique(self.labels, return_counts=True)
            stats["label_distribution"] = {
                int(label): int(count) for label, count in zip(unique_labels, counts)
            }

        return stats


class RawBookDataset(Dataset):
    """
    PyTorch Dataset for raw order book features.

    Uses all available numeric features without specific formatting.
    """

    def __init__(
        self,
        book_states: pl.DataFrame,
        sequence_length: int = 100,
        horizon: int = 10,
        threshold: float = 0.002,
        normalize: bool = True,
        label_type: str = "classification",
        stride: int = 1,
        scaler: Optional[StandardScaler] = None,
        feature_cols: Optional[list] = None,
        cache: bool = False
    ):
        """
        Initialize Raw Book Dataset.

        Args:
            book_states: DataFrame from reconstruct_book_states()
            sequence_length: Length of sequences
            horizon: Prediction horizon for labels
            threshold: Price movement threshold for classification
            normalize: Whether to normalize features
            label_type: "classification" or "regression"
            stride: Step size between sequences
            scaler: Pre-fitted StandardScaler (if None, will fit on data)
            feature_cols: List of columns to use (if None, uses all numeric)
            cache: Whether to cache processed sequences in memory
        """
        self.book_states = book_states
        self.sequence_length = sequence_length
        self.horizon = horizon
        self.threshold = threshold
        self.normalize = normalize
        self.label_type = label_type
        self.stride = stride
        self.external_scaler = scaler
        self.feature_cols = feature_cols
        self.cache = cache

        # Prepare data
        self._prepare_data()

        # Cache for processed sequences
        self._cache = {} if cache else None

    def _prepare_data(self):
        """Prepare raw features."""

        # Select feature columns
        if self.feature_cols is None:
            # Use all numeric features
            self.feature_cols = [col for col in self.book_states.columns
                                if col not in ['ts', 'ts_local'] and
                                self.book_states[col].dtype in [pl.Float32, pl.Float64, pl.Int32, pl.Int64]]

        self.data = self.book_states.select(self.feature_cols).to_numpy()

        # Handle infinite values
        self.data = np.where(np.isinf(self.data), 0, self.data)
        self.data = np.nan_to_num(self.data, 0)

        # Normalize if requested
        self.scaler = self.external_scaler
        if self.normalize:
            if self.scaler is None:
                self.scaler = StandardScaler()
                self.data = self.scaler.fit_transform(self.data)
            else:
                self.data = self.scaler.transform(self.data)

        # Prepare labels
        self._prepare_labels()

        # Calculate number of sequences
        self.n_sequences = max(0, (len(self.data) - self.sequence_length - self.horizon + 1) // self.stride)

    def _prepare_labels(self):
        """Prepare labels based on mid-price movement."""

        # Calculate mid prices
        bid_prices = self.book_states['bids[0].price'].to_numpy()
        ask_prices = self.book_states['asks[0].price'].to_numpy()

        # Handle infinite values
        bid_prices = np.where(np.isinf(bid_prices), np.nan, bid_prices)
        ask_prices = np.where(np.isinf(ask_prices), np.nan, ask_prices)

        mid_prices = (bid_prices + ask_prices) / 2
        mid_prices = pd.Series(mid_prices).ffill().bfill().values

        self.labels = []

        for i in range(0, len(mid_prices) - self.sequence_length - self.horizon + 1, self.stride):
            current_mid = mid_prices[i + self.sequence_length - 1]
            future_mid = mid_prices[i + self.sequence_length + self.horizon - 1]

            if np.isnan(current_mid) or np.isnan(future_mid) or current_mid == 0:
                label = 1  # Default to stationary
            elif self.label_type == "classification":
                price_change = (future_mid - current_mid) / current_mid

                if price_change < -self.threshold:
                    label = 0  # Down
                elif price_change > self.threshold:
                    label = 2  # Up
                else:
                    label = 1  # Stationary

            elif self.label_type == "regression":
                label = (future_mid - current_mid) / current_mid if current_mid != 0 else 0.0

            self.labels.append(label)

        self.labels = np.array(self.labels)

    def __len__(self) -> int:
        """Return number of sequences."""
        return self.n_sequences

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get a single sequence and label."""

        # Check cache
        if self._cache is not None and idx in self._cache:
            return self._cache[idx]

        # Calculate actual index based on stride
        start_idx = idx * self.stride
        end_idx = start_idx + self.sequence_length

        # Get sequence
        sequence = self.data[start_idx:end_idx]
        x = torch.tensor(sequence, dtype=torch.float32)

        # Get label
        if idx < len(self.labels):
            if self.label_type == "classification":
                y = torch.tensor(self.labels[idx], dtype=torch.long)
            else:
                y = torch.tensor(self.labels[idx], dtype=torch.float32)
        else:
            y = torch.tensor(1 if self.label_type == "classification" else 0.0,
                           dtype=torch.long if self.label_type == "classification" else torch.float32)

        # Cache if enabled
        if self._cache is not None:
            self._cache[idx] = (x, y)

        return x, y

    def get_scaler(self) -> Optional[StandardScaler]:
        """Return the scaler used for normalization."""
        return self.scaler

    def get_feature_names(self) -> list:
        """Return list of feature column names."""
        return self.feature_cols

    def get_stats(self) -> Dict[str, Any]:
        """Return dataset statistics."""

        if len(self.labels) == 0:
            return {
                "n_sequences": 0,
                "error": "No valid sequences found"
            }

        stats = {
            "n_sequences": self.n_sequences,
            "sequence_length": self.sequence_length,
            "n_features": len(self.feature_cols),
            "format": "raw",
            "normalized": self.normalize,
            "label_type": self.label_type,
            "horizon": self.horizon,
            "threshold": self.threshold
        }

        if self.label_type == "classification":
            unique_labels, counts = np.unique(self.labels, return_counts=True)
            stats["label_distribution"] = {
                int(label): int(count) for label, count in zip(unique_labels, counts)
            }
        else:
            stats["label_stats"] = {
                "mean": float(np.mean(self.labels)),
                "std": float(np.std(self.labels)),
                "min": float(np.min(self.labels)),
                "max": float(np.max(self.labels))
            }

        return stats


def create_lob_pytorch_dataloaders(
    train_states: pl.DataFrame,
    val_states: Optional[pl.DataFrame] = None,
    test_states: Optional[pl.DataFrame] = None,
    batch_size: int = 32,
    format: str = "deeplob",
    sequence_length: int = 100,
    depth: int = 10,
    horizon: int = 10,
    threshold: float = 0.002,
    normalize: bool = True,
    label_type: str = "classification",
    stride: int = 1,
    num_workers: int = 0,
    shuffle_train: bool = True,
    pin_memory: bool = True,
    drop_last: bool = False
) -> Tuple[Dict[str, DataLoader], StandardScaler]:
    """
    Create PyTorch DataLoaders for training, validation, and testing.

    Returns:
        Tuple of (DataLoaders dict, scaler used for normalization)
    """

    dataloaders = {}

    # Create training dataset based on format
    if format == "deeplob":
        train_dataset = L2BookDataset(
            train_states, sequence_length=sequence_length,
            depth=depth, horizon=horizon, threshold=threshold,
            normalize=normalize, label_type=label_type, stride=stride
        )
    elif format == "raw":
        train_dataset = RawBookDataset(
            train_states, sequence_length=sequence_length,
            horizon=horizon, threshold=threshold,
            normalize=normalize, label_type=label_type, stride=stride
        )
    else:
        raise ValueError(f"Unsupported format for this function: {format}. Use TokenBookDataset directly for tokenized format.")

    print(f"Training dataset: {train_dataset.get_stats()}")

    dataloaders['train'] = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=shuffle_train,
        num_workers=num_workers, pin_memory=pin_memory, drop_last=drop_last
    )

    # Get training scaler for validation and test
    train_scaler = train_dataset.get_scaler()

    # Create validation loader if provided
    if val_states is not None:
        if format == "deeplob":
            val_dataset = L2BookDataset(
                val_states, sequence_length=sequence_length,
                depth=depth, horizon=horizon, threshold=threshold,
                normalize=normalize, label_type=label_type, stride=stride * 10,  # Sparser sampling
                scaler=train_scaler  # Use training scaler
            )
        elif format == "raw":
            val_dataset = RawBookDataset(
                val_states, sequence_length=sequence_length,
                horizon=horizon, threshold=threshold,
                normalize=normalize, label_type=label_type, stride=stride * 10,
                scaler=train_scaler
            )

        print(f"Validation dataset: {val_dataset.get_stats()}")

        dataloaders['val'] = DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=pin_memory, drop_last=False
        )

    # Create test loader if provided
    if test_states is not None:
        if format == "deeplob":
            test_dataset = L2BookDataset(
                test_states, sequence_length=sequence_length,
                depth=depth, horizon=horizon, threshold=threshold,
                normalize=normalize, label_type=label_type, stride=stride * 10,  # Sparser sampling
                scaler=train_scaler  # Use training scaler
            )
        elif format == "raw":
            test_dataset = RawBookDataset(
                test_states, sequence_length=sequence_length,
                horizon=horizon, threshold=threshold,
                normalize=normalize, label_type=label_type, stride=stride * 10,
                scaler=train_scaler
            )

        print(f"Test dataset: {test_dataset.get_stats()}")

        dataloaders['test'] = DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=pin_memory, drop_last=False
        )

    return dataloaders, train_scaler