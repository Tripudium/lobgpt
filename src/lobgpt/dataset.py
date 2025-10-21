"""
Native PyTorch Dataset classes for Limit Order Book data.
"""

from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import polars as pl
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset

from lobgpt.time_encoding import TimeEncoder, create_time_encoder_for_crypto
from lobgpt.preprocessing import prepare_message_volume_features, preprocess_messages_for_tokenization
from lobgpt.tokenizer import ConfigurableTokenizer, DEFAULT_TOKENIZER_CONFIG_PATH


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
            counts = np.bincount(self.labels.astype(int), minlength=3)
            stats["label_distribution"] = {
                "down": int(counts[0]),
                "stationary": int(counts[1]),
                "up": int(counts[2])
            }
        else:
            stats["label_mean"] = float(np.mean(self.labels))
            stats["label_std"] = float(np.std(self.labels))

        return stats


class TokenBookDataset(Dataset):
    """
    Tokenized dataset for LOB Transformer training.
    """

    def __init__(
        self,
        inc_df: pl.DataFrame,
        book_states: pl.DataFrame,
        tokenizer,
        sequence_length: int = 1024,
        depth: int = 10,
        horizon: int = 10,
        threshold: float = 0.002,
        label_type: str = "classification",
        stride: int = 1,
        cache: bool = False,
        time_encoder: Optional[TimeEncoder] = None
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
            time_encoder: TimeEncoder for generating time buckets (optional)
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

        # Set up time encoder (use default crypto encoder if none provided)
        self.time_encoder = time_encoder or create_time_encoder_for_crypto()

        # Prepare data
        self._prepare_data()

        # Cache for processed sequences
        self._cache = {} if cache else None

    def _prepare_data(self):
        """Prepare tokenized data."""

        # Tokenize events
        self.tokens = self.tokenizer.tokenize(self.inc_df)

        # Prepare time buckets from timestamps
        self._prepare_time_features()

        # Prepare labels from book states
        self._prepare_labels()

        # Calculate number of sequences
        self.n_sequences = max(0, (len(self.tokens) - self.sequence_length - self.horizon + 1) // self.stride)

    def _prepare_time_features(self):
        """Prepare time-based features from timestamps."""
        # Extract timestamps from incremental data
        timestamp_column = None
        for candidate in ("tst", "ts", "timestamp"):
            if candidate in self.inc_df.columns:
                timestamp_column = candidate
                break

        if timestamp_column is None:
            raise ValueError("Incremental dataframe must contain a timestamp column (expected one of 'tst', 'ts', 'timestamp').")

        event_timestamps = self.inc_df[timestamp_column].to_numpy()

        # Filter timestamps to match the number of tokens
        if len(event_timestamps) != len(self.tokens):
            event_timestamps = event_timestamps[:len(self.tokens)]

        # Generate time buckets
        self.time_buckets = self.time_encoder.timestamps_to_time_buckets(event_timestamps)

    def _prepare_labels(self):
        """Prepare labels based on mid-price movement."""

        # Calculate mid prices
        bid_prices = self.book_states['bids[0].price'].to_numpy()
        ask_prices = self.book_states['asks[0].price'].to_numpy()

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
        return self.n_sequences

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        if self._cache is not None and idx in self._cache:
            return self._cache[idx]

        start_idx = idx * self.stride
        end_idx = start_idx + self.sequence_length

        token_slice = self.tokens[start_idx:end_idx]
        input_ids = torch.tensor(token_slice, dtype=torch.long)

        time_slice = self.time_buckets[start_idx:end_idx]
        time_buckets = torch.tensor(time_slice, dtype=torch.long)

        if idx < len(self.labels):
            if self.label_type == "classification":
                label = torch.tensor(self.labels[idx], dtype=torch.long)
            else:
                label = torch.tensor(self.labels[idx], dtype=torch.float32)
        else:
            label = torch.tensor(1 if self.label_type == "classification" else 0.0,
                                 dtype=torch.long if self.label_type == "classification" else torch.float32)

        item = {
            "input_ids": input_ids,
            "time_buckets": time_buckets,
            "labels": label
        }

        if self._cache is not None:
            self._cache[idx] = item

        return item

    def get_stats(self) -> Dict[str, Any]:
        stats = {
            "n_sequences": self.n_sequences,
            "sequence_length": self.sequence_length,
            "horizon": self.horizon,
            "threshold": self.threshold,
            "label_type": self.label_type
        }

        if len(self.labels) > 0:
            if self.label_type == "classification":
                counts = np.bincount(self.labels.astype(int), minlength=3)
                stats["label_distribution"] = {
                    "down": int(counts[0]),
                    "stationary": int(counts[1]),
                    "up": int(counts[2])
                }
            else:
                stats["label_mean"] = float(np.mean(self.labels))
                stats["label_std"] = float(np.std(self.labels))
        return stats


def create_lob_pytorch_dataloaders(
    dataset: TokenBookDataset,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 0,
    pin_memory: bool = True
) -> DataLoader:
    """
    Create a PyTorch DataLoader for the tokenized LOB dataset.
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=lambda batch: {
            "input_ids": torch.stack([item["input_ids"] for item in batch]),
            "time_buckets": torch.stack([item["time_buckets"] for item in batch]),
            "labels": torch.stack([item["labels"] for item in batch]),
        }
    )


class LOBPyTorchDataset(TokenBookDataset):
    """
    Backwards-compatible alias for TokenBookDataset.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class MessageVolumeDataset(Dataset):
    """Sequence dataset combining message features and volume images."""

    def __init__(
        self,
        features: pl.DataFrame,
        *,
        sequence_length: int = 128,
        stride: int = 1,
        timestamp_column: str = "tst",
        target_column: str = "event_code",
        categorical_columns: Optional[Sequence[str]] = None,
        numeric_columns: Optional[Sequence[str]] = None,
        volume_columns: Optional[Sequence[str]] = None,
    ) -> None:
        if features.is_empty():
            raise ValueError("Features dataframe must be non-empty.")

        self.sequence_length = sequence_length
        self.stride = stride
        self.timestamp_column = timestamp_column
        self.target_column = target_column

        available_cols = set(features.columns)
        if timestamp_column not in available_cols or target_column not in available_cols:
            raise ValueError("Features must contain timestamp and target columns.")

        if categorical_columns is None:
            categorical_columns = [col for col in ("event_code", "side") if col in available_cols]
        if numeric_columns is None:
            numeric_columns = [col for col in ("price_offset_ticks", "size", "inter_arrival_ns") if col in available_cols]
        if volume_columns is None:
            volume_columns = [col for col in features.columns if col.startswith("ask_volume_") or col.startswith("bid_volume_") or col == "mid_price"]

        self.categorical_columns = list(categorical_columns)
        self.numeric_columns = list(numeric_columns)
        self.volume_columns = list(volume_columns)

        self.categorical_array = features.select(self.categorical_columns).to_numpy() if self.categorical_columns else None
        self.numeric_array = features.select(self.numeric_columns).to_numpy() if self.numeric_columns else None
        self.volume_array = features.select(self.volume_columns).to_numpy() if self.volume_columns else None

        self.targets = features.select(target_column).to_numpy().reshape(-1)
        self.timestamps = features.select(timestamp_column).to_numpy().reshape(-1)

        total = len(self.targets)
        self.n_sequences = max(0, (total - sequence_length))
        if self.n_sequences <= 0:
            raise ValueError("Not enough rows to create sequences with the specified length.")

    def __len__(self) -> int:
        return self.n_sequences

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        start = idx * self.stride
        end = start + self.sequence_length
        target_idx = end

        if target_idx >= len(self.targets):
            raise IndexError("Index exceeds dataset bounds")

        item: Dict[str, torch.Tensor] = {}

        if self.categorical_array is not None:
            cat_slice = self.categorical_array[start:end]
            item["categorical_inputs"] = torch.from_numpy(cat_slice).long()

        if self.numeric_array is not None:
            num_slice = self.numeric_array[start:end]
            item["numeric_inputs"] = torch.from_numpy(num_slice).float()

        if self.volume_array is not None:
            vol_slice = self.volume_array[start:end]
            item["volume_inputs"] = torch.from_numpy(vol_slice).float()

        item["target_event"] = torch.tensor(int(self.targets[target_idx]), dtype=torch.long)
        item["target_timestamp"] = torch.tensor(int(self.timestamps[target_idx]), dtype=torch.long)

        return item
