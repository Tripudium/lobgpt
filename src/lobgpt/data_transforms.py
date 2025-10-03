"""
Data transformation utilities for different model formats.
"""

from typing import Optional, Tuple

import numpy as np
import pandas as pd
import polars as pl
import torch
from sklearn.preprocessing import StandardScaler


def book_states_to_deeplob_format(
    book_states: pl.DataFrame,
    sequence_length: int = 100,
    depth: int = 10,
    normalize: bool = True
) -> Tuple[torch.Tensor, Optional[StandardScaler]]:
    """
    Convert reconstructed book states to DeepLOB input format.

    Args:
        book_states: DataFrame from reconstruct_book_states()
        sequence_length: Length of sequences (default 100)
        depth: Number of price levels (default 10)
        normalize: Whether to apply z-score normalization

    Returns:
        X: Tensor of shape (n_sequences, 1, sequence_length, 40)
        scaler: StandardScaler object if normalize=True, None otherwise
    """

    # Extract features in DeepLOB order: ask_price, ask_vol, bid_price, bid_vol for each level
    features = []

    for level in range(depth):
        # Ask price and volume
        ask_price_col = f'asks[{level}].price'
        ask_amount_col = f'asks[{level}].amount'

        # Bid price and volume
        bid_price_col = f'bids[{level}].price'
        bid_amount_col = f'bids[{level}].amount'

        features.extend([ask_price_col, ask_amount_col, bid_price_col, bid_amount_col])

    # Convert to numpy array
    data = book_states.select(features).to_numpy()

    # Handle infinite values
    data = np.where(np.isinf(data), np.nan, data)

    # Forward fill NaN values
    df_temp = pd.DataFrame(data)
    df_temp = df_temp.fillna(method='ffill').fillna(method='bfill')
    data = df_temp.values

    # Normalize if requested
    scaler = None
    if normalize:
        scaler = StandardScaler()
        # Reshape for scaling: (n_samples * n_features,)
        data_reshaped = data.reshape(-1, data.shape[-1])
        data_reshaped = scaler.fit_transform(data_reshaped)
        data = data_reshaped.reshape(data.shape)

    # Create sequences
    sequences = []
    for i in range(len(data) - sequence_length + 1):
        sequences.append(data[i:i + sequence_length])

    # Convert to tensor: (n_sequences, sequence_length, n_features)
    X = torch.tensor(sequences, dtype=torch.float32)

    # Reshape to DeepLOB format: (n_sequences, 1, sequence_length, n_features)
    X = X.unsqueeze(1)  # Add channel dimension

    return X, scaler


def create_deeplob_labels(
    book_states: pl.DataFrame,
    sequence_length: int = 100,
    horizon: int = 10,
    threshold: float = 0.002,
    label_type: str = "classification"
) -> torch.Tensor:
    """
    Create labels for DeepLOB prediction tasks.

    Args:
        book_states: DataFrame from reconstruct_book_states()
        sequence_length: Length of input sequences
        horizon: Prediction horizon (number of steps ahead)
        threshold: Price movement threshold for classification
        label_type: "classification" or "regression"

    Returns:
        y: Labels tensor
    """

    # Calculate mid prices
    mid_prices = (
        book_states['bids[0].price'].to_numpy() +
        book_states['asks[0].price'].to_numpy()
    ) / 2

    # Handle infinite values
    mid_prices = np.where(np.isinf(mid_prices), np.nan, mid_prices)
    mid_prices = pd.Series(mid_prices).fillna(method='ffill').fillna(method='bfill').values

    labels = []

    for i in range(len(mid_prices) - sequence_length - horizon + 1):
        current_mid = mid_prices[i + sequence_length - 1]
        future_mid = mid_prices[i + sequence_length + horizon - 1]

        if label_type == "classification":
            # 3-class classification: 0=down, 1=stationary, 2=up
            price_change = (future_mid - current_mid) / current_mid

            if price_change < -threshold:
                label = 0  # Down
            elif price_change > threshold:
                label = 2  # Up
            else:
                label = 1  # Stationary

        elif label_type == "regression":
            # Continuous price change
            label = (future_mid - current_mid) / current_mid

        labels.append(label)

    return torch.tensor(labels, dtype=torch.long if label_type == "classification" else torch.float32)


def create_deeplob_dataset(
    book_states: pl.DataFrame,
    sequence_length: int = 100,
    depth: int = 10,
    horizon: int = 10,
    threshold: float = 0.002,
    normalize: bool = True,
    label_type: str = "classification"
) -> Tuple[torch.Tensor, torch.Tensor, Optional[StandardScaler]]:
    """
    Create complete dataset for DeepLOB training.

    Returns:
        X: Input features (n_sequences, 1, sequence_length, 40)
        y: Labels
        scaler: StandardScaler if normalize=True
    """

    # Create features
    X, scaler = book_states_to_deeplob_format(
        book_states, sequence_length, depth, normalize
    )

    # Create labels
    y = create_deeplob_labels(
        book_states, sequence_length, horizon, threshold, label_type
    )

    # Ensure X and y have same number of samples
    min_samples = min(len(X), len(y))
    X = X[:min_samples]
    y = y[:min_samples]

    return X, y, scaler


def describe_deeplob_format():
    """Print description of DeepLOB input format."""

    print("=== DeepLOB Input Format ===")
    print("Shape: (batch_size, 1, 100, 40)")
    print("- batch_size: Number of sequences")
    print("- 1: Channel dimension (like grayscale image)")
    print("- 100: Time steps (sequence length)")
    print("- 40: Features per time step")
    print()
    print("Feature Layout (40 features per timestep):")
    print("For 10 levels of market depth:")
    for level in range(1, 11):
        start_idx = (level - 1) * 4
        print(f"  Level {level:2d}: features {start_idx:2d}-{start_idx+3:2d} = "
              f"ask_price_{level}, ask_vol_{level}, bid_price_{level}, bid_vol_{level}")
    print()
    print("Preprocessing:")
    print("- Z-score normalization (mean=0, std=1)")
    print("- Handle infinite values with forward/backward fill")
    print("- Convert to float32 tensors")
    print()
    print("Labels:")
    print("- Classification: 0=down, 1=stationary, 2=up")
    print("- Based on mid-price movement beyond threshold")
    print("- Prediction horizon: configurable (default 10 steps)")


if __name__ == "__main__":
    describe_deeplob_format()