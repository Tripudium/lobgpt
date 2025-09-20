"""
Hugging Face datasets integration for TLOB-style limit order book data.

This module provides integration with Hugging Face datasets for better dataset
management, versioning, and sharing capabilities. It converts our Polars-based
TLOB data into HF datasets format with proper train/val/test splits.
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
import polars as pl
import torch
from datasets import Dataset, DatasetDict, Features, Sequence, Value
from torch.utils.data import DataLoader


class LOBHuggingFaceDataset:
    """
    Hugging Face Dataset wrapper for TLOB limit order book data.

    Provides dataset management, splitting, and PyTorch integration
    using HuggingFace datasets infrastructure.
    """

    def __init__(
        self,
        df: pl.DataFrame,
        feature_cols: List[str],
        target_col: str,
        seq_size: int,
        stride: int = 1,
        split_ratios: Tuple[float, float, float] = (0.8, 0.1, 0.1),
        normalize: bool = True,
        dataset_name: Optional[str] = None
    ):
        """
        Initialize HuggingFace dataset from Polars DataFrame.

        Args:
            df: Polars DataFrame with features and labels
            feature_cols: List of feature column names
            target_col: Label column name
            seq_size: Sequence length for sliding windows
            stride: Step size between sequences
            split_ratios: (train, val, test) split ratios
            normalize: Whether to apply z-score normalization
            dataset_name: Optional name for the dataset
        """
        self.df = df
        self.feature_cols = feature_cols
        self.target_col = target_col
        self.seq_size = seq_size
        self.stride = stride
        self.split_ratios = split_ratios
        self.normalize = normalize
        self.dataset_name = dataset_name or "lob_dataset"

        # Will be set during creation
        self.dataset_dict = None
        self.normalization_stats = None

    def _create_sequences(self, df: pl.DataFrame, stride: int) -> Dict:
        """
        Create sequences from DataFrame.

        Args:
            df: Input DataFrame
            stride: Stride for sequence creation

        Returns:
            Dictionary with sequences and labels
        """
        features_df = df.select(self.feature_cols)
        labels = df[self.target_col].to_numpy()

        # Convert features to numpy
        features = features_df.to_numpy()

        # Create sequences
        sequences = []
        sequence_labels = []
        timestamps = []

        for i in range(0, len(df) - self.seq_size + 1, stride):
            # Skip if label is null
            if np.isnan(labels[i + self.seq_size - 1]):
                continue

            sequences.append(features[i:i + self.seq_size])
            sequence_labels.append(labels[i + self.seq_size - 1])

            # Add timestamp if available
            if 'ts' in df.columns:
                timestamps.append(df['ts'][i + self.seq_size - 1])

        result = {
            'sequences': sequences,
            'labels': sequence_labels
        }

        if timestamps:
            result['timestamps'] = timestamps

        return result

    def _normalize_features(self, train_data: Dict, val_data: Dict, test_data: Dict) -> Tuple[Dict, Dict, Dict]:
        """
        Normalize features using training set statistics.

        Args:
            train_data: Training data dictionary
            val_data: Validation data dictionary
            test_data: Test data dictionary

        Returns:
            Normalized data dictionaries
        """
        if not self.normalize:
            return train_data, val_data, test_data

        # Calculate stats from training set
        train_sequences = np.array(train_data['sequences'])
        mean = np.mean(train_sequences, axis=(0, 1), keepdims=True)
        std = np.std(train_sequences, axis=(0, 1), keepdims=True)
        std = np.where(std < 1e-8, 1.0, std)  # Avoid division by zero

        self.normalization_stats = {'mean': mean, 'std': std}

        # Apply normalization
        train_data['sequences'] = ((train_sequences - mean) / std).tolist()
        val_data['sequences'] = ((np.array(val_data['sequences']) - mean) / std).tolist()
        test_data['sequences'] = ((np.array(test_data['sequences']) - mean) / std).tolist()

        return train_data, val_data, test_data

    def create_dataset(self) -> DatasetDict:
        """
        Create HuggingFace DatasetDict with train/val/test splits.

        Returns:
            DatasetDict with train/val/test splits
        """
        # Split data temporally
        total_len = len(self.df)
        train_end = int(total_len * self.split_ratios[0])
        val_end = int(total_len * (self.split_ratios[0] + self.split_ratios[1]))

        train_df = self.df[:train_end]
        val_df = self.df[train_end:val_end]
        test_df = self.df[val_end:]

        # Create sequences for each split
        train_data = self._create_sequences(train_df, self.stride)
        val_data = self._create_sequences(val_df, max(1, self.stride * 10))  # Sparser sampling
        test_data = self._create_sequences(test_df, max(1, self.stride * 10))

        # Apply normalization
        train_data, val_data, test_data = self._normalize_features(train_data, val_data, test_data)

        # Define features schema
        features = Features({
            'sequences': Sequence(Sequence(Value('float32'))),
            'labels': Value('int64')
        })

        # Add timestamp feature if available
        if 'timestamps' in train_data:
            features['timestamps'] = Value('int64')

        # Create datasets
        train_dataset = Dataset.from_dict(train_data, features=features)
        val_dataset = Dataset.from_dict(val_data, features=features)
        test_dataset = Dataset.from_dict(test_data, features=features)

        # Create DatasetDict
        self.dataset_dict = DatasetDict({
            'train': train_dataset,
            'validation': val_dataset,
            'test': test_dataset
        })

        # Add metadata
        self.dataset_dict['train'] = self.dataset_dict['train'].add_column(
            'split', ['train'] * len(self.dataset_dict['train'])
        )
        self.dataset_dict['validation'] = self.dataset_dict['validation'].add_column(
            'split', ['validation'] * len(self.dataset_dict['validation'])
        )
        self.dataset_dict['test'] = self.dataset_dict['test'].add_column(
            'split', ['test'] * len(self.dataset_dict['test'])
        )

        # Print statistics
        print(f"\n🤗 HuggingFace Dataset '{self.dataset_name}' created:")
        print(f"  Train: {len(self.dataset_dict['train'])} samples")
        print(f"  Validation: {len(self.dataset_dict['validation'])} samples")
        print(f"  Test: {len(self.dataset_dict['test'])} samples")

        # Print label distribution
        train_labels = self.dataset_dict['train']['labels']
        unique, counts = np.unique(train_labels, return_counts=True)
        label_names = ["Up", "Stationary", "Down"]
        print("\n  Training label distribution:")
        for label, count in zip(unique, counts):
            pct = (count / len(train_labels)) * 100
            print(f"    {label_names[int(label)]}: {count} ({pct:.1f}%)")

        return self.dataset_dict

    def save_to_disk(self, path: str):
        """Save dataset to disk."""
        if self.dataset_dict is None:
            raise ValueError("Dataset not created yet. Call create_dataset() first.")
        self.dataset_dict.save_to_disk(path)
        print(f"Dataset saved to {path}")

    def load_from_disk(self, path: str):
        """Load dataset from disk."""
        self.dataset_dict = DatasetDict.load_from_disk(path)
        print(f"Dataset loaded from {path}")

    def push_to_hub(self, repo_id: str, token: Optional[str] = None):
        """Push dataset to Hugging Face Hub."""
        if self.dataset_dict is None:
            raise ValueError("Dataset not created yet. Call create_dataset() first.")
        self.dataset_dict.push_to_hub(repo_id, token=token)
        print(f"Dataset pushed to Hub: {repo_id}")


class HFLOBDataLoader:
    """
    PyTorch DataLoader wrapper for HuggingFace LOB datasets.
    """

    def __init__(self, hf_dataset: LOBHuggingFaceDataset):
        """
        Initialize with HuggingFace dataset.

        Args:
            hf_dataset: LOBHuggingFaceDataset instance
        """
        self.hf_dataset = hf_dataset
        if hf_dataset.dataset_dict is None:
            raise ValueError("HuggingFace dataset not created. Call create_dataset() first.")

    def get_dataloader(
        self,
        split: str,
        batch_size: int = 32,
        shuffle: Optional[bool] = None,
        num_workers: int = 4,
        pin_memory: bool = True
    ) -> DataLoader:
        """
        Get PyTorch DataLoader for a specific split.

        Args:
            split: 'train', 'validation', or 'test'
            batch_size: Batch size
            shuffle: Whether to shuffle (default: True for train, False for others)
            num_workers: Number of workers
            pin_memory: Pin memory for GPU

        Returns:
            PyTorch DataLoader
        """
        if shuffle is None:
            shuffle = (split == 'train')

        dataset = self.hf_dataset.dataset_dict[split]

        # Convert to PyTorch tensors
        def collate_fn(batch):
            sequences = torch.FloatTensor([item['sequences'] for item in batch])
            labels = torch.LongTensor([item['labels'] for item in batch])
            return sequences, labels

        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
            collate_fn=collate_fn,
            drop_last=(split == 'train')  # Drop last for training only
        )

    def get_all_dataloaders(
        self,
        batch_size: int = 32,
        test_batch_size: Optional[int] = None,
        **kwargs
    ) -> Dict[str, DataLoader]:
        """
        Get all DataLoaders at once.

        Args:
            batch_size: Training/validation batch size
            test_batch_size: Test batch size (default: batch_size * 4)
            **kwargs: Additional arguments for DataLoader

        Returns:
            Dictionary with 'train', 'validation', 'test' DataLoaders
        """
        test_batch_size = test_batch_size or (batch_size * 4)

        return {
            'train': self.get_dataloader('train', batch_size, **kwargs),
            'validation': self.get_dataloader('validation', batch_size, **kwargs),
            'test': self.get_dataloader('test', test_batch_size, **kwargs)
        }


def create_lob_hf_dataset(
    df: pl.DataFrame,
    target_horizon: int = 10,
    seq_size: int = 100,
    depth: int = 10,
    **kwargs
) -> LOBHuggingFaceDataset:
    """
    Convenience function to create HuggingFace dataset with standard LOB features.

    Args:
        df: DataFrame with TLOB labels already applied
        target_horizon: Which horizon's labels to use as target
        seq_size: Sequence length
        depth: LOB depth (number of levels)
        **kwargs: Additional arguments for LOBHuggingFaceDataset

    Returns:
        Configured LOBHuggingFaceDataset
    """
    # Define feature columns
    feature_cols = []
    for i in range(depth):
        feature_cols.extend([
            f"asks[{i}].price", f"asks[{i}].amount",
            f"bids[{i}].price", f"bids[{i}].amount"
        ])

    # Add derived features if they exist
    optional_features = ["spread", "weighted_mid_price", "total_volume_best"]
    for feat in optional_features:
        if feat in df.columns:
            feature_cols.append(feat)

    # Add volume imbalance features if they exist
    for i in range(depth):
        feat = f"volume_imbalance_{i}"
        if feat in df.columns:
            feature_cols.append(feat)

    target_col = f"label_h{target_horizon}"

    # Verify target column exists
    if target_col not in df.columns:
        raise ValueError(f"Target column {target_col} not found. Run add_tlob_labels first.")

    return LOBHuggingFaceDataset(
        df=df,
        feature_cols=feature_cols,
        target_col=target_col,
        seq_size=seq_size,
        dataset_name=f"lob_btc_h{target_horizon}_seq{seq_size}",
        **kwargs
    )