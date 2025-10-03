"""
Test PyTorch Dataset and DataLoader for LOB data.
"""

import torch
from torch.utils.data import DataLoader
from lobgpt.hdb import get_dataset
from lobgpt.book_state import reconstruct_from_tardis
from lobgpt.pytorch_dataset import LOBPyTorchDataset, create_lob_pytorch_dataloaders
from lobgpt.models.deeplob import deeplob


def test_single_dataset():
    """Test single dataset creation."""

    print("=== Testing Single Dataset ===\n")

    # Load data
    product = "BTCUSDT"
    times = ['250912.000000', '250912.000200']  # 2 minutes
    depth = 10

    print(f"Loading data for {product}...")
    book_states = reconstruct_from_tardis(product, times, depth)
    print(f"Loaded {len(book_states)} book states\n")

    # Test DeepLOB format
    print("1. Testing DeepLOB format:")
    dataset = LOBPyTorchDataset(
        book_states,
        format="deeplob",
        sequence_length=100,
        depth=10,
        horizon=10,
        threshold=0.002,
        normalize=True,
        label_type="classification",
        stride=1
    )

    print(f"   Dataset length: {len(dataset)}")
    print(f"   Dataset stats: {dataset.get_stats()}")

    # Get a sample
    x, y = dataset[0]
    print(f"   Sample shape: {x.shape}")
    print(f"   Label: {y}")

    # Test with DataLoader
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    batch_x, batch_y = next(iter(dataloader))
    print(f"   Batch shape: {batch_x.shape}")
    print(f"   Batch labels shape: {batch_y.shape}\n")

    # Test Raw format
    print("2. Testing Raw format:")
    dataset_raw = LOBPyTorchDataset(
        book_states,
        format="raw",
        sequence_length=50,
        normalize=True
    )

    print(f"   Dataset length: {len(dataset_raw)}")
    x_raw, y_raw = dataset_raw[0]
    print(f"   Sample shape: {x_raw.shape}\n")

    return dataset


def test_train_val_test_split():
    """Test train/val/test split with DataLoaders."""

    print("=== Testing Train/Val/Test Split ===\n")

    # Load data
    product = "BTCUSDT"
    print(f"Loading data for {product}...")

    # Load different time periods for train/val/test
    train_states = reconstruct_from_tardis(product, ['250912.000000', '250912.000300'], 10)
    val_states = reconstruct_from_tardis(product, ['250912.000300', '250912.000400'], 10)
    test_states = reconstruct_from_tardis(product, ['250912.000400', '250912.000500'], 10)

    print(f"Train states: {len(train_states)}")
    print(f"Val states: {len(val_states)}")
    print(f"Test states: {len(test_states)}\n")

    # Create DataLoaders
    dataloaders, scaler = create_lob_pytorch_dataloaders(
        train_states=train_states,
        val_states=val_states,
        test_states=test_states,
        batch_size=32,
        format="deeplob",
        sequence_length=100,
        depth=10,
        horizon=10,
        threshold=0.002,
        normalize=True,
        label_type="classification",
        stride=1,
        num_workers=0,
        shuffle_train=True,
        pin_memory=False
    )

    print("\nDataLoader sizes:")
    for name, loader in dataloaders.items():
        print(f"  {name}: {len(loader)} batches")

    # Test iteration
    print("\nTesting iteration:")
    for name, loader in dataloaders.items():
        batch_x, batch_y = next(iter(loader))
        print(f"  {name} batch: X={batch_x.shape}, y={batch_y.shape}")

    return dataloaders


def test_with_model():
    """Test dataset with DeepLOB model."""

    print("=== Testing with DeepLOB Model ===\n")

    # Load data
    product = "BTCUSDT"
    times = ['250912.000000', '250912.000200']
    book_states = reconstruct_from_tardis(product, times, 10)

    # Create dataset
    dataset = LOBPyTorchDataset(
        book_states,
        format="deeplob",
        sequence_length=100,
        depth=10,
        normalize=True
    )

    # Create DataLoader
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    # Create model
    model = deeplob(y_len=3)
    model.eval()

    # Test forward pass
    with torch.no_grad():
        for i, (batch_x, batch_y) in enumerate(dataloader):
            if i >= 3:  # Test first 3 batches
                break

            # Forward pass
            outputs = model(batch_x)

            # Calculate accuracy
            _, predicted = torch.max(outputs, 1)
            correct = (predicted == batch_y).sum().item()
            accuracy = correct / len(batch_y) * 100

            print(f"Batch {i+1}:")
            print(f"  Input shape: {batch_x.shape}")
            print(f"  Output shape: {outputs.shape}")
            print(f"  Accuracy: {accuracy:.1f}%")

            # Show prediction distribution
            pred_counts = torch.bincount(predicted, minlength=3)
            print(f"  Predictions: Down={pred_counts[0]}, Stat={pred_counts[1]}, Up={pred_counts[2]}")


def test_different_configurations():
    """Test different dataset configurations."""

    print("=== Testing Different Configurations ===\n")

    # Load data
    product = "BTCUSDT"
    times = ['250912.000000', '250912.000100']
    book_states = reconstruct_from_tardis(product, times, 10)

    configs = [
        {"format": "deeplob", "sequence_length": 50, "horizon": 5},
        {"format": "deeplob", "sequence_length": 100, "horizon": 10},
        {"format": "deeplob", "sequence_length": 200, "horizon": 20},
        {"format": "raw", "sequence_length": 50, "horizon": 5},
    ]

    for i, config in enumerate(configs, 1):
        print(f"Config {i}: {config}")

        try:
            dataset = LOBPyTorchDataset(
                book_states,
                **config,
                depth=10,
                normalize=True,
                label_type="classification"
            )

            if len(dataset) > 0:
                x, y = dataset[0]
                stats = dataset.get_stats()

                print(f"  Dataset length: {len(dataset)}")
                print(f"  Sample shape: {x.shape}")

                if "label_distribution" in stats:
                    dist = stats["label_distribution"]
                    print(f"  Label distribution: {dist}")
            else:
                print(f"  No valid sequences for this configuration")

        except Exception as e:
            print(f"  Error: {e}")

        print()


def main():
    """Run all tests."""

    print("=" * 60)
    print("PyTorch Dataset Test Suite for LOB Data")
    print("=" * 60 + "\n")

    # Test 1: Single dataset
    dataset = test_single_dataset()

    print("\n" + "=" * 60 + "\n")

    # Test 2: Train/val/test split
    dataloaders = test_train_val_test_split()

    print("\n" + "=" * 60 + "\n")

    # Test 3: With model
    test_with_model()

    print("\n" + "=" * 60 + "\n")

    # Test 4: Different configurations
    test_different_configurations()

    print("=" * 60)
    print("✅ All tests completed successfully!")
    print("=" * 60)

    return dataset, dataloaders


if __name__ == "__main__":
    dataset, dataloaders = main()