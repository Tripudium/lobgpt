"""
Test data formatting for DeepLOB model.
"""

import torch

from lobgpt.book_state import reconstruct_from_tardis
from lobgpt.data_transforms import create_deeplob_dataset, describe_deeplob_format
from lobgpt.models.deeplob import deeplob


def main():
    """Test DeepLOB data formatting and model."""

    print("=== DeepLOB Data Format Test ===\n")

    # Show format description
    describe_deeplob_format()

    # Load data
    product = "BTCUSDT"
    times = ['250912.000000', '250912.000200']  # 2 minutes for more data
    depth = 10

    print(f"\nLoading data for {product} from {times[0]} to {times[1]}...")

    # Reconstruct book states
    book_states = reconstruct_from_tardis(product, times, depth)
    print(f"Reconstructed {len(book_states)} book states")

    # Create DeepLOB dataset
    print("\nCreating DeepLOB dataset...")
    X, y, scaler = create_deeplob_dataset(
        book_states,
        sequence_length=100,
        depth=10,
        horizon=10,
        threshold=0.002,
        normalize=True,
        label_type="classification"
    )

    print(f"✓ Features shape: {X.shape}")
    print(f"✓ Labels shape: {y.shape}")
    print(f"✓ Data type: {X.dtype}")
    print(f"✓ Labels type: {y.dtype}")

    # Check data statistics
    print("\nData Statistics:")
    print(f"✓ Features mean: {X.mean():.6f}")
    print(f"✓ Features std: {X.std():.6f}")
    print(f"✓ Features min: {X.min():.6f}")
    print(f"✓ Features max: {X.max():.6f}")

    # Check label distribution
    unique_labels, counts = torch.unique(y, return_counts=True)
    print("\nLabel Distribution:")
    total = len(y)
    for label, count in zip(unique_labels, counts):
        label_name = ["Down", "Stationary", "Up"][label.item()]
        pct = count.item() / total * 100
        print(f"✓ {label_name:10s}: {count:5d} ({pct:5.1f}%)")

    # Test model compatibility
    print("\n=== Model Compatibility Test ===")

    # Create model
    model = deeplob(y_len=3)  # 3-class classification
    print(f"✓ Model created: {model.__class__.__name__}")

    # Test forward pass
    batch_size = 32
    sample_batch = X[:batch_size]
    sample_labels = y[:batch_size]

    print(f"✓ Sample batch shape: {sample_batch.shape}")

    # Forward pass
    model.eval()
    with torch.no_grad():
        predictions = model(sample_batch)

    print(f"✓ Predictions shape: {predictions.shape}")
    print(f"✓ Predictions sum (should be ~1): {predictions.sum(dim=1)[:5]}")

    # Test loss calculation
    criterion = torch.nn.CrossEntropyLoss()
    loss = criterion(predictions, sample_labels)
    print(f"✓ Loss: {loss.item():.6f}")

    # Test training step
    print("\n=== Training Step Test ===")
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # One training step
    optimizer.zero_grad()
    outputs = model(sample_batch)
    loss = criterion(outputs, sample_labels)
    loss.backward()
    optimizer.step()

    print("✓ Training step completed")
    print(f"✓ Loss after step: {loss.item():.6f}")

    # Calculate accuracy
    _, predicted = torch.max(outputs.data, 1)
    correct = (predicted == sample_labels).sum().item()
    accuracy = correct / batch_size * 100
    print(f"✓ Batch accuracy: {accuracy:.1f}%")

    # Show sample predictions
    print("\n=== Sample Predictions ===")
    print("True Label | Predicted | Confidence")
    print("-" * 35)
    for i in range(min(10, len(sample_labels))):
        true_label = sample_labels[i].item()
        pred_label = predicted[i].item()
        confidence = outputs[i][pred_label].item()

        true_name = ["Down", "Stat", "Up"][true_label]
        pred_name = ["Down", "Stat", "Up"][pred_label]

        print(f"{true_name:9s} | {pred_name:9s} | {confidence:.3f}")

    print("\n=== Summary ===")
    print("✓ Successfully converted book states to DeepLOB format")
    print(f"✓ Created {len(X)} sequences of length 100")
    print("✓ Model accepts the data format correctly")
    print("✓ Training loop works as expected")
    print("✓ Ready for DeepLOB training!")

    return X, y, model


if __name__ == "__main__":
    X, y, model = main()