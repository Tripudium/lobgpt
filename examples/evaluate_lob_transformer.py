"""
Comprehensive evaluation example for LOB Transformer.
"""

import torch
import numpy as np
from pathlib import Path

from lobgpt.hdb import get_dataset
from lobgpt.book_state import reconstruct_from_tardis
from lobgpt.tokenizer import LOBTokenizer
from lobgpt.pytorch_dataset import TokenBookDataset
from lobgpt.models.lob_transformer import LOBTransformerConfig, create_lob_transformer_small
from lobgpt.training.lob_trainer import LOBTransformerLightning
from lobgpt.evaluation.metrics import LOBEvaluator, evaluate_model
from lobgpt.evaluation.visualization import LOBTransformerVisualizer, plot_loss_landscape
from lobgpt.inference.generator import LOBInference, GenerationConfig


def create_evaluation_datasets():
    """Create datasets for evaluation."""
    print("Loading evaluation data...")

    # Load data
    dl = get_dataset("tardis")
    product = "BTCUSDT"

    # Test data (different from training)
    test_times = ['250912.140000', '250912.150000']  # 1 hour for testing
    test_book_states = reconstruct_from_tardis(product, test_times, depth=10)
    test_inc_df = dl.load_inc(product, test_times)

    print(f"Test book states: {len(test_book_states)}")
    print(f"Test inc events: {len(test_inc_df)}")

    # Create tokenizer (should match training)
    tokenizer = LOBTokenizer(tick_size=0.1, ref_size=1.0)

    # Create test dataset
    test_dataset = TokenBookDataset(
        inc_df=test_inc_df,
        book_states=test_book_states,
        tokenizer=tokenizer,
        sequence_length=200,
        horizon=10,
        threshold=0.002,
        stride=100  # Sparse sampling for evaluation
    )

    print(f"Test dataset: {test_dataset.get_stats()}")

    return test_dataset, tokenizer


def load_trained_model(checkpoint_path: str, tokenizer: LOBTokenizer):
    """Load a trained model from checkpoint."""
    print(f"Loading model from {checkpoint_path}")

    # Create model configuration (should match training)
    config = LOBTransformerConfig(
        vocab_size=len(tokenizer.vocab),
        d_model=512,
        n_layers=6,
        n_heads=8,
        d_ff=2048,
        max_seq_length=1024,
        dropout=0.1,
        use_learned_pos=True,
        use_time_encoding=True,
        tie_embeddings=True
    )

    # Load from checkpoint
    model = LOBTransformerLightning.load_from_checkpoint(
        checkpoint_path,
        config=config,
        strict=False  # Allow some parameter mismatches
    )

    return model.model, config


def evaluate_model_comprehensive(model, test_dataset, tokenizer, config, device="cuda"):
    """Perform comprehensive model evaluation."""
    print("\n" + "="*60)
    print("COMPREHENSIVE MODEL EVALUATION")
    print("="*60)

    model.to(device)
    model.eval()

    # Create data loader
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=8,
        shuffle=False,
        num_workers=2,
        collate_fn=lambda batch: {
            "input_ids": torch.stack([item[0] for item in batch])
        }
    )

    # 1. Basic Metrics Evaluation
    print("\n1. Computing evaluation metrics...")
    evaluator = LOBEvaluator(tokenizer)

    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            if batch_idx >= 50:  # Limit for faster evaluation
                break

            input_ids = batch["input_ids"].to(device)

            if input_ids.size(1) <= 1:
                continue

            inputs = input_ids[:, :-1]
            targets = input_ids[:, 1:]

            # Forward pass
            logits, _ = model(inputs)
            predictions = torch.argmax(logits, dim=-1)

            # Calculate loss
            loss = torch.nn.functional.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                targets.reshape(-1),
                ignore_index=-100
            )

            evaluator.update_batch(predictions, targets, loss)

    # Get final metrics
    metrics = evaluator.compute_metrics()
    print(f"\nEvaluation Results:")
    print(f"Token Accuracy: {metrics.token_accuracy:.4f}")
    print(f"Perplexity: {metrics.token_perplexity:.4f}")
    print(f"Event Type Accuracy: {metrics.event_type_accuracy:.4f}")
    print(f"Side Accuracy: {metrics.side_accuracy:.4f}")
    print(f"Direction Accuracy: {metrics.direction_accuracy:.4f}")

    # Generate detailed report
    report = evaluator.generate_report()
    print(report)

    # 2. Visualization Analysis
    print("\n2. Creating visualizations...")
    visualizer = LOBTransformerVisualizer(model, tokenizer)

    # Model complexity analysis
    visualizer.plot_model_complexity_analysis(save_path="./plots/model_complexity.png")

    # Token embeddings visualization
    visualizer.plot_token_embeddings(method="pca", save_path="./plots/token_embeddings_pca.png")

    # Attention patterns for a sample sequence
    sample_batch = next(iter(test_loader))
    sample_input = sample_batch["input_ids"][0][:50]  # First 50 tokens

    visualizer.plot_attention_patterns(
        sample_input,
        layer_idx=-1,  # Last layer
        head_idx=0,
        save_path="./plots/attention_patterns.png"
    )

    # 3. Generation Analysis
    print("\n3. Analyzing generation capabilities...")
    inference = LOBInference(model, tokenizer)

    # Generate sample sequences
    prompt_length = 20
    sample_prompt = sample_input[:prompt_length]

    # Generate with different strategies
    config_sampling = GenerationConfig(
        max_length=30,
        temperature=0.8,
        top_k=50,
        do_sample=True,
        return_probabilities=True
    )

    generated_output = model.generate(
        sample_prompt.unsqueeze(0).to(device),
        max_length=30,
        temperature=0.8,
        top_k=50,
        do_sample=True
    )

    # Analyze generation
    visualizer.plot_generation_analysis(
        sample_prompt,
        generated_output[0],
        save_path="./plots/generation_analysis.png"
    )

    # 4. Inference Testing
    print("\n4. Testing high-level inference...")

    # Create some sample events for testing
    try:
        # Decode first few tokens to get sample events
        sample_events = []
        for token_id in sample_prompt[:5]:
            event = tokenizer.decode_token(token_id.item())
            if event:
                sample_events.append(event)

        if sample_events:
            # Test next event prediction
            predicted_events = inference.predict_next_events(
                sample_events,
                n_predictions=5,
                strategy="sampling"
            )

            print(f"\nSample input events: {len(sample_events)}")
            print(f"Predicted next events: {len(predicted_events)}")

            # Print first few predictions
            for i, event in enumerate(predicted_events[:3]):
                print(f"  {i+1}. {event.to_string()}")

            # Test probability estimation
            if len(predicted_events) >= 2:
                event_probs = inference.estimate_probabilities(
                    sample_events,
                    predicted_events[:2]
                )

                print(f"\nEvent probabilities:")
                for event, prob in event_probs.items():
                    print(f"  {event.event_type.name:<6s}: {prob:.4f}")

    except Exception as e:
        print(f"Warning: Could not test inference capabilities: {e}")

    # 5. Confusion Matrices
    print("\n5. Generating confusion matrices...")
    evaluator.plot_confusion_matrices(save_path="./plots/confusion_matrices.png")

    # 6. Loss Landscape Analysis (optional, computationally expensive)
    print("\n6. Loss landscape analysis...")
    try:
        # Use a smaller sample for loss landscape
        small_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=4,
            shuffle=False,
            num_workers=1
        )

        plot_loss_landscape(
            model,
            small_loader,
            device=device,
            n_samples=20,
            save_path="./plots/loss_landscape.png"
        )
    except Exception as e:
        print(f"Warning: Could not generate loss landscape: {e}")

    print("\n" + "="*60)
    print("EVALUATION COMPLETE")
    print("="*60)
    print(f"Plots saved to ./plots/")
    print(f"Key metrics:")
    print(f"  - Token Accuracy: {metrics.token_accuracy:.4f}")
    print(f"  - Perplexity: {metrics.token_perplexity:.4f}")
    print(f"  - Event Type Accuracy: {metrics.event_type_accuracy:.4f}")

    return metrics


def main():
    """Main evaluation function."""
    # Create output directory
    Path("./plots").mkdir(exist_ok=True)

    # Check for CUDA
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Create evaluation dataset
    test_dataset, tokenizer = create_evaluation_datasets()

    # Option 1: Load from checkpoint (if available)
    checkpoint_path = "./checkpoints/last.ckpt"  # Adjust path as needed

    if Path(checkpoint_path).exists():
        print(f"Loading trained model from {checkpoint_path}")
        try:
            model, config = load_trained_model(checkpoint_path, tokenizer)
            print("Successfully loaded trained model")
        except Exception as e:
            print(f"Could not load checkpoint: {e}")
            print("Creating new model for evaluation (not trained)")
            config = LOBTransformerConfig(
                vocab_size=len(tokenizer.vocab),
                d_model=512,
                n_layers=6,
                n_heads=8,
                d_ff=2048,
                max_seq_length=1024
            )
            model = create_lob_transformer_small(config.vocab_size)
    else:
        print("No checkpoint found, creating new model for evaluation")
        config = LOBTransformerConfig(
            vocab_size=len(tokenizer.vocab),
            d_model=512,
            n_layers=6,
            n_heads=8,
            d_ff=2048,
            max_seq_length=1024
        )
        model = create_lob_transformer_small(config.vocab_size)

    # Run comprehensive evaluation
    metrics = evaluate_model_comprehensive(
        model, test_dataset, tokenizer, config, device
    )

    print("\nEvaluation completed! Check ./plots/ for visualizations.")


if __name__ == "__main__":
    main()
