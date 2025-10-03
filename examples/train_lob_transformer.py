"""
Example: Training a LOB Transformer for next-token prediction.
"""

import torch
from lobgpt.hdb import get_dataset
from lobgpt.book_state import reconstruct_from_tardis
from lobgpt.tokenizer import LOBTokenizer
from lobgpt.pytorch_dataset import TokenBookDataset
from lobgpt.models.lob_transformer import LOBTransformerConfig, create_lob_transformer_small
from lobgpt.training.lob_trainer import train_lob_transformer


def create_sample_datasets():
    """Create sample datasets for training."""

    print("Loading and preparing data...")

    # Load data for different time periods
    dl = get_dataset("tardis")
    product = "BTCUSDT"

    # Training data
    train_times = ['250912.000000', '250912.120000']  # 2 hours
    train_book_states = reconstruct_from_tardis(product, train_times, depth=10)
    train_inc_df = dl.load_inc(product, train_times)

    # Validation data
    val_times = ['250912.120000', '250912.130000']  # 1 hour
    val_book_states = reconstruct_from_tardis(product, val_times, depth=10)
    val_inc_df = dl.load_inc(product, val_times)

    print(f"Train book states: {len(train_book_states)}")
    print(f"Train inc events: {len(train_inc_df)}")
    print(f"Val book states: {len(val_book_states)}")
    print(f"Val inc events: {len(val_inc_df)}")

    # Create tokenizer
    tokenizer = LOBTokenizer(tick_size=0.1, ref_size=1.0)

    # Create datasets
    train_dataset = TokenBookDataset(
        inc_df=train_inc_df,
        book_states=train_book_states,
        tokenizer=tokenizer,
        sequence_length=200,
        horizon=10,
        threshold=0.002,
        stride=10
    )

    val_dataset = TokenBookDataset(
        inc_df=val_inc_df,
        book_states=val_book_states,
        tokenizer=tokenizer,
        sequence_length=200,
        horizon=10,
        threshold=0.002,
        stride=50  # Sparser sampling for validation
    )

    print(f"Train dataset: {train_dataset.get_stats()}")
    print(f"Val dataset: {val_dataset.get_stats()}")

    return train_dataset, val_dataset, tokenizer


def main():
    """Main training function."""

    print("=" * 60)
    print("LOB Transformer Training Example")
    print("=" * 60)

    # Create datasets
    train_dataset, val_dataset, tokenizer = create_sample_datasets()

    # Model configuration
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

    print(f"\nModel configuration:")
    print(f"  Vocab size: {config.vocab_size}")
    print(f"  Model dim: {config.d_model}")
    print(f"  Layers: {config.n_layers}")
    print(f"  Heads: {config.n_heads}")

    # Calculate approximate model size
    model = create_lob_transformer_small(config.vocab_size)
    n_params = model.get_num_params()
    print(f"  Parameters: {n_params:,} ({n_params/1e6:.1f}M)")

    # Training configuration
    training_config = {
        "max_epochs": 20,
        "batch_size": 16,  # Small batch for demo
        "learning_rate": 3e-4,
        "project_name": "lob-transformer-demo",
        "experiment_name": "demo-run",
        "checkpoint_dir": "./checkpoints"
    }

    print(f"\nTraining configuration:")
    for key, value in training_config.items():
        print(f"  {key}: {value}")

    # Train model
    print(f"\nStarting training...")
    try:
        trained_model = train_lob_transformer(
            config=config,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            **training_config
        )

        print("Training completed successfully!")

        # Generate some samples
        print("\n" + "=" * 60)
        print("Generating sample sequences...")
        print("=" * 60)

        model = trained_model.model
        model.eval()

        with torch.no_grad():
            # Create a random prompt
            prompt = torch.randint(1, config.vocab_size // 10, (1, 20))

            # Generate
            generated = model.generate(
                prompt,
                max_length=30,
                temperature=1.0,
                top_k=50,
                do_sample=True
            )

            print(f"Prompt tokens: {prompt[0].tolist()}")
            print(f"Generated tokens: {generated[0].tolist()}")

            # Decode if possible
            try:
                prompt_decoded = tokenizer.decode_tokens(prompt[0].numpy())
                generated_decoded = tokenizer.decode_tokens(generated[0].numpy())

                print(f"\nPrompt events: {prompt_decoded[:5]}")
                print(f"Generated events: {generated_decoded[-10:]}")

            except Exception as e:
                print(f"Decoding error: {e}")

    except Exception as e:
        print(f"Training failed: {e}")
        import traceback
        traceback.print_exc()

    print("\nDone!")


if __name__ == "__main__":
    main()