"""
Example of tokenizing limit order book events for sequence modeling.
"""

from __future__ import annotations

from collections import Counter
from typing import Iterable, List, Tuple

import numpy as np

from lobgpt.book_state import reconstruct_from_tardis
from lobgpt.hdb import get_dataset
from lobgpt.tokenizer import EventType, LOBTokenizer, TokenComponents


def format_token(token: TokenComponents) -> str:
    """Render token components as a compact string."""
    return token.to_string()


def create_sequences(tokens: np.ndarray, sequence_length: int, stride: int = 1) -> Tuple[np.ndarray, np.ndarray]:
    """Construct input/target sequences for autoregressive training."""
    sequences: List[np.ndarray] = []
    targets: List[np.ndarray] = []

    for start in range(0, len(tokens) - sequence_length, stride):
        seq = tokens[start : start + sequence_length]
        tgt = tokens[start + 1 : start + sequence_length + 1]
        sequences.append(seq)
        targets.append(tgt)

    if not sequences:
        return np.empty((0, sequence_length), dtype=np.int32), np.empty((0, sequence_length), dtype=np.int32)

    return np.stack(sequences), np.stack(targets)


def analyze_token_patterns(tokens: np.ndarray, tokenizer: LOBTokenizer) -> None:
    """Inspect short-range token transitions."""
    print("\n=== Pattern Analysis ===")
    if len(tokens) < 2:
        print("Not enough tokens for pattern analysis.")
        return

    transitions = Counter()
    for curr_id, next_id in zip(tokens[:-1], tokens[1:]):
        curr_token = tokenizer.decode_token(int(curr_id))
        next_token = tokenizer.decode_token(int(next_id))
        transitions[(curr_token.event_type, next_token.event_type)] += 1

    print("Most common event-type transitions:")
    for (curr_event, next_event), count in transitions.most_common(10):
        print(f"  {curr_event.name:>6s} → {next_event.name:<6s}: {count}")


def main() -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Demonstrate LOB event tokenization."""

    # Load data
    product = "BTCUSDT"
    times = ['250912.000000', '250912.000100']  # 1 minute
    depth = 10

    print(f"Loading data for {product} from {times[0]} to {times[1]}...")

    dl = get_dataset("tardis")
    inc_df = dl.load_inc(product, times)
    book_states = reconstruct_from_tardis(product, times, depth)

    print(f"Loaded {len(inc_df)} incremental updates")
    print(f"Reconstructed {len(book_states)} book states")

    # Initialize tokenizer
    tokenizer = LOBTokenizer()

    print(f"\nVocabulary size: {tokenizer.vocab_size}")

    # Tokenize events
    print("\nTokenizing events...")
    tokens = tokenizer.tokenize(inc_df)
    print(f"Generated {len(tokens)} tokens")

    # Show some example tokens
    print("\n=== First 20 tokens ===")
    first_tokens = tokens[:20]
    decoded = tokenizer.decode_tokens(first_tokens)

    for i, (token_id, token_components) in enumerate(zip(first_tokens, decoded)):
        print(f"{i:2d}: {int(token_id):5d} -> {format_token(token_components)}")

    # Analyze token distribution
    print("\n=== Token Analysis ===")
    unique_tokens, counts = np.unique(tokens, return_counts=True)
    print(f"Unique tokens: {len(unique_tokens)}")

    top_indices = np.argsort(counts)[-10:][::-1]
    for idx in top_indices:
        token_id = int(unique_tokens[idx])
        count = counts[idx]
        description = format_token(tokenizer.decode_token(token_id))
        print(f"  {token_id:5d}: {count:6d} ({count / len(tokens) * 100:.1f}%) - {description}")

    # Create sequences for training
    print("\n=== Creating Training Sequences ===")
    sequence_length = 50
    X, y = create_sequences(tokens, sequence_length=sequence_length, stride=10)

    print(f"Created {len(X)} sequences of length {sequence_length}")
    print(f"Input shape: {X.shape}")
    print(f"Target shape: {y.shape}")

    if len(X) > 0:
        sample_seq = X[0][:10]
        sample_decoded = tokenizer.decode_tokens(sample_seq)
        print("\nSample sequence (first 10 tokens):")
        for i, (token_id, token_components) in enumerate(zip(sample_seq, sample_decoded)):
            print(f"  {i:2d}: {int(token_id):5d} -> {format_token(token_components)}")

    # Event type distribution
    print("\n=== Event Type Distribution ===")
    event_counts = Counter()
    for token_id in tokens:
        event = tokenizer.decode_token(int(token_id))
        event_counts[event.event_type.name] += 1

    total_events = sum(event_counts.values())
    for event_name, count in sorted(event_counts.items(), key=lambda x: x[1], reverse=True):
        pct = count / total_events * 100 if total_events else 0.0
        print(f"  {event_name:8s}: {count:6d} ({pct:5.1f}%)")

    analyze_token_patterns(tokens, tokenizer)
    return tokens, X, y


if __name__ == "__main__":
    main()
