"""
Example of tokenizing limit order book events for sequence modeling.
"""

import numpy as np

from lobgpt.book_state import reconstruct_from_tardis
from lobgpt.hdb import get_dataset
from lobgpt.tokenizer import AdvancedLOBTokenizer, LOBTokenizer, create_lob_sequences


def main():
    """Demonstrate LOB event tokenization."""

    # Load data
    product = "BTCUSDT"
    times = ['250912.000000', '250912.000100']  # 1 minute
    depth = 10

    print(f"Loading data for {product} from {times[0]} to {times[1]}...")

    # Get incremental data and book states
    dl = get_dataset("tardis")
    inc_df = dl.load_inc(product, times)
    book_states = reconstruct_from_tardis(product, times, depth)

    print(f"Loaded {len(inc_df)} incremental updates")
    print(f"Reconstructed {len(book_states)} book states")

    # Initialize tokenizer
    tokenizer = LOBTokenizer(tick_size=0.1, ref_size=1.0)

    print(f"\nVocabulary size: {len(tokenizer.vocab)}")

    # Tokenize events
    print("\nTokenizing events...")
    tokens = tokenizer.tokenize_events(inc_df, book_states)

    print(f"Generated {len(tokens)} tokens")

    # Show some example tokens
    print("\n=== First 20 tokens ===")
    first_tokens = tokens[:20]
    decoded = tokenizer.decode_tokens(first_tokens)

    for i, (token_id, token_str) in enumerate(zip(first_tokens, decoded)):
        print(f"{i:2d}: {token_id:4d} -> {token_str}")

    # Analyze token distribution
    print("\n=== Token Analysis ===")
    unique_tokens, counts = np.unique(tokens, return_counts=True)
    print(f"Unique tokens: {len(unique_tokens)}")
    print("Most common tokens:")

    # Show top 10 most frequent tokens
    top_indices = np.argsort(counts)[-10:][::-1]
    for idx in top_indices:
        token_id = unique_tokens[idx]
        count = counts[idx]
        token_str = tokenizer.vocab.get(token_id, "UNK")
        print(f"  {token_id:4d}: {count:6d} ({count/len(tokens)*100:.1f}%) - {token_str}")

    # Create sequences for training
    print("\n=== Creating Training Sequences ===")
    sequence_length = 50
    X, y = create_lob_sequences(tokens, sequence_length=sequence_length, stride=10)

    print(f"Created {len(X)} sequences of length {sequence_length}")
    print(f"Input shape: {X.shape}")
    print(f"Target shape: {y.shape}")

    # Show a sample sequence
    print("\nSample sequence (first 10 tokens):")
    sample_seq = X[0][:10]
    sample_decoded = tokenizer.decode_tokens(sample_seq)
    for i, (token_id, token_str) in enumerate(zip(sample_seq, sample_decoded)):
        print(f"  {i:2d}: {token_id:4d} -> {token_str}")

    # Advanced tokenizer example
    print("\n=== Advanced Tokenizer Features ===")
    adv_tokenizer = AdvancedLOBTokenizer(tick_size=0.1, ref_size=1.0)

    # Add special tokens
    tokens_with_special = adv_tokenizer.add_special_tokens(tokens[:100], add_cls=True, add_sep=True)
    print(f"Original tokens: {len(tokens[:100])}")
    print(f"With special tokens: {len(tokens_with_special)}")

    # Create masked language modeling targets
    masked_tokens, labels = adv_tokenizer.create_mlm_targets(tokens[:100], mask_prob=0.15)
    mask_count = (masked_tokens == adv_tokenizer.special_tokens['MASK']).sum()
    print(f"Masked {mask_count} tokens out of {len(tokens[:100])} ({mask_count/len(tokens[:100])*100:.1f}%)")

    # Create attention mask
    attention_mask = adv_tokenizer.create_attention_mask(tokens_with_special)
    print(f"Attention mask shape: {attention_mask.shape}")

    # Event type analysis
    print("\n=== Event Type Distribution ===")
    from lobgpt.tokenizer import EventType, LOBToken

    event_counts = {event.name: 0 for event in EventType}
    for token_id in tokens:
        try:
            token = LOBToken.from_int(token_id)
            event_counts[token.event_type.name] += 1
        except (ValueError, IndexError):
            continue

    total_events = sum(event_counts.values())
    for event_name, count in event_counts.items():
        if count > 0:
            pct = count / total_events * 100
            print(f"  {event_name:8s}: {count:6d} ({pct:5.1f}%)")

    return tokens, X, y


def analyze_token_patterns(tokens: np.ndarray, tokenizer: LOBTokenizer):
    """Analyze patterns in the tokenized sequence."""
    print("\n=== Pattern Analysis ===")

    from lobgpt.tokenizer import LOBToken

    # Analyze transitions between different event types
    transitions = {}
    for i in range(len(tokens) - 1):
        try:
            curr_token = LOBToken.from_int(tokens[i])
            next_token = LOBToken.from_int(tokens[i + 1])

            curr_event = curr_token.event_type.name
            next_event = next_token.event_type.name

            transition = f"{curr_event} -> {next_event}"
            transitions[transition] = transitions.get(transition, 0) + 1
        except (ValueError, IndexError):
            continue

    # Show most common transitions
    print("Most common event transitions:")
    sorted_transitions = sorted(transitions.items(), key=lambda x: x[1], reverse=True)
    for transition, count in sorted_transitions[:10]:
        pct = count / (len(tokens) - 1) * 100
        print(f"  {transition:20s}: {count:6d} ({pct:5.1f}%)")

    # Analyze price level distribution
    level_counts = {}
    for token_id in tokens:
        try:
            token = LOBToken.from_int(token_id)
            level = token.price_level.name
            level_counts[level] = level_counts.get(level, 0) + 1
        except (ValueError, IndexError):
            continue

    print("\nPrice level distribution:")
    for level, count in level_counts.items():
        pct = count / len(tokens) * 100
        print(f"  {level:8s}: {count:6d} ({pct:5.1f}%)")


if __name__ == "__main__":
    tokens, X, y = main()

    # Additional analysis
    tokenizer = LOBTokenizer(tick_size=0.1, ref_size=1.0)
    analyze_token_patterns(tokens, tokenizer)

    print("\n=== Summary ===")
    print("✓ Successfully tokenized LOB events")
    print("✓ Created training sequences for next-token prediction")
    print("✓ Analyzed token patterns and distributions")
    print("✓ Ready for transformer training!")