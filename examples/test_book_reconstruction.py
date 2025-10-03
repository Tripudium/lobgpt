"""
Test order book reconstruction from incremental updates.
"""

import polars as pl
from lobgpt.hdb import get_dataset
from lobgpt.book_state import reconstruct_from_tardis, reconstruct_book_states


def main():
    """Test book reconstruction with real data."""

    # Test parameters
    product = "BTCUSDT"
    times = ['250912.000000', '250912.000100']  # 1 minute of data
    depth = 10

    print(f"Loading data for {product} from {times[0]} to {times[1]}...")

    # Method 1: Use convenience function
    print("\n=== Method 1: Using convenience function ===")
    book_states = reconstruct_from_tardis(product, times, depth)

    print(f"Reconstructed {len(book_states)} book states")
    print(f"Columns: {book_states.columns[:10]}...")  # Show first 10 columns

    # Show first few states
    print("\nFirst 3 book states:")
    print(book_states.head(3))

    # Method 2: Manual process with more control
    print("\n=== Method 2: Manual process ===")
    dl = get_dataset("tardis")

    # Load incremental data
    inc_df = dl.load_inc(product, times)
    print(f"Loaded {len(inc_df)} incremental updates")

    # Count event types
    event_counts = inc_df.group_by('event').agg(pl.count())
    print("\nEvent type counts:")
    print(event_counts)

    # Extract initial snapshot - find first continuous block of snapshots
    snapshot_rows = inc_df.filter(pl.col('is_snapshot'))
    if not snapshot_rows.is_empty():
        first_snapshot_ts = snapshot_rows['ts'][0]
        initial_snapshot = inc_df.filter(
            (pl.col('is_snapshot')) & (pl.col('ts') == first_snapshot_ts)
        )
    else:
        initial_snapshot = None

    snapshot_size = len(initial_snapshot) if initial_snapshot is not None else 0
    print(f"\nInitial snapshot contains {snapshot_size} rows")

    # Reconstruct book states
    book_states_manual = reconstruct_book_states(inc_df, depth, initial_snapshot)
    print(f"Reconstructed {len(book_states_manual)} book states")

    # Analyze spread over time
    print("\n=== Spread Analysis ===")
    book_states_with_spread = book_states.with_columns(
        (pl.col('asks[0].price') - pl.col('bids[0].price')).alias('spread')
    )

    spread_stats = book_states_with_spread.select(
        pl.col('spread').min().alias('min_spread'),
        pl.col('spread').mean().alias('avg_spread'),
        pl.col('spread').max().alias('max_spread'),
        pl.col('spread').std().alias('std_spread'),
    )
    print("Spread statistics:")
    print(spread_stats)

    # Check for any infinite prices (indicating empty levels)
    n_inf_asks = sum([
        (book_states[f'asks[{i}].price'] == float('inf')).sum()
        for i in range(depth)
    ])
    n_inf_bids = sum([
        (book_states[f'bids[{i}].price'] == float('-inf')).sum()
        for i in range(depth)
    ])
    print(f"\nEmpty levels - Asks: {n_inf_asks}, Bids: {n_inf_bids}")

    # Sample a specific timestamp
    sample_idx = len(book_states) // 2
    sample_state = book_states[sample_idx]
    print(f"\n=== Book state at index {sample_idx} ===")
    print(f"Timestamp: {sample_state['ts'][0]}")
    print(f"Best bid: {sample_state['bids[0].price'][0]:.1f} @ {sample_state['bids[0].amount'][0]:.3f}")
    print(f"Best ask: {sample_state['asks[0].price'][0]:.1f} @ {sample_state['asks[0].amount'][0]:.3f}")
    print(f"Spread: {sample_state['asks[0].price'][0] - sample_state['bids[0].price'][0]:.1f}")

    # Show top 5 levels for this state
    print("\nTop 5 bid levels:")
    for i in range(5):
        price = sample_state[f'bids[{i}].price'][0]
        amount = sample_state[f'bids[{i}].amount'][0]
        if price != float('-inf'):
            print(f"  Level {i}: {price:.1f} @ {amount:.3f}")

    print("\nTop 5 ask levels:")
    for i in range(5):
        price = sample_state[f'asks[{i}].price'][0]
        amount = sample_state[f'asks[{i}].amount'][0]
        if price != float('inf'):
            print(f"  Level {i}: {price:.1f} @ {amount:.3f}")

    return book_states


if __name__ == "__main__":
    book_states = main()