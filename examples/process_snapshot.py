"""
Extract and process initial order book snapshot from incremental book data.
"""

import polars as pl

from lobgpt.hdb import get_dataset


def process_initial_snapshot(df: pl.DataFrame, tick_size: float = 0.1) -> dict:
    """
    Process the initial snapshot block from incremental book data.

    Args:
        df: DataFrame with incremental book data
        tick_size: Price increment for constructing levels (default 0.1)

    Returns:
        Dictionary containing:
            - snapshot_rows: Initial snapshot data
            - lowest_ask: Lowest ask price in snapshot
            - highest_bid: Highest bid price in snapshot
            - ask_levels: 25 ask levels with prices and amounts
            - bid_levels: 25 bid levels with prices and amounts
    """
    df_with_groups = df.with_columns(
        (pl.col('is_snapshot') != pl.col('is_snapshot').shift(1, fill_value=True))
        .cum_sum()
        .alias('snapshot_group')
    )

    # Get only the first group (which should be the initial snapshot)
    snapshot_df = df_with_groups.filter(pl.col('snapshot_group') == 0).drop('snapshot_group')

    # Verify it's actually a snapshot
    if snapshot_df.is_empty() or not snapshot_df['is_snapshot'][0]:
        print("Warning: No initial snapshot found in data")
        snapshot_df = df.filter(pl.col('is_snapshot'))

    # Separate asks and bids
    asks_df = snapshot_df.filter(pl.col('side') == 'ask')
    bids_df = snapshot_df.filter(pl.col('side') == 'bid')

    # Find lowest ask and highest bid
    lowest_ask = asks_df['price'].min()
    highest_bid = bids_df['price'].max()

    print(f"Snapshot contains {len(snapshot_df)} rows")
    print(f"Lowest ask price: {lowest_ask}")
    print(f"Highest bid price: {highest_bid}")

    # Group by price and sum amounts for each side
    asks_by_price = (
        asks_df
        .group_by('price')
        .agg(pl.col('amount').sum())
        .sort('price')
    )

    bids_by_price = (
        bids_df
        .group_by('price')
        .agg(pl.col('amount').sum())
        .sort('price', descending=True)
    )

    # Construct 25 ask levels starting from lowest ask
    ask_levels = []
    for i in range(25):
        price = lowest_ask + (i * tick_size)
        # Find amount at this price level
        amount_row = asks_by_price.filter(pl.col('price') == price)
        if len(amount_row) > 0:
            amount = amount_row['amount'][0]
        else:
            amount = 0.0
        ask_levels.append({'level': i, 'price': price, 'amount': amount})

    # Construct 25 bid levels going down from highest bid
    bid_levels = []
    for i in range(25):
        price = highest_bid - (i * tick_size)
        # Find amount at this price level
        amount_row = bids_by_price.filter(pl.col('price') == price)
        if len(amount_row) > 0:
            amount = amount_row['amount'][0]
        else:
            amount = 0.0
        bid_levels.append({'level': i, 'price': price, 'amount': amount})

    return {
        'snapshot_rows': snapshot_df,
        'lowest_ask': lowest_ask,
        'highest_bid': highest_bid,
        'ask_levels': pl.DataFrame(ask_levels),
        'bid_levels': pl.DataFrame(bid_levels)
    }


def main():
    """
    Example usage of snapshot processing.
    """
    # Initialize data loader
    dl = get_dataset("tardis")

    # Define time range (starting at beginning of day to ensure we get snapshot)
    times = ['250912.000000', '250912.001500']  # Short range for testing

    print(f"Loading incremental book data for BTCUSDT from {times[0]} to {times[1]}...")

    # Load incremental book data
    df = dl.load_book("BTCUSDT", times, type="incremental_book_L2")

    print(f"Loaded {len(df)} rows")
    print(f"Number of snapshot rows: {df.filter(pl.col('is_snapshot')).height}")

    # Process the snapshot
    result = process_initial_snapshot(df)

    print("\n=== Ask Levels (25 levels from lowest ask) ===")
    print(result['ask_levels'])

    print("\n=== Bid Levels (25 levels from highest bid) ===")
    print(result['bid_levels'])

    # Show summary statistics
    ask_levels = result['ask_levels']
    bid_levels = result['bid_levels']

    print("\n=== Summary ===")
    print(f"Ask levels with liquidity: {(ask_levels['amount'] > 0).sum()}/25")
    print(f"Total ask volume: {ask_levels['amount'].sum():.4f}")
    print(f"Bid levels with liquidity: {(bid_levels['amount'] > 0).sum()}/25")
    print(f"Total bid volume: {bid_levels['amount'].sum():.4f}")
    print(f"Spread: {result['lowest_ask'] - result['highest_bid']:.1f}")

    return result


if __name__ == "__main__":
    result = main()