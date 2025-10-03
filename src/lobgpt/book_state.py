"""
Order book state reconstruction from incremental updates.
"""

from typing import Optional

import numpy as np
import polars as pl
from numba import jit


@jit(nopython=True)
def update_book_state(
    timestamp: int,
    event_type: int,  # 0 for trade, 1 for book update
    is_snapshot: bool,
    side: int,  # 1 for buy/bid, -1 for sell/ask
    price: float,
    amount: float,
    bid_prices: np.ndarray,
    bid_amounts: np.ndarray,
    ask_prices: np.ndarray,
    ask_amounts: np.ndarray,
    depth: int,
) -> None:
    """
    Update order book state in-place based on incremental update.

    Args:
        timestamp: Current timestamp
        event_type: 0 for trade (remove liquidity), 1 for book update
        is_snapshot: Whether this is a snapshot update
        side: 1 for buy/bid, -1 for sell/ask
        price: Price level
        amount: Amount (0 means delete the level)
        bid_prices: Array of bid prices (modified in-place)
        bid_amounts: Array of bid amounts (modified in-place)
        ask_prices: Array of ask prices (modified in-place)
        ask_amounts: Array of ask amounts (modified in-place)
        depth: Number of levels to maintain
    """

    # Handle trades (market orders) - they consume liquidity
    if event_type == 0:
        # Trades consume from the opposite side
        if side == 1:  # Buy trade consumes asks
            # Find and reduce liquidity at best ask
            if ask_amounts[0] > 0:
                ask_amounts[0] = max(0, ask_amounts[0] - amount)
                if ask_amounts[0] == 0:
                    # Shift asks up if best ask depleted
                    for i in range(depth - 1):
                        ask_prices[i] = ask_prices[i + 1]
                        ask_amounts[i] = ask_amounts[i + 1]
                    ask_prices[depth - 1] = np.inf
                    ask_amounts[depth - 1] = 0.0
        else:  # Sell trade consumes bids
            if bid_amounts[0] > 0:
                bid_amounts[0] = max(0, bid_amounts[0] - amount)
                if bid_amounts[0] == 0:
                    # Shift bids up if best bid depleted
                    for i in range(depth - 1):
                        bid_prices[i] = bid_prices[i + 1]
                        bid_amounts[i] = bid_amounts[i + 1]
                    bid_prices[depth - 1] = -np.inf
                    bid_amounts[depth - 1] = 0.0

    # Handle book updates (limit orders)
    elif event_type == 1:
        if side == -1:  # Ask side update
            if amount == 0:
                # Remove this price level
                for i in range(depth):
                    if ask_prices[i] == price:
                        # Shift remaining levels up
                        for j in range(i, depth - 1):
                            ask_prices[j] = ask_prices[j + 1]
                            ask_amounts[j] = ask_amounts[j + 1]
                        ask_prices[depth - 1] = np.inf
                        ask_amounts[depth - 1] = 0.0
                        break
            else:
                # Add or update price level
                inserted = False
                for i in range(depth):
                    if ask_prices[i] == price:
                        # Update existing level
                        ask_amounts[i] = amount
                        inserted = True
                        break
                    elif ask_prices[i] > price:
                        # Insert new level here
                        # Shift others down
                        for j in range(depth - 1, i, -1):
                            ask_prices[j] = ask_prices[j - 1]
                            ask_amounts[j] = ask_amounts[j - 1]
                        ask_prices[i] = price
                        ask_amounts[i] = amount
                        inserted = True
                        break

        else:  # Bid side update
            if amount == 0:
                # Remove this price level
                for i in range(depth):
                    if bid_prices[i] == price:
                        # Shift remaining levels up
                        for j in range(i, depth - 1):
                            bid_prices[j] = bid_prices[j + 1]
                            bid_amounts[j] = bid_amounts[j + 1]
                        bid_prices[depth - 1] = -np.inf
                        bid_amounts[depth - 1] = 0.0
                        break
            else:
                # Add or update price level
                inserted = False
                for i in range(depth):
                    if bid_prices[i] == price:
                        # Update existing level
                        bid_amounts[i] = amount
                        inserted = True
                        break
                    elif bid_prices[i] < price:
                        # Insert new level here
                        # Shift others down
                        for j in range(depth - 1, i, -1):
                            bid_prices[j] = bid_prices[j - 1]
                            bid_amounts[j] = bid_amounts[j - 1]
                        bid_prices[i] = price
                        bid_amounts[i] = amount
                        inserted = True
                        break


def reconstruct_book_states(
    inc_df: pl.DataFrame,
    depth: int = 10,
    initial_snapshot: Optional[pl.DataFrame] = None,
) -> pl.DataFrame:
    """
    Reconstruct order book states at each timestamp from incremental updates.

    Args:
        inc_df: DataFrame from load_inc with columns:
                ts, ts_local, event, id, is_snapshot, side, price, amount
        depth: Number of levels to maintain (default 10)
        initial_snapshot: Optional initial snapshot DataFrame to start from

    Returns:
        DataFrame with book state at each timestamp with columns:
        - ts: Timestamp
        - ts_local: Local timestamp
        - asks[0].price, asks[0].amount, ..., asks[depth-1].price, asks[depth-1].amount
        - bids[0].price, bids[0].amount, ..., bids[depth-1].price, bids[depth-1].amount
    """

    # Convert to numpy arrays for faster processing
    timestamps = inc_df['ts'].to_numpy()
    ts_locals = inc_df['ts_local'].to_numpy()
    events = inc_df['event'].to_numpy()
    is_snapshots = inc_df['is_snapshot'].to_numpy()
    sides = inc_df['side'].to_numpy()
    prices = inc_df['price'].to_numpy()
    amounts = inc_df['amount'].to_numpy()

    n_updates = len(timestamps)

    # Initialize book state arrays
    bid_prices = np.full(depth, -np.inf, dtype=np.float64)
    bid_amounts = np.zeros(depth, dtype=np.float64)
    ask_prices = np.full(depth, np.inf, dtype=np.float64)
    ask_amounts = np.zeros(depth, dtype=np.float64)

    # Arrays to store book states at each timestamp
    all_timestamps = []
    all_ts_locals = []
    all_bid_prices = []
    all_bid_amounts = []
    all_ask_prices = []
    all_ask_amounts = []

    # Process initial snapshot if provided
    if initial_snapshot is not None and not initial_snapshot.is_empty():
        # Group by side and price, summing amounts
        # Note: side is numeric (-1 for ask, 1 for bid)
        ask_snapshot = (
            initial_snapshot.filter(pl.col('side') == -1)
            .group_by('price')
            .agg(pl.col('amount').sum())
            .sort('price')
            .head(depth)
        )

        bid_snapshot = (
            initial_snapshot.filter(pl.col('side') == 1)
            .group_by('price')
            .agg(pl.col('amount').sum())
            .sort('price', descending=True)
            .head(depth)
        )

        # Initialize arrays from snapshot
        for i, row in enumerate(ask_snapshot.iter_rows()):
            if i < depth:
                ask_prices[i] = row[0]
                ask_amounts[i] = row[1]

        for i, row in enumerate(bid_snapshot.iter_rows()):
            if i < depth:
                bid_prices[i] = row[0]
                bid_amounts[i] = row[1]

    # Process each update
    snapshot_ts = initial_snapshot['ts'][0] if initial_snapshot is not None and not initial_snapshot.is_empty() else None

    for i in range(n_updates):
        # Skip snapshot rows if we already processed initial snapshot
        if is_snapshots[i] and snapshot_ts is not None and timestamps[i] == snapshot_ts:
            continue

        # Apply update
        update_book_state(
            timestamps[i],
            events[i],
            is_snapshots[i],
            sides[i],
            prices[i],
            amounts[i],
            bid_prices,
            bid_amounts,
            ask_prices,
            ask_amounts,
            depth
        )

        # Store current state
        all_timestamps.append(timestamps[i])
        all_ts_locals.append(ts_locals[i])
        all_bid_prices.append(bid_prices.copy())
        all_bid_amounts.append(bid_amounts.copy())
        all_ask_prices.append(ask_prices.copy())
        all_ask_amounts.append(ask_amounts.copy())

    # Create result DataFrame
    result_data = {
        'ts': all_timestamps,
        'ts_local': all_ts_locals,
    }

    # Add bid columns
    for i in range(depth):
        result_data[f'bids[{i}].price'] = [bp[i] for bp in all_bid_prices]
        result_data[f'bids[{i}].amount'] = [ba[i] for ba in all_bid_amounts]

    # Add ask columns
    for i in range(depth):
        result_data[f'asks[{i}].price'] = [ap[i] for ap in all_ask_prices]
        result_data[f'asks[{i}].amount'] = [aa[i] for aa in all_ask_amounts]

    return pl.DataFrame(result_data)


def reconstruct_from_tardis(
    product: str,
    times: list[str],
    depth: int = 10,
) -> pl.DataFrame:
    """
    Convenience function to reconstruct book states directly from Tardis data.

    Args:
        product: Trading pair symbol (e.g., "BTCUSDT")
        times: List of two strings in format '%y%m%d.%H%M%S' [start, end]
        depth: Number of levels to maintain (default 10)

    Returns:
        DataFrame with reconstructed book states
    """
    from lobgpt.hdb import get_dataset

    dl = get_dataset("tardis")

    # Load incremental data
    inc_df = dl.load_inc(product, times)

    # Find initial snapshot block - it may not start at row 0
    # Look for the first continuous block of snapshot rows
    snapshot_rows = inc_df.filter(pl.col('is_snapshot'))

    if not snapshot_rows.is_empty():
        # Get the timestamp of the first snapshot
        first_snapshot_ts = snapshot_rows['ts'][0]

        # Get all snapshots with the same timestamp (they form a complete snapshot)
        initial_snapshot = inc_df.filter(
            (pl.col('is_snapshot')) & (pl.col('ts') == first_snapshot_ts)
        )
    else:
        # No snapshot found
        initial_snapshot = None

    # Reconstruct book states
    return reconstruct_book_states(inc_df, depth, initial_snapshot)