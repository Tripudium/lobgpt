"""
Example: Using the LOBSTER dataloader for US equity limit order book data.
"""

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

from lobgpt.hdb import get_dataset
from lobgpt.hdb.lobster_dataloader import (
    SUBMISSION, PARTIAL_CANCEL, DELETION, VISIBLE_EXECUTION,
    HIDDEN_EXECUTION, CROSS_TRADE, TRADING_HALT
)


def explore_lobster_data():
    """Explore LOBSTER limit order book data."""
    print("="*60)
    print("LOBSTER Data Exploration")
    print("="*60)

    # Initialize LOBSTER dataloader
    dl = get_dataset("lobster")

    # Check available symbols and dates
    symbols = dl.get_available_symbols()
    print(f"\nAvailable symbols: {symbols}")

    if not symbols:
        print("\nNo LOBSTER data found. Please place raw LOBSTER files in:")
        print(f"  {dl.raw_path}")
        print("\nFile naming convention:")
        print("  SYMBOL_YYYYMMDD_message_LEVELS.csv")
        print("  SYMBOL_YYYYMMDD_orderbook_LEVELS.csv")
        print("\nExample:")
        print("  AAPL_20240101_message_10.csv")
        print("  AAPL_20240101_orderbook_10.csv")
        return

    # Use first available symbol
    symbol = symbols[0]
    dates = dl.get_available_dates(symbol)
    print(f"\nAvailable dates for {symbol}: {dates}")

    if not dates:
        print(f"No dates available for {symbol}")
        return

    # Load message data
    print(f"\n1. Loading message data for {symbol} on {dates[0]}...")
    messages_df = dl.load_messages(symbol, [dates[0]])

    if messages_df is not None:
        print(f"   Loaded {len(messages_df)} messages")
        print(f"   Event types: {messages_df['event_type'].unique().sort().to_list()}")
        print(f"   Time range: {messages_df['time'].min():.3f} - {messages_df['time'].max():.3f} seconds after midnight")

        # Show sample messages
        print("\n   Sample messages:")
        print(messages_df.head(5))

        # Event type distribution
        event_counts = messages_df.group_by("event_type").count()
        print("\n   Event type distribution:")
        print(event_counts)

    # Load orderbook data
    print(f"\n2. Loading orderbook data for {symbol} on {dates[0]}...")
    orderbook_df = dl.load_book(symbol, [dates[0]], depth=5, type="orderbook")

    if orderbook_df is not None:
        print(f"   Loaded {len(orderbook_df)} orderbook snapshots")

        # Calculate spreads
        orderbook_df = orderbook_df.with_columns([
            (pl.col("ask_price_1") - pl.col("bid_price_1")).alias("spread"),
            ((pl.col("ask_price_1") + pl.col("bid_price_1")) / 2).alias("mid_price")
        ])

        # Show sample orderbook
        print("\n   Sample orderbook snapshots:")
        print(orderbook_df.select([
            "seq_num", "bid_price_1", "bid_size_1",
            "ask_price_1", "ask_size_1", "spread", "mid_price"
        ]).head(5))

        # Spread statistics
        print(f"\n   Spread statistics:")
        print(f"     Mean: ${orderbook_df['spread'].mean():.4f}")
        print(f"     Std:  ${orderbook_df['spread'].std():.4f}")
        print(f"     Min:  ${orderbook_df['spread'].min():.4f}")
        print(f"     Max:  ${orderbook_df['spread'].max():.4f}")

    # Load incremental book data (L3 messages)
    print(f"\n2b. Loading incremental book L3 data for {symbol} on {dates[0]}...")
    l3_df = dl.load_book(symbol, [dates[0]], type="incremental_book_L3")

    if l3_df is not None:
        print(f"   Loaded {len(l3_df)} L3 incremental messages")
        print(f"   Same as messages: {len(l3_df) == len(messages_df)}")
        print("   L3 data contains all order-level events (submissions, cancellations, executions)")

    # Load combined data
    print(f"\n3. Loading combined message and orderbook data...")
    combined_df = dl.load_combined(symbol, [dates[0]], depth=3)

    if combined_df is not None:
        print(f"   Combined {len(combined_df)} records")

        # Analyze market impact of different event types
        print("\n   Market impact by event type:")
        event_type_names = {
            SUBMISSION: "submission",
            PARTIAL_CANCEL: "partial_cancel",
            DELETION: "deletion",
            VISIBLE_EXECUTION: "visible_execution"
        }

        for event_type in [SUBMISSION, PARTIAL_CANCEL, DELETION, VISIBLE_EXECUTION]:
            subset = combined_df.filter(pl.col("event_type") == event_type)
            if len(subset) > 0:
                event_name = event_type_names[event_type]
                avg_spread = subset.with_columns(
                    (pl.col("ask_price_1") - pl.col("bid_price_1")).alias("spread")
                )["spread"].mean()
                print(f"     {event_name}: avg spread = ${avg_spread:.4f}")

    # Convert to incremental format for tokenization
    print(f"\n4. Converting to incremental format for tokenization...")
    inc_df = dl.load_inc(symbol, [dates[0]])

    if inc_df is not None:
        print(f"   Converted {len(inc_df)} messages to incremental format")
        print("\n   Sample incremental updates:")
        print(inc_df.head(5))

    # Load trades
    print(f"\n5. Loading trade data...")
    trades_df = dl.load_trades(symbol, [dates[0]])

    if trades_df is not None:
        print(f"   Found {len(trades_df)} trades")
        print(f"   Trade types: {trades_df['trade_type'].unique().to_list()}")

        # Trade statistics
        print(f"\n   Trade statistics:")
        print(f"     Total volume: {trades_df['amount'].sum():,} shares")
        print(f"     Average size: {trades_df['amount'].mean():.1f} shares")
        print(f"     Average price: ${trades_df['price'].mean():.2f}")

        # Trade distribution by type
        trade_dist = trades_df.group_by("trade_type").agg([
            pl.count("price").alias("count"),
            pl.sum("amount").alias("total_volume"),
            pl.mean("price").alias("avg_price")
        ])
        print("\n   Trade distribution by type:")
        print(trade_dist)


def visualize_lobster_data():
    """Create visualizations of LOBSTER data."""
    print("\n" + "="*60)
    print("LOBSTER Data Visualization")
    print("="*60)

    dl = get_dataset("lobster")

    # Get first available data
    symbols = dl.get_available_symbols()
    if not symbols:
        print("No data available for visualization")
        return

    symbol = symbols[0]
    dates = dl.get_available_dates(symbol)
    if not dates:
        print(f"No dates available for {symbol}")
        return

    # Load data
    messages_df = dl.load_messages(symbol, [dates[0]])
    orderbook_df = dl.load_book(symbol, [dates[0]], depth=5)

    if messages_df is None or orderbook_df is None:
        print("Could not load data for visualization")
        return

    # Create visualizations
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # 1. Event type distribution
    event_counts = messages_df.group_by("event_type").count().sort("event_type")
    axes[0,0].bar(
        event_counts["event_type"].to_numpy(),
        event_counts["count"].to_numpy()
    )
    axes[0,0].set_xlabel("Event Type")
    axes[0,0].set_ylabel("Count")
    axes[0,0].set_title("Event Type Distribution")
    axes[0,0].set_xticks(range(1, 8))
    axes[0,0].set_xticklabels([
        "Submit", "Partial\nCancel", "Delete", "Visible\nExec",
        "Hidden\nExec", "Cross", "Halt"
    ], rotation=45)

    # 2. Order size distribution
    sizes = messages_df.filter(pl.col("event_type") == 1)["size"].to_numpy()
    if len(sizes) > 0:
        axes[0,1].hist(sizes, bins=50, alpha=0.7, edgecolor='black')
        axes[0,1].set_xlabel("Order Size (shares)")
        axes[0,1].set_ylabel("Frequency")
        axes[0,1].set_title("Order Size Distribution (New Submissions)")
        axes[0,1].set_yscale('log')

    # 3. Mid-price evolution
    orderbook_df = orderbook_df.with_columns(
        ((pl.col("ask_price_1") + pl.col("bid_price_1")) / 2).alias("mid_price")
    )
    mid_prices = orderbook_df["mid_price"].to_numpy()
    axes[1,0].plot(range(len(mid_prices)), mid_prices, linewidth=0.5)
    axes[1,0].set_xlabel("Event Number")
    axes[1,0].set_ylabel("Mid Price ($)")
    axes[1,0].set_title("Mid-Price Evolution")

    # 4. Spread over time
    orderbook_df = orderbook_df.with_columns(
        (pl.col("ask_price_1") - pl.col("bid_price_1")).alias("spread")
    )
    spreads = orderbook_df["spread"].to_numpy()
    axes[1,1].plot(range(len(spreads)), spreads * 100, linewidth=0.5, alpha=0.7)
    axes[1,1].set_xlabel("Event Number")
    axes[1,1].set_ylabel("Spread (cents)")
    axes[1,1].set_title("Bid-Ask Spread Evolution")

    plt.suptitle(f"LOBSTER Data Analysis: {symbol} on {dates[0]}")
    plt.tight_layout()
    plt.savefig("lobster_analysis.png", dpi=300, bbox_inches='tight')
    print(f"\nVisualization saved to lobster_analysis.png")


def prepare_for_modeling():
    """Prepare LOBSTER data for LOB modeling."""
    print("\n" + "="*60)
    print("Preparing LOBSTER Data for Modeling")
    print("="*60)

    from lobgpt.tokenizer import LOBTokenizer
    from lobgpt.book_state import BookState

    dl = get_dataset("lobster")

    symbols = dl.get_available_symbols()
    if not symbols:
        print("No data available")
        return

    symbol = symbols[0]
    dates = dl.get_available_dates(symbol)

    if not dates:
        print(f"No dates available for {symbol}")
        return

    # Load incremental data
    inc_df = dl.load_inc(symbol, [dates[0]])

    if inc_df is None:
        print("Could not load incremental data")
        return

    print(f"\nLoaded {len(inc_df)} incremental updates")

    # Initialize tokenizer for US equity markets
    # Note: LOBSTER prices are in dollars, so we need appropriate tick size
    tokenizer = LOBTokenizer(
        tick_size=0.01,  # Penny tick size for US equities
        ref_size=100.0   # Reference size in shares
    )

    # Convert to book states format
    # This would require orderbook snapshots for proper reconstruction
    orderbook_df = dl.load_book(symbol, [dates[0]], depth=10)

    if orderbook_df is not None:
        print(f"\nLoaded {len(orderbook_df)} orderbook snapshots")

        # Create book states compatible with tokenization
        # Note: This is a simplified example
        book_states = []
        for row in orderbook_df.iter_rows(named=True):
            state = {
                "ts": row.get("seq_num", 0),  # Use sequence number as timestamp
                "symbol": row["symbol"],
            }

            # Add bid levels
            for i in range(1, 11):
                if f"bid_price_{i}" in row and row[f"bid_price_{i}"] is not None:
                    state[f"bids[{i-1}].price"] = row[f"bid_price_{i}"]
                    state[f"bids[{i-1}].amount"] = row[f"bid_size_{i}"]
                else:
                    state[f"bids[{i-1}].price"] = float('inf')
                    state[f"bids[{i-1}].amount"] = 0

            # Add ask levels
            for i in range(1, 11):
                if f"ask_price_{i}" in row and row[f"ask_price_{i}"] is not None:
                    state[f"asks[{i-1}].price"] = row[f"ask_price_{i}"]
                    state[f"asks[{i-1}].amount"] = row[f"ask_size_{i}"]
                else:
                    state[f"asks[{i-1}].price"] = float('inf')
                    state[f"asks[{i-1}].amount"] = 0

            book_states.append(state)

        # Convert to polars DataFrame
        import polars as pl
        book_states_df = pl.DataFrame(book_states)

        print(f"\nCreated {len(book_states_df)} book states")
        print("Ready for tokenization and model training!")

        return inc_df, book_states_df, tokenizer

    return None


def main():
    """Main demonstration function."""
    print("LOBSTER Data Loader Example")
    print("=" * 60)
    print("\nNote: LOBSTER is a commercial data provider.")
    print("To use this loader, you need to:")
    print("1. Obtain LOBSTER data from https://lobsterdata.com/")
    print("2. Place files in data/lobster/raw/ with naming convention:")
    print("   SYMBOL_YYYYMMDD_message_LEVELS.csv")
    print("   SYMBOL_YYYYMMDD_orderbook_LEVELS.csv")
    print()

    # Explore available data
    explore_lobster_data()

    # Create visualizations if data is available
    dl = get_dataset("lobster")
    if dl.get_available_symbols():
        visualize_lobster_data()

        # Prepare for modeling
        result = prepare_for_modeling()
        if result:
            inc_df, book_states_df, tokenizer = result
            print(f"\nData successfully prepared for modeling:")
            print(f"  Incremental updates: {len(inc_df)}")
            print(f"  Book states: {len(book_states_df)}")
            print(f"  Tokenizer vocab size: {len(tokenizer.vocab)}")


if __name__ == "__main__":
    import polars as pl  # Import here to avoid issues if module is run directly
    main()