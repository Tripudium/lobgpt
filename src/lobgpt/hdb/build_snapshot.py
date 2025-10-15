"""
Utilities for reconstructing order book snapshots from incremental streams.
"""

from __future__ import annotations

from typing import Sequence

import polars as pl

DEFAULT_SUBMISSION_CODES: tuple[int, ...] = (1,)
DEFAULT_CANCEL_CODES: tuple[int, ...] = (2, 3)
DEFAULT_EXECUTION_CODES: tuple[int, ...] = (4,)


def reconstruct_snapshots(
    messages: pl.DataFrame,
    initial_snapshot: pl.DataFrame,
    depth: int = 10,
    *,
    event_column: str = "event_code",
    price_column: str = "prc",
    volume_column: str = "vol",
    side_column: str = "is_buy",
    submission_codes: Sequence[int] = DEFAULT_SUBMISSION_CODES,
    cancel_codes: Sequence[int] = DEFAULT_CANCEL_CODES,
    execution_codes: Sequence[int] = DEFAULT_EXECUTION_CODES,
) -> pl.DataFrame:
    """
    Rebuild book snapshots by replaying incremental messages.

    Parameters
    ----------
    messages:
        Incremental order flow (e.g., ``load_book(..., type=\"incremental_book_L3\")``).
    initial_snapshot:
        Snapshot dataframe containing at least one row with columns matching the format
        produced by ``load_book(..., type=\"snapshot\")``.
    depth:
        Number of levels per side to maintain in the reconstructed book.
    event_column, price_column, volume_column, side_column:
        Column names for event metadata. Adjust these if the input dataframe uses
        different conventions.
    submission_codes, cancel_codes, execution_codes:
        Event codes that should increase or decrease resting volume. Values not listed
        are ignored (e.g., trading halts).

    Returns
    -------
    polars.DataFrame
        Snapshots aligned with the incremental message timestamps.
    """
    if messages.is_empty():
        raise ValueError("Messages dataframe is empty; cannot reconstruct snapshots.")
    if initial_snapshot.is_empty():
        raise ValueError("Initial snapshot dataframe must contain at least one row.")

    required_columns = {event_column, price_column, volume_column, side_column, "tst"}
    missing_cols = required_columns - set(messages.columns)
    if missing_cols:
        raise ValueError(f"Messages missing required columns: {sorted(missing_cols)}")

    if "tst" not in initial_snapshot.columns:
        raise ValueError("Initial snapshot must contain a 'tst' timestamp column.")

    messages = messages.with_row_count("row_nr")

    submission_set = set(submission_codes)
    removal_set = set(cancel_codes) | set(execution_codes)

    def _snapshot_to_books(row: dict) -> tuple[dict[float, float], dict[float, float]]:
        asks: dict[float, float] = {}
        bids: dict[float, float] = {}
        for level_idx in range(depth):
            ask_price = row.get(f"asks[{level_idx}].price")
            ask_amount = row.get(f"asks[{level_idx}].amount")
            bid_price = row.get(f"bids[{level_idx}].price")
            bid_amount = row.get(f"bids[{level_idx}].amount")
            if ask_price is not None and ask_amount is not None and ask_amount > 0:
                asks[float(ask_price)] = asks.get(float(ask_price), 0.0) + float(ask_amount)
            if bid_price is not None and bid_amount is not None and bid_amount > 0:
                bids[float(bid_price)] = bids.get(float(bid_price), 0.0) + float(bid_amount)
        return asks, bids

    def _serialise_levels(ts: int, asks: dict[float, float], bids: dict[float, float]) -> dict:
        row = {"tst": ts}
        ask_levels = sorted(asks.items())
        bid_levels = sorted(bids.items(), key=lambda x: -x[0])
        for idx in range(depth):
            if idx < len(ask_levels):
                price, vol = ask_levels[idx]
                row[f"asks[{idx}].price"] = price
                row[f"asks[{idx}].amount"] = vol
            else:
                row[f"asks[{idx}].price"] = None
                row[f"asks[{idx}].amount"] = 0.0

            if idx < len(bid_levels):
                price, vol = bid_levels[idx]
                row[f"bids[{idx}].price"] = price
                row[f"bids[{idx}].amount"] = vol
            else:
                row[f"bids[{idx}].price"] = None
                row[f"bids[{idx}].amount"] = 0.0
        return row

    def _apply_update(book: dict[float, float], price: float, delta: float) -> None:
        if price <= 0 or abs(delta) == 0:
            return
        updated = book.get(price, 0.0) + delta
        if updated <= 1e-9:
            book.pop(price, None)
        else:
            book[price] = updated

    initial_row = initial_snapshot.head(1).row(0, named=True)
    ask_book, bid_book = _snapshot_to_books(initial_row)
    reconstructed: list[dict] = [_serialise_levels(int(initial_row["tst"]), ask_book, bid_book)]

    initial_ts = int(initial_row["tst"])
    try:
        first_index = (
            messages.filter(pl.col("tst") == initial_ts)
            .select(pl.col("row_nr").min())
            .item()
        )
    except pl.exceptions.ComputeError:
        first_index = messages["row_nr"][0]

    for msg in messages.iter_rows(named=True):
        if msg["row_nr"] <= first_index:
            continue

        event_code = int(msg[event_column])
        price = float(msg[price_column])
        volume = float(msg[volume_column])
        side = "bids" if bool(msg[side_column]) else "asks"

        if event_code in submission_set:
            _apply_update(bid_book if side == "bids" else ask_book, price, volume)
        elif event_code in removal_set:
            _apply_update(bid_book if side == "bids" else ask_book, price, -volume)
        else:
            # ignore other events (halts, etc.)
            pass

        reconstructed.append(_serialise_levels(int(msg["tst"]), ask_book, bid_book))
    schema = {"tst": pl.Int64}
    for idx in range(depth):
        schema[f"asks[{idx}].price"] = pl.Float64
        schema[f"asks[{idx}].amount"] = pl.Float64
        schema[f"bids[{idx}].price"] = pl.Float64
        schema[f"bids[{idx}].amount"] = pl.Float64

    return pl.from_dicts(reconstructed, schema=schema)


__all__ = ["reconstruct_snapshots"]
