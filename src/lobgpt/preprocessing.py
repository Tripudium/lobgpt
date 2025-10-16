"""
Pre-processing utilities for LOB message streams.
"""

from __future__ import annotations

from typing import Dict, Iterable, Sequence

import polars as pl


class OrderBookState:
    """Simple price-level order book for reconstruction."""

    def __init__(self, depth: int):
        self.depth = depth
        self.asks: Dict[float, float] = {}
        self.bids: Dict[float, float] = {}
        self.orders: Dict[int, Dict[str, float]] = {}

    def best_mid_price(self, fallback: float) -> float:
        ask_prices = [price for price, vol in self.asks.items() if vol > 0]
        bid_prices = [price for price, vol in self.bids.items() if vol > 0]

        if ask_prices and bid_prices:
            return (min(ask_prices) + max(bid_prices)) / 2
        if ask_prices:
            return min(ask_prices)
        if bid_prices:
            return max(bid_prices)
        return fallback

    def _book_for_side(self, is_buy: bool) -> Dict[float, float]:
        return self.bids if is_buy else self.asks

    def _apply_volume(self, book: Dict[float, float], price: float, delta: float) -> None:
        if price <= 0 or delta == 0:
            return
        updated = book.get(price, 0.0) + delta
        if updated <= 1e-9:
            book.pop(price, None)
        else:
            book[price] = updated

        if len(book) > self.depth * 4:  # prevent runaway growth
            if book is self.asks:
                sorted_prices = sorted(book.keys())[: self.depth]
            else:
                sorted_prices = sorted(book.keys(), reverse=True)[: self.depth]
            book_keys = set(sorted_prices)
            for price in list(book.keys()):
                if price not in book_keys:
                    book.pop(price, None)

    def add_order(self, order_id: int, price: float, volume: float, is_buy: bool, timestamp: int) -> None:
        if order_id == 0:
            return
        self.orders[order_id] = {
            "price": price,
            "volume": volume,
            "is_buy": is_buy,
            "timestamp": timestamp,
        }
        self._apply_volume(self._book_for_side(is_buy), price, volume)

    def remove_volume(self, order_id: int, delta: float) -> Dict[str, float] | None:
        order = self.orders.get(order_id)
        if not order:
            return None
        delta = min(delta, order["volume"])
        order["volume"] -= delta
        self._apply_volume(self._book_for_side(bool(order["is_buy"])), order["price"], -delta)
        if order["volume"] <= 1e-9:
            return self.orders.pop(order_id)
        return order

    def get_order(self, order_id: int) -> Dict[str, float] | None:
        return self.orders.get(order_id)


def preprocess_messages_for_tokenization(
    messages: pl.DataFrame,
    *,
    tick_size: float,
    event_column: str = "event_code",
    order_id_column: str = "order_id",
    price_column: str = "prc",
    volume_column: str = "vol",
    side_column: str = "is_buy",
    timestamp_column: str = "tst",
    submission_codes: Sequence[int] = (1,),
    cancel_codes: Sequence[int] = (2,),
    deletion_codes: Sequence[int] = (3,),
    execution_codes: Sequence[int] = (4, 5),
) -> pl.DataFrame:
    """
    Convert raw incremental messages into the feature representation described in Nagy et al.

    Each output row contains:
        - event_code: original message type
        - side: 1 for buy, -1 for sell
        - price_offset_ticks: price offset from mid-price expressed in ticks
        - size: reported message size
        - inter_arrival_ns: inter-arrival time in nanoseconds
        - prev_price, prev_size, prev_timestamp: attributes of referenced order (if any)
        - order_id: carried forward for downstream processing

    Returns
    -------
    pl.DataFrame
        Normalised message features ready for tokenisation.
    """
    required = {event_column, order_id_column, price_column, volume_column, side_column, timestamp_column}
    missing = required - set(messages.columns)
    if missing:
        raise ValueError(f"Messages missing required columns: {sorted(missing)}")

    if messages.is_empty():
        return pl.DataFrame(
            schema={
                timestamp_column: pl.Int64,
                event_column: pl.Int32,
                "side": pl.Int8,
                "price_offset_ticks": pl.Int32,
                "size": pl.Float64,
                "inter_arrival_ns": pl.Int64,
                "prev_price": pl.Float64,
                "prev_size": pl.Float64,
                "prev_timestamp": pl.Int64,
                order_id_column: pl.Int64,
            }
        )

    submission_set = set(submission_codes)
    cancel_set = set(cancel_codes)
    deletion_set = set(deletion_codes)
    execution_set = set(execution_codes)
    removal_set = cancel_set | deletion_set | execution_set

    book = OrderBookState(depth=100)

    records: list[dict] = []
    last_timestamp: int | None = None

    sorted_messages = messages.sort(timestamp_column)

    for row in sorted_messages.iter_rows(named=True):
        ts = int(row[timestamp_column])
        event_code = int(row[event_column])
        order_id = int(row[order_id_column]) if order_id_column in row and row[order_id_column] is not None else 0
        is_buy = bool(row[side_column])
        size = float(row[volume_column]) if row[volume_column] is not None else 0.0
        price_from_msg = float(row[price_column]) if row[price_column] is not None else 0.0

        inter_arrival = 0 if last_timestamp is None else max(ts - last_timestamp, 0)
        last_timestamp = ts

        stored_order = book.get_order(order_id)
        prev_price = stored_order["price"] if stored_order else None
        prev_size = stored_order["volume"] if stored_order else None
        prev_timestamp = stored_order["timestamp"] if stored_order else None

        reference_price = price_from_msg if price_from_msg > 0 else (prev_price if prev_price else 0.0)
        mid_price = book.best_mid_price(reference_price)

        if tick_size <= 0:
            price_offset_ticks = 0
        else:
            price_offset_ticks = int(round((reference_price - mid_price) / tick_size))

        record = {
            timestamp_column: ts,
            event_column: event_code,
            "side": 1 if is_buy else -1,
            "price_offset_ticks": price_offset_ticks,
            "size": size,
            "inter_arrival_ns": inter_arrival,
            "prev_price": prev_price,
            "prev_size": prev_size,
            "prev_timestamp": prev_timestamp,
            order_id_column: order_id,
        }
        records.append(record)

        # Apply book update --------------------------------------------
        if event_code in submission_set:
            effective_price = reference_price
            book.add_order(order_id, effective_price, size, is_buy, ts)
        elif event_code in cancellation_set:
            if stored_order:
                book.remove_volume(order_id, size)
        elif event_code in deletion_set:
            if stored_order:
                book.remove_volume(order_id, size)
        elif event_code in execution_set:
            if stored_order:
                book.remove_volume(order_id, size)
        # Other event types are ignored

    schema = {
        timestamp_column: pl.Int64,
        event_column: pl.Int32,
        "side": pl.Int8,
        "price_offset_ticks": pl.Int32,
        "size": pl.Float64,
        "inter_arrival_ns": pl.Int64,
        "prev_price": pl.Float64,
        "prev_size": pl.Float64,
        "prev_timestamp": pl.Int64,
        order_id_column: pl.Int64,
    }

    return pl.from_dicts(records, schema=schema)


__all__ = ["preprocess_messages_for_tokenization"]
