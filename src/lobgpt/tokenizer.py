"""
Event tokenization for level-3 limit order book message streams.

This module implements a stateful tokenizer that converts raw order-book
messages—such into discrete tokens.  The design follows recent work on autoregressive
modelling of L3 event streams (e.g., Nagy et al., 2023; Nagy et al., 2025) and
is tailored to the data columns that are common across our loaders:

    - ``tst``: nanosecond timestamp of the event
    - ``event_code``: provider-specific event identifier
    - ``is_buy``: True for bid-side events, False for ask-side events
    - ``prc``: event price
    - ``vol``: event volume

Each message is mapped to a five-dimensional categorical code comprising:

1. Canonical event action
2. Side (bid / ask)
3. Relative price bucket (basis-point move w.r.t. an adaptive reference price)
4. Size bucket (share / contract size in log-spaced bins)
5. Inter-arrival-time bucket (nanosecond gaps on a log scale)

Tokens are then produced by base-N encoding of these components, yielding a
compact vocabulary that can be fed to language-model-style architectures.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Dict, Iterable, Optional, Sequence

import numpy as np
import polars as pl

# ---------------------------------------------------------------------------
# Enumerations describing the categorical factors of the token
# ---------------------------------------------------------------------------


class EventType(IntEnum):
    """Canonicalised event actions across feeds."""

    ADD = 0
    CANCEL = 1
    TRADE = 2
    HALT = 3
    OTHER = 4


class Side(IntEnum):
    """Book side indicator."""

    BID = 0
    ASK = 1


class PriceBucket(IntEnum):
    """
    Basis-point move of the event price relative to the adaptive reference.

    Negative buckets correspond to price deterioration, positive buckets to
    improvements.  ``ABS_GT_50`` captures moves larger than 50 bp in magnitude.
    """

    NEG_GT_50 = 0
    NEG_20_50 = 1
    NEG_10_20 = 2
    NEG_5_10 = 3
    NEG_2_5 = 4
    NEG_1_2 = 5
    NEG_0_1 = 6
    AT_REFERENCE = 7
    POS_0_1 = 8
    POS_1_2 = 9
    POS_2_5 = 10
    POS_5_10 = 11
    POS_10_20 = 12
    POS_20_50 = 13
    POS_GT_50 = 14


class SizeBucket(IntEnum):
    """Order size expressed in log-spaced buckets."""

    ZERO = 0            # 0
    MICRO = 1           # 0 < size <= 10
    SMALL = 2           # 10 < size <= 50
    MEDIUM = 3          # 50 < size <= 100
    LARGE = 4           # 100 < size <= 250
    XLARGE = 5          # 250 < size <= 500
    XXLARGE = 6         # 500 < size <= 1000
    BLOCK = 7           # > 1000


class TimeBucket(IntEnum):
    """Inter-arrival time bucket (nanoseconds)."""

    LEQ_1_US = 0
    LEQ_10_US = 1
    LEQ_100_US = 2
    LEQ_1_MS = 3
    LEQ_5_MS = 4
    LEQ_10_MS = 5
    LEQ_100_MS = 6
    LEQ_1_S = 7
    GT_1_S = 8


# ---------------------------------------------------------------------------
# Token structure
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TokenComponents:
    """Human-readable view of a tokenised message."""

    event_type: EventType
    side: Side
    price_bucket: PriceBucket
    size_bucket: SizeBucket
    time_bucket: TimeBucket

    def to_tuple(self) -> tuple[int, int, int, int, int]:
        return (
            int(self.event_type),
            int(self.side),
            int(self.price_bucket),
            int(self.size_bucket),
            int(self.time_bucket),
        )

    def to_string(self) -> str:
        """Compact string representation for logging/debugging."""
        return "|".join(
            [
                self.event_type.name,
                self.side.name,
                self.price_bucket.name,
                self.size_bucket.name,
                self.time_bucket.name,
            ]
        )

    def __str__(self) -> str:  # pragma: no cover - convenience formatting
        return self.to_string()

    def __repr__(self) -> str:  # pragma: no cover - convenience formatting
        return f"TokenComponents({self.to_string()})"

class LOBTokenizer:
    """
    Stateful tokenizer for level-3 order book message streams.

    Parameters
    ----------
    price_reference_alpha:
        Exponential decay factor used when updating the adaptive price
        reference.  Values in (0, 1]; higher values react faster.
    price_bucket_bounds:
        Sorted list of basis-point thresholds (negative to positive) that
        define the bucket edges.  The default corresponds to:

        ``[-inf, -50, -20, -10, -5, -2, -1, -0.5, 0.5, 1, 2, 5, 10, 20, 50, inf)``
    size_bucket_bounds:
        Monotonic sequence of size thresholds (in raw units) delimiting the
        MICRO → BLOCK buckets.
    time_bucket_bounds_ns:
        Monotonic sequence of inter-arrival thresholds expressed in
        nanoseconds.
    """

    LOBSTER_EVENT_MAP: Dict[int, EventType] = {
        1: EventType.ADD,       # submission
        2: EventType.CANCEL,    # deletion
        3: EventType.CANCEL,    # partial cancel
        4: EventType.TRADE,     # visible execution
        5: EventType.TRADE,     # hidden execution
        6: EventType.TRADE,     # cross trade
        7: EventType.HALT,      # trading halt
    }

    MIC_EVENT_MAP: Dict[int, EventType] = {
        1: EventType.ADD,
        2: EventType.CANCEL,
        3: EventType.TRADE,
    }

    def __init__(
        self,
        price_reference_alpha: float = 0.1,
        price_bucket_bounds: Optional[Sequence[float]] = None,
        size_bucket_bounds: Optional[Sequence[float]] = None,
        time_bucket_bounds_ns: Optional[Sequence[int]] = None,
    ) -> None:
        if not 0 < price_reference_alpha <= 1:
            raise ValueError("price_reference_alpha must be in (0, 1]")

        self.price_reference_alpha = price_reference_alpha

        # Default bucket boundaries
        if price_bucket_bounds is None:
            price_bucket_bounds = [
                -50.0,
                -20.0,
                -10.0,
                -5.0,
                -2.0,
                -1.0,
                -0.5,
                0.5,
                1.0,
                2.0,
                5.0,
                10.0,
                20.0,
                50.0,
            ]
        if size_bucket_bounds is None:
            size_bucket_bounds = [0.0, 10.0, 50.0, 100.0, 250.0, 500.0, 1000.0]
        if time_bucket_bounds_ns is None:
            time_bucket_bounds_ns = [
                1_000,         # 1 µs
                10_000,        # 10 µs
                100_000,       # 100 µs
                1_000_000,     # 1 ms
                5_000_000,     # 5 ms
                10_000_000,    # 10 ms
                100_000_000,   # 100 ms
                1_000_000_000, # 1 s
            ]

        self.price_bucket_bounds = np.asarray(price_bucket_bounds, dtype=np.float64)
        self.size_bucket_bounds = np.asarray(size_bucket_bounds, dtype=np.float64)
        self.time_bucket_bounds = np.asarray(time_bucket_bounds_ns, dtype=np.int64)

        if (
            not np.all(np.diff(self.price_bucket_bounds) > 0)
            or not np.all(np.diff(self.size_bucket_bounds) >= 0)
            or not np.all(np.diff(self.time_bucket_bounds) > 0)
        ):
            raise ValueError("Bucket bounds must be strictly increasing.")

        self.event_cardinality = len(EventType)
        self.side_cardinality = len(Side)
        self.price_cardinality = len(PriceBucket)
        self.size_cardinality = len(SizeBucket)
        self.time_cardinality = len(TimeBucket)
        self.vocab_size = (
            self.event_cardinality
            * self.side_cardinality
            * self.price_cardinality
            * self.size_cardinality
            * self.time_cardinality
        )

        self.vocab = self._build_vocab_strings()
        self.id_to_token = self.vocab

        self._timestamp_col: Optional[str] = None
        self._event_type_col: Optional[str] = None
        self._side_col: Optional[str] = None
        self._price_col: Optional[str] = None
        self._volume_col: Optional[str] = None

        self.reset_state()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset_state(self) -> None:
        """Reset adaptive statistics for a new message stream."""
        self._reference_price: Optional[float] = None
        self._last_timestamp: Optional[int] = None

    def tokenize(self, events: pl.DataFrame) -> np.ndarray:
        """
        Convert an event stream into integer tokens.

        Parameters
        ----------
        events:
            Polars DataFrame containing at least the columns listed in the
            module docstring.
        """
        if events.is_empty():
            return np.empty(0, dtype=np.int32)

        self._prepare_columns(events.columns)
        self.reset_state()

        tokens = np.empty(events.height, dtype=np.int32)
        for idx, row in enumerate(events.iter_rows(named=True)):
            tokens[idx] = self._encode_row(row)
        return tokens

    def decode_token(self, token_id: int) -> TokenComponents:
        """Invert a token id back to its categorical components."""
        if not 0 <= token_id < self.vocab_size:
            raise ValueError(f"token_id {token_id} outside vocabulary (size={self.vocab_size})")

        remainder = token_id
        time_bucket = remainder % self.time_cardinality
        remainder //= self.time_cardinality

        size_bucket = remainder % self.size_cardinality
        remainder //= self.size_cardinality

        price_bucket = remainder % self.price_cardinality
        remainder //= self.price_cardinality

        side = remainder % self.side_cardinality
        remainder //= self.side_cardinality

        event_type = remainder % self.event_cardinality

        return TokenComponents(
            event_type=EventType(event_type),
            side=Side(side),
            price_bucket=PriceBucket(price_bucket),
            size_bucket=SizeBucket(size_bucket),
            time_bucket=TimeBucket(time_bucket),
        )

    def decode_tokens(self, tokens: Iterable[int]) -> list[TokenComponents]:
        """Vectorised decode convenience wrapper."""
        return [self.decode_token(token) for token in tokens]

    def encode_token(self, components: TokenComponents) -> int:
        """Convert token components back to an integer id."""
        return self._compose_token(
            components.event_type,
            components.side,
            components.price_bucket,
            components.size_bucket,
            components.time_bucket,
        )

    def encode_tokens(self, components_list: Iterable[TokenComponents]) -> np.ndarray:
        """Vectorised encoder for sequences of token components."""
        return np.fromiter(
            (self.encode_token(comp) for comp in components_list),
            dtype=np.int32,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _prepare_columns(self, columns: Sequence[str]) -> None:
        """Identify canonical column names in the supplied frame."""
        self._timestamp_col = self._find_column(columns, ("tst", "ts", "timestamp"))
        self._event_type_col = self._find_column(columns, ("event_type", "event_code", "type"))
        self._side_col = self._find_column(columns, ("is_buy", "side", "direction"))
        self._price_col = self._find_column(columns, ("prc", "price"))
        self._volume_col = self._find_column(columns, ("vol", "size", "amount"))

        missing = [
            name
            for name, col in {
                "timestamp": self._timestamp_col,
                "event_type": self._event_type_col,
                "side": self._side_col,
                "price": self._price_col,
                "volume": self._volume_col,
            }.items()
            if col is None
        ]
        if missing:
            raise ValueError(f"Required columns not found: {missing}")

    @staticmethod
    def _find_column(columns: Sequence[str], candidates: Sequence[str]) -> Optional[str]:
        lower = {col.lower(): col for col in columns}
        for name in candidates:
            if name.lower() in lower:
                return lower[name.lower()]
        return None

    def _encode_row(self, row: Dict[str, object]) -> int:
        ts_ns = int(row[self._timestamp_col])  # type: ignore[arg-type]
        event_raw = row[self._event_type_col]  # type: ignore[index]
        side_raw = row[self._side_col]         # type: ignore[index]
        price = float(row[self._price_col])    # type: ignore[arg-type]
        volume = float(row[self._volume_col])  # type: ignore[arg-type]

        event_type = self._canonical_event_type(event_raw)
        side = self._resolve_side(side_raw)
        price_bucket = self._bucket_price(price)
        size_bucket = self._bucket_size(volume)
        time_bucket = self._bucket_time(ts_ns)

        token_id = self._compose_token(event_type, side, price_bucket, size_bucket, time_bucket)

        self._update_reference_price(price)
        self._last_timestamp = ts_ns
        return token_id

    def _compose_token(
        self,
        event_type: EventType,
        side: Side,
        price_bucket: PriceBucket,
        size_bucket: SizeBucket,
        time_bucket: TimeBucket,
    ) -> int:
        token = int(event_type)
        token = token * self.side_cardinality + int(side)
        token = token * self.price_cardinality + int(price_bucket)
        token = token * self.size_cardinality + int(size_bucket)
        token = token * self.time_cardinality + int(time_bucket)
        return token

    def _build_vocab_strings(self) -> Dict[int, TokenComponents]:
        vocab: Dict[int, TokenComponents] = {}
        for event in EventType:
            for side in Side:
                for price in PriceBucket:
                    for size in SizeBucket:
                        for time in TimeBucket:
                            token_id = self._compose_token(event, side, price, size, time)
                            vocab[token_id] = TokenComponents(
                                event_type=event,
                                side=side,
                                price_bucket=price,
                                size_bucket=size,
                                time_bucket=time,
                            )
        return vocab

    def _canonical_event_type(self, raw: object) -> EventType:
        if raw is None:
            return EventType.OTHER

        if isinstance(raw, (np.integer, int)):
            value = int(raw)
            if value in self.LOBSTER_EVENT_MAP:
                return self.LOBSTER_EVENT_MAP[value]
            if value in self.MIC_EVENT_MAP:
                return self.MIC_EVENT_MAP[value]
            if value == 0:
                return EventType.OTHER
            return EventType.OTHER

        if isinstance(raw, float) and raw.is_integer():
            return self._canonical_event_type(int(raw))

        if isinstance(raw, str):
            key = raw.strip().lower()
            if key in {"a", "add", "submission", "insert", "new"}:
                return EventType.ADD
            if key in {"d", "del", "delete", "cancel", "remove"}:
                return EventType.CANCEL
            if key in {"t", "trade", "exec", "execution"}:
                return EventType.TRADE
            if key in {"halt", "pause"}:
                return EventType.HALT
            return EventType.OTHER

        return EventType.OTHER

    @staticmethod
    def _resolve_side(raw: object) -> Side:
        if isinstance(raw, bool):
            return Side.BID if raw else Side.ASK
        if isinstance(raw, (np.integer, int)):
            value = int(raw)
            if value in (1,):
                return Side.BID
            if value in (-1,):
                return Side.ASK
            if value == 0:
                return Side.BID
        if isinstance(raw, float) and raw.is_integer():
            return LOBTokenizer._resolve_side(int(raw))
        if isinstance(raw, str):
            key = raw.strip().lower()
            if key in {"b", "bid", "buy"}:
                return Side.BID
            if key in {"a", "ask", "sell"}:
                return Side.ASK
        # Default to bid to avoid negative indexing
        return Side.BID

    def _bucket_price(self, price: float) -> PriceBucket:
        if price <= 0 or not np.isfinite(price):
            delta_bps = 0.0
        elif self._reference_price is None or self._reference_price <= 0:
            delta_bps = 0.0
        else:
            delta_bps = 10_000.0 * (price / self._reference_price - 1.0)

        idx = int(np.searchsorted(self.price_bucket_bounds, delta_bps, side="right"))
        idx = min(max(idx, 0), self.price_cardinality - 1)
        return PriceBucket(idx)

    def _bucket_size(self, volume: float) -> SizeBucket:
        if volume <= 0 or not np.isfinite(volume):
            return SizeBucket.ZERO
        idx = int(np.searchsorted(self.size_bucket_bounds, volume, side="right"))
        idx = min(max(idx, 0), self.size_cardinality - 1)
        return SizeBucket(idx)

    def _bucket_time(self, ts_ns: int) -> TimeBucket:
        if self._last_timestamp is None:
            delta = 0
        else:
            delta = max(ts_ns - self._last_timestamp, 0)
        idx = int(np.searchsorted(self.time_bucket_bounds, delta, side="right"))
        idx = min(max(idx, 0), self.time_cardinality - 1)
        return TimeBucket(idx)

    def _update_reference_price(self, price: float) -> None:
        if not np.isfinite(price) or price <= 0:
            return
        if self._reference_price is None:
            self._reference_price = price
        else:
            self._reference_price = (
                self.price_reference_alpha * price
                + (1.0 - self.price_reference_alpha) * self._reference_price
            )


__all__ = [
    "EventType",
    "Side",
    "PriceBucket",
    "SizeBucket",
    "TimeBucket",
    "TokenComponents",
    "LOBTokenizer",
]
    ...
