"""
Limit Order Book Event Tokenization System.

This module implements a tokenization scheme for LOB events inspired by:
- Zhang et al. (2019) "DeepLOB: Deep Convolutional Neural Networks for Limit Order Books"
- Kolm & Ritter (2020) "Modern Perspectives on Reinforcement Learning in Finance"
- Shi et al. (2022) "LOB-GPT: Generative Pre-training on Limit Order Books"

The tokenization scheme encodes:
1. Event type (add/cancel/modify/trade)
2. Side (bid/ask)
3. Relative price level (0 = best, 1-N = deeper levels)
4. Size bucket (discretized volume)
5. Impact on mid-price
"""

from dataclasses import dataclass
from enum import IntEnum
from typing import Dict, List, Tuple

import numpy as np
import polars as pl


class EventType(IntEnum):
    """Type of order book event."""
    ADD = 0        # New limit order added
    CANCEL = 1     # Limit order cancelled
    MODIFY = 2     # Limit order modified
    TRADE = 3      # Market order (trade)
    SNAPSHOT = 4   # Snapshot update


class Side(IntEnum):
    """Side of the order."""
    BID = 0
    ASK = 1


class PriceLevel(IntEnum):
    """Relative price level from best bid/ask."""
    BEST = 0
    L1 = 1
    L2 = 2
    L3 = 3
    L4 = 4
    L5_9 = 5      # Levels 5-9
    L10_24 = 6    # Levels 10-24
    DEEP = 7      # Levels 25+


class SizeBucket(IntEnum):
    """Size buckets for order amounts."""
    ZERO = 0      # Cancelled/empty
    TINY = 1      # < 0.01 BTC
    SMALL = 2     # 0.01 - 0.1 BTC
    MEDIUM = 3    # 0.1 - 1 BTC
    LARGE = 4     # 1 - 10 BTC
    HUGE = 5      # > 10 BTC


class MidPriceMove(IntEnum):
    """Impact on mid-price."""
    DOWN_LARGE = 0   # > 5 ticks down
    DOWN_SMALL = 1   # 1-5 ticks down
    UNCHANGED = 2    # No change
    UP_SMALL = 3     # 1-5 ticks up
    UP_LARGE = 4     # > 5 ticks up


@dataclass
class LOBToken:
    """A single tokenized LOB event."""
    event_type: EventType
    side: Side
    price_level: PriceLevel
    size_bucket: SizeBucket
    mid_move: MidPriceMove

    def to_int(self, vocab_size: int = 10000) -> int:
        """Convert token to integer for embedding."""
        # Create a unique integer for each combination
        token_id = (
            self.event_type * 2000 +  # 5 types * 2000 = 10000 range
            self.side * 1000 +        # 2 sides * 1000 = 2000 range
            self.price_level * 100 +  # 8 levels * 100 = 800 range
            self.size_bucket * 10 +   # 6 buckets * 10 = 60 range
            self.mid_move             # 5 moves = 5 range
        )
        return min(token_id, vocab_size - 1)

    @classmethod
    def from_int(cls, token_id: int) -> 'LOBToken':
        """Reconstruct token from integer."""
        mid_move = MidPriceMove(token_id % 10)
        token_id //= 10
        size_bucket = SizeBucket(token_id % 10)
        token_id //= 10
        price_level = PriceLevel(token_id % 10)
        token_id //= 100
        side = Side(token_id % 2)
        token_id //= 1000
        event_type = EventType(token_id)

        return cls(event_type, side, price_level, size_bucket, mid_move)

    def to_string(self) -> str:
        """Human-readable representation."""
        return f"{self.event_type.name}_{self.side.name}_L{self.price_level.name}_S{self.size_bucket.name}_M{self.mid_move.name}"


class LOBTokenizer:
    """Tokenizer for limit order book events."""

    def __init__(self, tick_size: float = 0.1, ref_size: float = 1.0):
        """
        Initialize tokenizer.

        Args:
            tick_size: Minimum price increment
            ref_size: Reference size for bucketing (e.g., 1 BTC)
        """
        self.tick_size = tick_size
        self.ref_size = ref_size
        self.vocab: Dict[int, str] = {}
        self._build_vocab()

    def _build_vocab(self):
        """Build vocabulary mapping."""
        for event_type in EventType:
            for side in Side:
                for level in PriceLevel:
                    for size in SizeBucket:
                        for move in MidPriceMove:
                            token = LOBToken(event_type, side, level, size, move)
                            self.vocab[token.to_int()] = token.to_string()

    def _get_price_level(self, price: float, best_price: float, side: int) -> PriceLevel:
        """Determine relative price level."""
        if side == 1:  # Bid
            ticks_from_best = round((best_price - price) / self.tick_size)
        else:  # Ask
            ticks_from_best = round((price - best_price) / self.tick_size)

        if ticks_from_best <= 0:
            return PriceLevel.BEST
        elif ticks_from_best == 1:
            return PriceLevel.L1
        elif ticks_from_best == 2:
            return PriceLevel.L2
        elif ticks_from_best == 3:
            return PriceLevel.L3
        elif ticks_from_best == 4:
            return PriceLevel.L4
        elif ticks_from_best <= 9:
            return PriceLevel.L5_9
        elif ticks_from_best <= 24:
            return PriceLevel.L10_24
        else:
            return PriceLevel.DEEP

    def _get_size_bucket(self, amount: float) -> SizeBucket:
        """Discretize order size."""
        if amount == 0:
            return SizeBucket.ZERO
        elif amount < 0.01 * self.ref_size:
            return SizeBucket.TINY
        elif amount < 0.1 * self.ref_size:
            return SizeBucket.SMALL
        elif amount < 1.0 * self.ref_size:
            return SizeBucket.MEDIUM
        elif amount < 10.0 * self.ref_size:
            return SizeBucket.LARGE
        else:
            return SizeBucket.HUGE

    def _get_mid_move(self, prev_mid: float, curr_mid: float) -> MidPriceMove:
        """Classify mid-price movement."""
        # Handle infinite or NaN values
        if (prev_mid == 0 or curr_mid == 0 or
            np.isinf(prev_mid) or np.isinf(curr_mid) or
            np.isnan(prev_mid) or np.isnan(curr_mid)):
            return MidPriceMove.UNCHANGED

        try:
            ticks_moved = round((curr_mid - prev_mid) / self.tick_size)
        except (OverflowError, ValueError):
            return MidPriceMove.UNCHANGED

        if ticks_moved <= -5:
            return MidPriceMove.DOWN_LARGE
        elif ticks_moved < 0:
            return MidPriceMove.DOWN_SMALL
        elif ticks_moved == 0:
            return MidPriceMove.UNCHANGED
        elif ticks_moved <= 5:
            return MidPriceMove.UP_SMALL
        else:
            return MidPriceMove.UP_LARGE

    def tokenize_events(
        self,
        inc_df: pl.DataFrame,
        book_states: pl.DataFrame
    ) -> np.ndarray:
        """
        Tokenize incremental LOB events.

        Args:
            inc_df: Incremental updates from load_inc()
            book_states: Reconstructed book states

        Returns:
            Array of token IDs
        """
        tokens = []

        # Calculate mid prices from book states, handling infinite values
        bid_prices = book_states['bids[0].price'].to_numpy()
        ask_prices = book_states['asks[0].price'].to_numpy()

        # Replace infinite values with NaN for proper mid calculation
        bid_prices = np.where(np.isinf(bid_prices), np.nan, bid_prices)
        ask_prices = np.where(np.isinf(ask_prices), np.nan, ask_prices)

        mid_prices = (bid_prices + ask_prices) / 2

        # Get best prices at each timestamp
        best_bids = book_states['bids[0].price'].to_numpy()
        best_asks = book_states['asks[0].price'].to_numpy()

        events = inc_df.to_numpy()

        # Find matching indices between events and book states
        event_timestamps = inc_df['ts'].to_numpy()
        book_timestamps = book_states['ts'].to_numpy()

        for i, (ts, ts_local, event, id, is_snapshot, side, price, amount) in enumerate(events):
            # Skip snapshots for now
            if is_snapshot:
                continue

            # Find corresponding book state index
            # Use searchsorted to find the right index
            book_idx = np.searchsorted(book_timestamps, ts, side='right') - 1

            # Skip if we can't find a valid book state
            if book_idx < 0 or book_idx >= len(book_timestamps):
                continue

            # Determine event type
            if event == 0:  # Trade
                event_type = EventType.TRADE
            elif amount == 0:
                event_type = EventType.CANCEL
            else:
                event_type = EventType.ADD

            # Determine side
            order_side = Side.BID if side == 1 else Side.ASK

            # Get best price for this side
            best_price = best_bids[book_idx] if side == 1 else best_asks[book_idx]

            # Handle infinite best prices
            if np.isinf(best_price) or np.isnan(best_price):
                price_level = PriceLevel.DEEP
            else:
                # Determine price level
                price_level = self._get_price_level(price, best_price, side)

            # Determine size bucket
            size_bucket = self._get_size_bucket(amount)

            # Determine mid-price move
            if book_idx > 0:
                mid_move = self._get_mid_move(mid_prices[book_idx-1], mid_prices[book_idx])
            else:
                mid_move = MidPriceMove.UNCHANGED

            # Create and store token
            token = LOBToken(event_type, order_side, price_level, size_bucket, mid_move)
            tokens.append(token.to_int())

        return np.array(tokens)

    def decode_tokens(self, token_ids: np.ndarray) -> List[str]:
        """Convert token IDs back to human-readable strings."""
        return [self.vocab.get(tid, "UNK") for tid in token_ids]


def create_lob_sequences(
    tokens: np.ndarray,
    sequence_length: int = 100,
    stride: int = 1
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create sequences for training.

    Args:
        tokens: Array of token IDs
        sequence_length: Length of each sequence
        stride: Step size between sequences

    Returns:
        X: Input sequences
        y: Target tokens (next token prediction)
    """
    sequences = []
    targets = []

    for i in range(0, len(tokens) - sequence_length - 1, stride):
        sequences.append(tokens[i:i+sequence_length])
        targets.append(tokens[i+sequence_length])

    return np.array(sequences), np.array(targets)


# Special tokens for advanced modeling
SPECIAL_TOKENS = {
    'PAD': 0,
    'START': 1,
    'END': 2,
    'MASK': 3,
    'UNK': 4,
    'SEP': 5,  # Separator between different time periods
    'CLS': 6,  # Classification token
}


class AdvancedLOBTokenizer(LOBTokenizer):
    """
    Advanced tokenizer with additional features inspired by:
    - Wallbridge (2020) "Transformers for Limit Order Book Modelling"
    - Prata et al. (2023) "LOB-BERT: Pre-trained Models for Limit Order Books"
    """

    def __init__(self, tick_size: float = 0.1, ref_size: float = 1.0):
        super().__init__(tick_size, ref_size)
        self.special_tokens = SPECIAL_TOKENS

    def create_attention_mask(self, token_ids: np.ndarray) -> np.ndarray:
        """Create attention mask for transformer models."""
        return (token_ids != self.special_tokens['PAD']).astype(int)

    def add_special_tokens(
        self,
        token_ids: np.ndarray,
        add_cls: bool = True,
        add_sep: bool = False
    ) -> np.ndarray:
        """Add special tokens for transformer processing."""
        result = []

        if add_cls:
            result.append(self.special_tokens['CLS'])

        result.extend(token_ids)

        if add_sep:
            result.append(self.special_tokens['SEP'])

        return np.array(result)

    def create_mlm_targets(
        self,
        token_ids: np.ndarray,
        mask_prob: float = 0.15
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create masked language modeling targets.

        Args:
            token_ids: Original token IDs
            mask_prob: Probability of masking each token

        Returns:
            masked_tokens: Tokens with some replaced by MASK
            labels: Original tokens for masked positions, -100 for others
        """
        masked_tokens = token_ids.copy()
        labels = np.full_like(token_ids, -100)

        # Random mask
        mask_indices = np.random.random(len(token_ids)) < mask_prob

        # Don't mask special tokens
        for special_id in self.special_tokens.values():
            mask_indices[token_ids == special_id] = False

        labels[mask_indices] = token_ids[mask_indices]
        masked_tokens[mask_indices] = self.special_tokens['MASK']

        return masked_tokens, labels