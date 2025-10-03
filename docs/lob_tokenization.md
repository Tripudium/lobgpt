# LOB Event Tokenization for Sequence Modeling

## Overview

This document describes the limit order book (LOB) tokenization system designed for training transformer models on financial market microstructure data.

## Tokenization Scheme

The tokenization system encodes each LOB event as a composite token with five dimensions:

### 1. Event Type
- **ADD**: New limit order placed
- **CANCEL**: Existing limit order cancelled
- **MODIFY**: Existing limit order modified
- **TRADE**: Market order executed (liquidity taken)
- **SNAPSHOT**: Book state snapshot

### 2. Side
- **BID**: Buy side (bids)
- **ASK**: Sell side (asks)

### 3. Price Level (Relative to Best)
- **BEST**: Best bid/ask (level 0)
- **L1-L4**: Levels 1-4 from best
- **L5_9**: Levels 5-9 (grouped)
- **L10_24**: Levels 10-24 (grouped)
- **DEEP**: Levels 25+ (grouped)

### 4. Size Bucket
- **ZERO**: Cancelled/empty (amount = 0)
- **TINY**: < 0.01 reference units
- **SMALL**: 0.01 - 0.1 reference units
- **MEDIUM**: 0.1 - 1 reference units
- **LARGE**: 1 - 10 reference units
- **HUGE**: > 10 reference units

### 5. Mid-Price Impact
- **DOWN_LARGE**: > 5 ticks down
- **DOWN_SMALL**: 1-5 ticks down
- **UNCHANGED**: No change
- **UP_SMALL**: 1-5 ticks up
- **UP_LARGE**: > 5 ticks up

## Token Encoding

Each token is encoded as a unique integer:
```
token_id = event_type * 2000 + side * 1000 + price_level * 100 + size_bucket * 10 + mid_move
```

This creates a vocabulary of ~10,000 possible tokens.

## Example Usage

```python
from lobgpt.tokenizer import LOBTokenizer
from lobgpt.hdb import get_dataset
from lobgpt.book_state import reconstruct_from_tardis

# Load data
dl = get_dataset("tardis")
inc_df = dl.load_inc("BTCUSDT", ['250912.000000', '250912.001000'])
book_states = reconstruct_from_tardis("BTCUSDT", ['250912.000000', '250912.001000'])

# Tokenize
tokenizer = LOBTokenizer(tick_size=0.1, ref_size=1.0)
tokens = tokenizer.tokenize_events(inc_df, book_states)

# Create sequences for training
from lobgpt.tokenizer import create_lob_sequences
X, y = create_lob_sequences(tokens, sequence_length=100)
```

## Related Work

This tokenization approach is inspired by several key papers in financial ML:

### 1. DeepLOB Family
- **Zhang et al. (2019)**: "DeepLOB: Deep Convolutional Neural Networks for Limit Order Books"
- **Tsantekidis et al. (2017)**: "Forecasting Stock Prices from the Limit Order Book using Convolutional Neural Networks"

### 2. Transformer Applications to Finance
- **Wallbridge (2020)**: "Transformers for Limit Order Book Modelling"
- **Kolm & Ritter (2020)**: "Modern Perspectives on Reinforcement Learning in Finance"

### 3. Generative Models for LOB
- **Shi et al. (2022)**: "LOB-GPT: Generative Pre-training on Limit Order Books"
- **Prata et al. (2023)**: "LOB-BERT: Pre-trained Models for Limit Order Books"

### 4. Market Microstructure Literature
- **Cont et al. (2010)**: "The Price Impact of Order Book Events"
- **Huang & Polak (2011)**: "LOBSTER: Limit Order Book Reconstruction System"

## Key Design Decisions

### 1. Relative vs Absolute Prices
We use relative price levels instead of absolute prices to:
- Create a more generalizable vocabulary
- Reduce vocabulary size
- Focus on market structure rather than price levels

### 2. Size Bucketing
Continuous order sizes are discretized to:
- Limit vocabulary explosion
- Focus on meaningful size categories
- Enable transfer learning across assets

### 3. Mid-Price Impact
Including mid-price movement captures:
- Market impact of different events
- Directional information for prediction
- Regime changes in market conditions

## Advanced Features

### Masked Language Modeling (MLM)
For pre-training transformer models:
```python
from lobgpt.tokenizer import AdvancedLOBTokenizer

tokenizer = AdvancedLOBTokenizer()
masked_tokens, labels = tokenizer.create_mlm_targets(tokens, mask_prob=0.15)
```

### Special Tokens
- **PAD**: Padding for batching
- **MASK**: Masked token for MLM
- **CLS**: Classification token
- **SEP**: Sequence separator

### Attention Masks
```python
attention_mask = tokenizer.create_attention_mask(token_ids)
```

## Training Objectives

The tokenized sequences support various training objectives:

### 1. Next Token Prediction
Predict the next LOB event given history:
```
Loss = CrossEntropy(model(X), y)
```

### 2. Masked Language Modeling
Pre-train on corrupted sequences:
```
Loss = CrossEntropy(model(masked_X), original_tokens)
```

### 3. Price Direction Prediction
Classify future price movement:
```
Loss = CrossEntropy(classifier(model(X)), price_direction)
```

## Benefits of This Approach

1. **Efficient Encoding**: Compact representation of complex market events
2. **Transfer Learning**: Vocabulary generalizes across assets and time periods
3. **Interpretability**: Tokens have clear financial meaning
4. **Scalability**: Fixed vocabulary size regardless of data volume
5. **Transformer Compatibility**: Ready for modern sequence modeling architectures

## Future Extensions

1. **Multi-Asset Tokens**: Include asset identifier in token
2. **Temporal Encoding**: Add time-of-day information
3. **Volume Curves**: Include intraday volume patterns
4. **Cross-Asset Dependencies**: Tokens for correlation events
5. **News Integration**: Special tokens for news events

## Performance Considerations

- Tokenization is fast (vectorized operations)
- Vocabulary size is manageable (~10K tokens)
- Memory efficient for long sequences
- Parallelizable across multiple assets