# L3 Event Tokenization

The goal is to provide a compact, information-rich vocabulary that can be consumed by transformer models while
remaining agnostic to feed-specific quirks.

The design takes inspiration from recent autoregressive modelling work on L3
order flows (e.g., Nagy *et al.* 2023; Nagy *et al.* 2025) and compresses each
message into five categorical components:

| Component          | Description                                                     | Cardinality |
| ------------------ | --------------------------------------------------------------- | ----------- |
| Event action       | Canonical operation across feeds (`ADD`, `CANCEL`, `TRADE`, …)  | 5           |
| Side               | Bid/ask flag derived from directional fields                    | 2           |
| Price bucket       | Basis-point move relative to an adaptive reference price        | 15          |
| Size bucket        | Log-spaced volume buckets (shares, contracts, lots)             | 8           |
| Time bucket        | Inter-arrival time bucket on a log scale (ns → seconds)         | 9           |

The resulting vocabulary size is `5 × 2 × 15 × 8 × 9 = 10,800` tokens.

## Canonical event actions

Provider event codes are mapped to the following canonical set:

| Canonical code | LOBSTER events (raw)                 | MIC events (raw) | Semantics                                   |
| -------------- | ------------------------------------ | ---------------- | ------------------------------------------- |
| `ADD`          | `1` (submission)                     | `1`              | Liquidity provision (limit order add)       |
| `CANCEL`       | `2`, `3` (deletion/partial cancel)   | `2`              | Liquidity removal without execution         |
| `TRADE`        | `4`, `5`, `6` (executions/crosses)   | `3` *(optional)* | Liquidity taken or internalised             |
| `HALT`         | `7`                                  | —                | Trading halt / auction pause                |
| `OTHER`        | any unrecognised code                | any other value  | Catch-all for feed-specific messages        |

String-valued feeds (`"ADD"`, `"DEL"`, `"TRADE"`, etc.) are handled by simple
case-insensitive heuristics.

## Side

The tokenizer expects a boolean or categorical column indicating whether the
event applies to the bid or ask side.  The convention is:

* truthy / positive values (`True`, `1`, `"bid"`, `"buy"`) → `Side.BID`
* falsy / negative values (`False`, `-1`, `"ask"`, `"sell"`) → `Side.ASK`

When the column is missing or ambiguous we default to `Side.BID` to avoid
negative indices, although downstream datasets should always supply the flag.

## Adaptive price buckets

To avoid hard-coded instrument-specific tick ladders, the tokenizer maintains
an **adaptive reference price** \\( \\hat{p}_t \\) updated via an exponential
moving average:

\\[
\\hat{p}_t = \\alpha p_t + (1-\\alpha) \\hat{p}_{t-1}, \\quad \\alpha \\in (0,1].
\\]

Using \\( \\hat{p}_t \\) the price component is expressed as a basis-point move

\\[
\\Delta_{bp} = 10^4 \\left( \\frac{p_t}{\\hat{p}_{t-1}} - 1 \\right),
\\]

and bucketed using symmetric thresholds:

```
(-inf, -50], (-50, -20], (-20, -10], (-10, -5], (-5, -2],
(-2, -1], (-1, -0.5], (-0.5, 0.5], (0.5, 1], (1, 2],
(2, 5], (5, 10], (10, 20], (20, 50], (50, inf)
```

These map directly to the `PriceBucket` enum:

```
[NEG_GT_50, NEG_20_50, …, AT_REFERENCE, …, POS_20_50, POS_GT_50]
```

## Size buckets

Order sizes (shares/contracts) are discretised into log-spaced buckets:

| Bucket    | Range (inclusive of upper bound) |
| --------- | --------------------------------- |
| `ZERO`    | size ≤ 0                          |
| `MICRO`   | 0 < size ≤ 10                     |
| `SMALL`   | 10 < size ≤ 50                    |
| `MEDIUM`  | 50 < size ≤ 100                   |
| `LARGE`   | 100 < size ≤ 250                  |
| `XLARGE`  | 250 < size ≤ 500                  |
| `XXLARGE` | 500 < size ≤ 1000                 |
| `BLOCK`   | size > 1000                       |

The thresholds can be overridden when constructing the tokenizer to suit a
particular asset class.

## Time buckets

Inter-arrival times are measured in nanoseconds.  The gap to the previous
event is assigned to one of nine exponentially expanding buckets:

```
[0, 1 µs], (1 µs, 10 µs], (10 µs, 100 µs], (100 µs, 1 ms],
(1 ms, 5 ms], (5 ms, 10 ms], (10 ms, 100 ms], (100 ms, 1 s], > 1 s
```

These correspond to `TimeBucket.LEQ_1_US` through `TimeBucket.GT_1_S`, providing
explicit timing structure for the language model.

## Token composition

Each component is converted to an integer via mixed-radix encoding:

```
token =
    ((((event * 2 + side) * 15 + price) * 8 + size) * 9 + time)
```

This ensures a dense, collision-free vocabulary.  The inverse mapping is
implemented by `LOBTokenizer.decode_token`.

## Usage example

```python
import polars as pl
from lobgpt.hdb import get_dataset
from lobgpt.tokenizer import LOBTokenizer

# Load MIC or LOBSTER messages (columns: tst, event_type, is_buy, prc, vol, …)
dl = get_dataset("lobster")
msgs = dl.load_book("AAPL", ["120621.130000", "120621.133000"])

tokenizer = LOBTokenizer(price_reference_alpha=0.1)
tokens = tokenizer.tokenize(msgs)

decoded = tokenizer.decode_token(int(tokens[0]))
print(decoded.to_string())
# ADD|BID|NEG_0_1|SMALL|LEQ_1_US
```

The tokenizer exposes:

* `tokenizer.vocab_size` – integer vocabulary cardinality
* `tokenizer.vocab` – dictionary of token id → `TokenComponents`
* `tokenizer.decode_token(token_id)` – recover structured components
* `tokenizer.encode_token(components)` – obtain the integer id for given components
* `tokenizer.reset_state()` – reset adaptive statistics when starting a new day

## Design rationale

* **Feed agnostic:** normalises MIC and LOBSTER event codes into a shared
  canonical space.
* **Scale invariance:** relative price buckets and log-spaced size bins allow a
  single tokenizer to operate across products with vastly different price
  levels and liquidity.
* **Temporal context:** explicit time-bucket dimension captures the bursty
  nature of order flow without requiring raw timestamps.
* **Compact vocabulary:** 10.8k tokens strike a balance between expressiveness
  and parameter efficiency for transformer models.

## References

* Nagy, P., et al. (2023). *Model-based Reinforcement Learning for Limit Order
  Book Markets.* arXiv:2309.00638.
* Nagy, P., et al. (2025). *Generative Pre-Training on Limit Order Books.* arXiv:2502.09172.
* Huang, W., & Polak, L. (2011). *LOBSTER: Limit Order Book Reconstruction System.*
