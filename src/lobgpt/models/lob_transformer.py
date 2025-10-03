"""
LOB Transformer: GPT-style transformer for limit order book token prediction.

Inspired by:
- GPT architecture (Radford et al., 2019)
- Financial sequence modeling (Wallbridge, 2020)
- LOB-specific adaptations for market microstructure
"""

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class LOBTransformerConfig:
    """Configuration for LOB Transformer."""

    # Model architecture
    vocab_size: int = 10000
    d_model: int = 768
    n_layers: int = 12
    n_heads: int = 12
    d_ff: int = 3072
    max_seq_length: int = 1024
    dropout: float = 0.1

    # Position encoding
    use_learned_pos: bool = True
    use_time_encoding: bool = True
    time_buckets: int = 288  # 5-minute buckets in a day

    # Training
    tie_embeddings: bool = True
    layer_norm_eps: float = 1e-5
    initializer_range: float = 0.02

    # Financial-specific
    regime_conditioning: bool = False
    n_regimes: int = 4  # bull/bear/sideways/volatile


class PositionalEncoding(nn.Module):
    """
    Positional encoding with optional time-aware components.
    """

    def __init__(self, config: LOBTransformerConfig):
        super().__init__()

        if config.use_learned_pos:
            self.pos_embedding = nn.Embedding(config.max_seq_length, config.d_model)
        else:
            # Sinusoidal position encoding
            pe = torch.zeros(config.max_seq_length, config.d_model)
            position = torch.arange(0, config.max_seq_length).unsqueeze(1).float()

            div_term = torch.exp(torch.arange(0, config.d_model, 2).float() *
                               -(math.log(10000.0) / config.d_model))

            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)

            self.register_buffer('pe', pe.unsqueeze(0))

        # Time-of-day encoding for intraday patterns
        if config.use_time_encoding:
            self.time_embedding = nn.Embedding(config.time_buckets, config.d_model)

        self.config = config

    def forward(self, x: torch.Tensor, time_buckets: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: Token embeddings (batch_size, seq_len, d_model)
            time_buckets: Time bucket IDs (batch_size, seq_len)
        """
        seq_len = x.size(1)

        if self.config.use_learned_pos:
            pos_ids = torch.arange(seq_len, device=x.device).unsqueeze(0)
            pos_emb = self.pos_embedding(pos_ids)
        else:
            pos_emb = self.pe[:, :seq_len]

        x = x + pos_emb

        # Add time-of-day information
        if self.config.use_time_encoding and time_buckets is not None:
            time_emb = self.time_embedding(time_buckets)
            x = x + time_emb

        return x


class MultiHeadAttention(nn.Module):
    """
    Multi-head self-attention with causal masking.
    """

    def __init__(self, config: LOBTransformerConfig):
        super().__init__()
        assert config.d_model % config.n_heads == 0

        self.d_model = config.d_model
        self.n_heads = config.n_heads
        self.d_k = config.d_model // config.n_heads

        self.w_q = nn.Linear(config.d_model, config.d_model, bias=False)
        self.w_k = nn.Linear(config.d_model, config.d_model, bias=False)
        self.w_v = nn.Linear(config.d_model, config.d_model, bias=False)
        self.w_o = nn.Linear(config.d_model, config.d_model)

        self.dropout = nn.Dropout(config.dropout)

        # Causal mask
        self.register_buffer('causal_mask',
                           torch.tril(torch.ones(config.max_seq_length, config.max_seq_length)))

    def forward(self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Input tensor (batch_size, seq_len, d_model)
            attention_mask: Optional mask (batch_size, seq_len)

        Returns:
            output: Attention output (batch_size, seq_len, d_model)
            attention_weights: Attention weights (batch_size, n_heads, seq_len, seq_len)
        """
        batch_size, seq_len, _ = x.shape

        # Linear projections
        q = self.w_q(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        k = self.w_k(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        v = self.w_v(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)

        # Scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)

        # Apply causal mask
        causal_mask = self.causal_mask[:seq_len, :seq_len]
        scores = scores.masked_fill(causal_mask == 0, -1e9)

        # Apply attention mask if provided
        if attention_mask is not None:
            scores = scores.masked_fill(attention_mask.unsqueeze(1).unsqueeze(2) == 0, -1e9)

        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)

        # Apply attention to values
        out = torch.matmul(attention_weights, v)
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)

        # Final linear projection
        out = self.w_o(out)

        return out, attention_weights


class FeedForward(nn.Module):
    """Position-wise feed-forward network."""

    def __init__(self, config: LOBTransformerConfig):
        super().__init__()
        self.linear1 = nn.Linear(config.d_model, config.d_ff)
        self.linear2 = nn.Linear(config.d_ff, config.d_model)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(self.dropout(F.gelu(self.linear1(x))))


class TransformerBlock(nn.Module):
    """Single transformer block with pre-norm."""

    def __init__(self, config: LOBTransformerConfig):
        super().__init__()
        self.attention = MultiHeadAttention(config)
        self.feed_forward = FeedForward(config)
        self.norm1 = nn.LayerNorm(config.d_model, eps=config.layer_norm_eps)
        self.norm2 = nn.LayerNorm(config.d_model, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        # Pre-norm self-attention
        normed_x = self.norm1(x)
        attn_out, attn_weights = self.attention(normed_x, attention_mask)
        x = x + self.dropout(attn_out)

        # Pre-norm feed-forward
        normed_x = self.norm2(x)
        ff_out = self.feed_forward(normed_x)
        x = x + self.dropout(ff_out)

        return x, attn_weights


class LOBTransformer(nn.Module):
    """
    GPT-style transformer for LOB token prediction.

    Architecture:
    - Token embedding + positional encoding
    - N transformer blocks with causal self-attention
    - Language modeling head for next-token prediction
    """

    def __init__(self, config: LOBTransformerConfig):
        super().__init__()
        self.config = config

        # Embeddings
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.pos_encoding = PositionalEncoding(config)

        # Regime conditioning (optional)
        if config.regime_conditioning:
            self.regime_embedding = nn.Embedding(config.n_regimes, config.d_model)

        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(config) for _ in range(config.n_layers)
        ])

        # Final layer norm
        self.norm = nn.LayerNorm(config.d_model, eps=config.layer_norm_eps)

        # Language modeling head
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        # Tie embeddings if requested
        if config.tie_embeddings:
            self.lm_head.weight = self.token_embedding.weight

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """Initialize weights following GPT conventions."""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        time_buckets: Optional[torch.Tensor] = None,
        regime_ids: Optional[torch.Tensor] = None,
        return_attention: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            input_ids: Token IDs (batch_size, seq_len)
            attention_mask: Attention mask (batch_size, seq_len)
            time_buckets: Time bucket IDs (batch_size, seq_len)
            regime_ids: Market regime IDs (batch_size,)
            return_attention: Whether to return attention weights

        Returns:
            logits: Next-token logits (batch_size, seq_len, vocab_size)
            attention_weights: Optional attention weights
        """
        batch_size, seq_len = input_ids.shape

        # Token embeddings
        x = self.token_embedding(input_ids)

        # Add regime conditioning
        if self.config.regime_conditioning and regime_ids is not None:
            regime_emb = self.regime_embedding(regime_ids).unsqueeze(1)
            x = x + regime_emb

        # Positional encoding
        x = self.pos_encoding(x, time_buckets)

        # Transformer blocks
        all_attention_weights = []
        for block in self.blocks:
            x, attn_weights = block(x, attention_mask)
            if return_attention:
                all_attention_weights.append(attn_weights)

        # Final norm
        x = self.norm(x)

        # Language modeling head
        logits = self.lm_head(x)

        attention_weights = torch.stack(all_attention_weights) if return_attention else None

        return logits, attention_weights

    @torch.no_grad()
    def generate(
        self,
        prompt_ids: torch.Tensor,
        max_length: int = 100,
        temperature: float = 1.0,
        top_k: Optional[int] = 50,
        top_p: Optional[float] = None,
        do_sample: bool = True
    ) -> torch.Tensor:
        """
        Generate LOB event sequences autoregressively.

        Args:
            prompt_ids: Initial token sequence (batch_size, seq_len)
            max_length: Maximum generation length
            temperature: Sampling temperature
            top_k: Top-k sampling
            top_p: Nucleus sampling
            do_sample: Whether to sample or use greedy decoding

        Returns:
            generated_ids: Generated token sequence (batch_size, seq_len + max_length)
        """
        self.eval()
        batch_size = prompt_ids.size(0)
        generated = prompt_ids.clone()

        for _ in range(max_length):
            # Get logits for next token
            logits, _ = self(generated)
            next_token_logits = logits[:, -1, :] / temperature

            if do_sample:
                # Apply top-k filtering
                if top_k is not None:
                    indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                    next_token_logits[indices_to_remove] = -float('Inf')

                # Apply top-p (nucleus) filtering
                if top_p is not None:
                    sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

                    # Remove tokens with cumulative probability above threshold
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0

                    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                    next_token_logits[indices_to_remove] = -float('Inf')

                # Sample from the filtered distribution
                probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                # Greedy decoding
                next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)

            # Append to sequence
            generated = torch.cat([generated, next_token], dim=-1)

            # Truncate if exceeding max sequence length
            if generated.size(1) > self.config.max_seq_length:
                generated = generated[:, -self.config.max_seq_length:]

        return generated

    def get_num_params(self, non_embedding: bool = True) -> int:
        """Return number of parameters."""
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.token_embedding.weight.numel()
            if hasattr(self.pos_encoding, 'pos_embedding'):
                n_params -= self.pos_encoding.pos_embedding.weight.numel()
        return n_params


# Model size presets
def create_lob_transformer_small(vocab_size: int = 10000) -> LOBTransformer:
    """Create small LOB transformer (~25M parameters)."""
    config = LOBTransformerConfig(
        vocab_size=vocab_size,
        d_model=512,
        n_layers=6,
        n_heads=8,
        d_ff=2048,
        max_seq_length=1024
    )
    return LOBTransformer(config)


def create_lob_transformer_base(vocab_size: int = 10000) -> LOBTransformer:
    """Create base LOB transformer (~110M parameters)."""
    config = LOBTransformerConfig(
        vocab_size=vocab_size,
        d_model=768,
        n_layers=12,
        n_heads=12,
        d_ff=3072,
        max_seq_length=1024
    )
    return LOBTransformer(config)


def create_lob_transformer_large(vocab_size: int = 10000) -> LOBTransformer:
    """Create large LOB transformer (~350M parameters)."""
    config = LOBTransformerConfig(
        vocab_size=vocab_size,
        d_model=1024,
        n_layers=24,
        n_heads=16,
        d_ff=4096,
        max_seq_length=1024
    )
    return LOBTransformer(config)