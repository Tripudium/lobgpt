# Transformer Architecture for LOB Token Prediction

## Overview

This document outlines the implementation of a GPT-style transformer for predicting limit order book (LOB) events, treating market microstructure as a language modeling problem.

## 1. Architecture Design

### 1.1 Core Transformer Components

```python
class LOBTransformer(nn.Module):
    """
    GPT-style transformer for LOB token prediction.

    Architecture:
    - Token Embedding (vocab_size -> d_model)
    - Positional Encoding (learnable or sinusoidal)
    - N Transformer Blocks
    - Output Projection (d_model -> vocab_size)
    """

    Components:
    - Token Embeddings: Maps token IDs to dense vectors
    - Position Embeddings: Encodes temporal/sequential information
    - Transformer Blocks: Self-attention + FFN
    - Layer Norm: Pre-norm or post-norm
    - Output Head: Projects to vocabulary for next-token prediction
```

### 1.2 Key Design Decisions

#### Model Size Variants
- **LOB-Small**: 6 layers, 512 dim, 8 heads (~25M params)
- **LOB-Base**: 12 layers, 768 dim, 12 heads (~110M params)
- **LOB-Large**: 24 layers, 1024 dim, 16 heads (~350M params)

#### Special Considerations for Financial Data
1. **Temporal Encoding**: Beyond position, encode time-of-day, day-of-week
2. **Multi-Scale Attention**: Different heads for different time horizons
3. **Causal Masking**: Strict for future prediction
4. **Numerical Stability**: Careful initialization for financial volatility

## 2. Model Implementation

### 2.1 Token & Position Embeddings

```python
class LOBEmbeddings(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.token_embeddings = nn.Embedding(config.vocab_size, config.d_model)
        self.position_embeddings = nn.Embedding(config.max_seq_length, config.d_model)

        # Optional: Time-aware embeddings
        self.time_of_day_embeddings = nn.Embedding(config.time_buckets, config.d_model)
        self.day_of_week_embeddings = nn.Embedding(7, config.d_model)

        self.norm = nn.LayerNorm(config.d_model)
        self.dropout = nn.Dropout(config.dropout)
```

### 2.2 Transformer Block

```python
class TransformerBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        # Multi-head self-attention
        self.attention = nn.MultiheadAttention(
            config.d_model,
            config.n_heads,
            dropout=config.dropout,
            batch_first=True
        )

        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(config.d_model, config.d_ff),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_ff, config.d_model),
            nn.Dropout(config.dropout)
        )

        # Layer norms
        self.norm1 = nn.LayerNorm(config.d_model)
        self.norm2 = nn.LayerNorm(config.d_model)
```

### 2.3 Output Head

```python
class PredictionHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.d_model, config.d_model)
        self.norm = nn.LayerNorm(config.d_model)
        self.output = nn.Linear(config.d_model, config.vocab_size)

        # Tie embeddings (optional)
        if config.tie_embeddings:
            self.output.weight = self.token_embeddings.weight
```

## 3. Training Infrastructure

### 3.1 PyTorch Lightning Module

```python
class LOBTransformerLightning(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.model = LOBTransformer(config)
        self.config = config

    def forward(self, input_ids, attention_mask=None):
        return self.model(input_ids, attention_mask)

    def training_step(self, batch, batch_idx):
        input_ids, labels = batch

        # Shift for next-token prediction
        logits = self(input_ids[:, :-1])
        loss = F.cross_entropy(
            logits.reshape(-1, self.config.vocab_size),
            labels[:, 1:].reshape(-1)
        )

        # Log metrics
        self.log('train_loss', loss)
        self.log('perplexity', torch.exp(loss))

        return loss

    def configure_optimizers(self):
        # AdamW with cosine schedule
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
            betas=(0.9, 0.95)
        )

        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=self.config.max_steps,
            eta_min=self.config.learning_rate * 0.1
        )

        return [optimizer], [scheduler]
```

### 3.2 Data Pipeline

```python
class LOBTokenDataModule(pl.LightningDataModule):
    def __init__(self, config):
        super().__init__()
        self.config = config

    def setup(self, stage=None):
        # Load and tokenize data
        inc_df = load_incremental_data(...)
        book_states = reconstruct_book_states(...)

        tokenizer = LOBTokenizer(...)
        tokens = tokenizer.tokenize_events(inc_df, book_states)

        # Create datasets
        self.train_dataset = TokenBookDataset(...)
        self.val_dataset = TokenBookDataset(...)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True
        )
```

### 3.3 Training Configuration

```yaml
# config.yaml
model:
  d_model: 768
  n_layers: 12
  n_heads: 12
  d_ff: 3072
  vocab_size: 10000
  max_seq_length: 1024
  dropout: 0.1
  tie_embeddings: true

training:
  batch_size: 32
  learning_rate: 6e-4
  weight_decay: 0.01
  max_epochs: 100
  gradient_clip_val: 1.0
  accumulate_grad_batches: 4

  # Mixed precision
  precision: 16

  # Checkpointing
  save_top_k: 3
  monitor: "val_loss"

  # Early stopping
  patience: 10
```

## 4. Advanced Features

### 4.1 Masked Language Modeling (MLM)

```python
def create_mlm_batch(tokens, mask_prob=0.15):
    """Create masked language modeling targets."""
    masked_tokens = tokens.clone()
    labels = tokens.clone()

    # Create random mask
    mask = torch.rand(tokens.shape) < mask_prob

    # 80% MASK, 10% random, 10% unchanged
    masked_tokens[mask] = MASK_TOKEN

    # Only compute loss on masked tokens
    labels[~mask] = -100

    return masked_tokens, labels
```

### 4.2 Generation/Inference

```python
@torch.no_grad()
def generate(model, prompt_tokens, max_length=100, temperature=1.0, top_k=50):
    """Generate LOB event sequences."""
    model.eval()
    generated = prompt_tokens.clone()

    for _ in range(max_length):
        # Get predictions
        logits = model(generated)[:, -1, :]

        # Apply temperature
        logits = logits / temperature

        # Top-k sampling
        if top_k > 0:
            indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
            logits[indices_to_remove] = -float('Inf')

        # Sample
        probs = F.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)

        # Append
        generated = torch.cat([generated, next_token], dim=-1)

    return generated
```

### 4.3 Market Regime Conditioning

```python
class ConditionalLOBTransformer(LOBTransformer):
    """Transformer conditioned on market regime."""

    def __init__(self, config):
        super().__init__(config)
        # Add regime embeddings
        self.regime_embeddings = nn.Embedding(
            config.n_regimes,
            config.d_model
        )

    def forward(self, input_ids, regime_id=None):
        # Add regime information to embeddings
        if regime_id is not None:
            regime_emb = self.regime_embeddings(regime_id)
            # Add to sequence or use as prefix
```

## 5. Evaluation & Monitoring

### 5.1 Metrics

```python
class LOBMetrics:
    """Custom metrics for LOB prediction."""

    def __init__(self):
        self.reset()

    def update(self, predictions, targets):
        # Token accuracy
        self.correct += (predictions == targets).sum()
        self.total += targets.numel()

        # Event-type accuracy
        pred_events = predictions // 1000  # Extract event type
        true_events = targets // 1000
        self.event_correct += (pred_events == true_events).sum()

        # Price direction accuracy
        # ... extract and compare price movements

    def compute(self):
        return {
            'token_accuracy': self.correct / self.total,
            'event_accuracy': self.event_correct / self.total,
            'perplexity': torch.exp(self.loss / self.total)
        }
```

### 5.2 Visualization

```python
def visualize_attention(model, tokens):
    """Visualize attention patterns."""
    # Get attention weights
    _, attention_weights = model(tokens, return_attention=True)

    # Plot heatmap
    plt.figure(figsize=(10, 10))
    sns.heatmap(attention_weights[0, 0].cpu().numpy())
    plt.xlabel("Keys")
    plt.ylabel("Queries")
    plt.title("Self-Attention Weights")
```

## 6. Production Deployment

### 6.1 Model Optimization

```python
# Quantization
quantized_model = torch.quantization.quantize_dynamic(
    model, {nn.Linear}, dtype=torch.qint8
)

# ONNX Export
torch.onnx.export(
    model,
    dummy_input,
    "lob_transformer.onnx",
    export_params=True,
    opset_version=11
)

# TorchScript
scripted_model = torch.jit.script(model)
scripted_model.save("lob_transformer.pt")
```

### 6.2 Serving Infrastructure

```python
class LOBTransformerServer:
    """REST API for model serving."""

    def __init__(self, model_path):
        self.model = load_model(model_path)
        self.tokenizer = load_tokenizer()

    async def predict_next_events(self,
                                  recent_events: List[Dict],
                                  n_predictions: int = 10):
        # Tokenize recent events
        tokens = self.tokenizer.encode(recent_events)

        # Generate predictions
        predictions = generate(
            self.model,
            tokens,
            max_length=n_predictions
        )

        # Decode back to events
        events = self.tokenizer.decode(predictions)

        return events
```

## 7. Training Pipeline

### 7.1 Full Training Script

```python
def train():
    # Config
    config = load_config("config.yaml")

    # Data
    data_module = LOBTokenDataModule(config)

    # Model
    model = LOBTransformerLightning(config)

    # Callbacks
    callbacks = [
        ModelCheckpoint(
            monitor='val_loss',
            save_top_k=3
        ),
        EarlyStopping(
            monitor='val_loss',
            patience=10
        ),
        LearningRateMonitor(),
        RichProgressBar()
    ]

    # Trainer
    trainer = pl.Trainer(
        max_epochs=config.max_epochs,
        accelerator='gpu',
        devices=1,
        precision=16,
        gradient_clip_val=1.0,
        callbacks=callbacks,
        logger=WandbLogger(project="lob-transformer")
    )

    # Train
    trainer.fit(model, data_module)

    # Test
    trainer.test(model, data_module)
```

## 8. Experimental Variations

### 8.1 Architecture Variants

1. **Sparse Attention**: For longer sequences
2. **Performer/Linformer**: Linear complexity attention
3. **Temporal Fusion Transformer**: Explicit time modeling
4. **Cross-Attention**: Between multiple assets

### 8.2 Training Objectives

1. **Next-N Token Prediction**: Predict multiple future tokens
2. **Contrastive Learning**: Learn market regime representations
3. **Multi-Task**: Predict events + price movements + volatility
4. **Adversarial Training**: Robust to market manipulation

### 8.3 Data Augmentation

1. **Time Warping**: Stretch/compress sequences
2. **Token Dropout**: Randomly drop events
3. **Mixup**: Interpolate between sequences
4. **Synthetic Events**: Generate realistic fake events

## Implementation Checklist

- [ ] Core transformer architecture
- [ ] Custom positional encoding for time
- [ ] Efficient attention mechanisms
- [ ] Training infrastructure with Lightning
- [ ] Data pipeline and tokenization
- [ ] Generation/sampling strategies
- [ ] Evaluation metrics
- [ ] Visualization tools
- [ ] Model optimization (quantization, pruning)
- [ ] Production serving setup
- [ ] A/B testing framework
- [ ] Monitoring and alerting

## References

1. Vaswani et al. (2017) - "Attention Is All You Need"
2. Radford et al. (2019) - "Language Models are Unsupervised Multitask Learners"
3. Zhang et al. (2019) - "DeepLOB: Deep Convolutional Neural Networks for Limit Order Books"
4. Wallbridge (2020) - "Transformers for Limit Order Book Modelling"
5. Brown et al. (2020) - "Language Models are Few-Shot Learners" (GPT-3)