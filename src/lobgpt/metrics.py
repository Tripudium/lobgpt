"""
Evaluation metrics for LOB Transformer models.
"""

from dataclasses import dataclass
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix

from lobgpt.tokenizer import EventType, LOBTokenizer, PriceBucket, Side


@dataclass
class LOBMetrics:
    """Container for LOB prediction metrics."""

    # Token-level metrics
    token_accuracy: float
    token_perplexity: float

    # Event-level metrics
    event_type_accuracy: float
    side_accuracy: float
    price_bucket_accuracy: float
    size_bucket_accuracy: float

    # Financial metrics
    direction_accuracy: float  # Price movement direction
    mid_price_mae: float       # Mid-price prediction error
    spread_mae: float          # Spread prediction error

    # Sequence-level metrics
    sequence_similarity: float  # Edit distance based
    event_rate_error: float     # Event frequency prediction

    # Distribution metrics
    token_distribution_kl: float    # KL divergence from true distribution
    event_timing_mae: float         # Time between events error


class LOBEvaluator:
    """Comprehensive evaluation for LOB Transformer models."""

    def __init__(self, tokenizer: LOBTokenizer):
        self.tokenizer = tokenizer
        self.reset()

    def reset(self):
        """Reset accumulated metrics."""
        self.predictions = []
        self.targets = []
        self.sequences_pred = []
        self.sequences_true = []

        # Accumulated counts
        self.total_tokens = 0
        self.correct_tokens = 0
        self.total_loss = 0.0

        # Event-level accumulators
        self.event_correct = {"type": 0, "side": 0, "price": 0, "size": 0}
        self.event_total = {"type": 0, "side": 0, "price": 0, "size": 0}

        # Financial metrics
        self.direction_correct = 0
        self.direction_total = 0
        self.mid_price_errors = []
        self.spread_errors = []

    def update_batch(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        loss: Optional[torch.Tensor] = None
    ):
        """Update metrics with a batch of predictions."""
        batch_size, seq_len = predictions.shape

        # Flatten for token-level metrics
        pred_flat = predictions.view(-1)
        target_flat = targets.view(-1)

        # Filter out padding tokens (-100)
        valid_mask = target_flat != -100
        valid_pred = pred_flat[valid_mask]
        valid_target = target_flat[valid_mask]

        if len(valid_pred) == 0:
            return

        # Token accuracy
        correct = (valid_pred == valid_target).sum().item()
        self.correct_tokens += correct
        self.total_tokens += len(valid_pred)

        # Loss
        if loss is not None:
            self.total_loss += loss.item() * len(valid_pred)

        # Store for detailed analysis
        self.predictions.extend(valid_pred.cpu().numpy())
        self.targets.extend(valid_target.cpu().numpy())

        # Event-level analysis
        self._update_event_metrics(valid_pred.cpu().numpy(), valid_target.cpu().numpy())

        # Sequence-level analysis
        for i in range(batch_size):
            pred_seq = predictions[i][targets[i] != -100]
            true_seq = targets[i][targets[i] != -100]

            if len(pred_seq) > 0 and len(true_seq) > 0:
                self.sequences_pred.append(pred_seq.cpu().numpy())
                self.sequences_true.append(true_seq.cpu().numpy())

    def _update_event_metrics(self, predictions: np.ndarray, targets: np.ndarray):
        """Update event-level metrics."""
        try:
            for pred_token, true_token in zip(predictions, targets):
                # Decode tokens to extract event components
                pred_event = self.tokenizer.decode_token(pred_token)
                true_event = self.tokenizer.decode_token(true_token)

                if pred_event is None or true_event is None:
                    continue

                # Event type accuracy
                if pred_event.event_type == true_event.event_type:
                    self.event_correct["type"] += 1
                self.event_total["type"] += 1

                # Side accuracy
                if pred_event.side == true_event.side:
                    self.event_correct["side"] += 1
                self.event_total["side"] += 1

                # Price level accuracy
                if pred_event.price_bucket == true_event.price_bucket:
                    self.event_correct["price"] += 1
                self.event_total["price"] += 1

                # Size bucket accuracy
                if pred_event.size_bucket == true_event.size_bucket:
                    self.event_correct["size"] += 1
                self.event_total["size"] += 1

                # Direction accuracy based on price bucket sign
                if self._price_bucket_sign(pred_event.price_bucket) == self._price_bucket_sign(true_event.price_bucket):
                    self.direction_correct += 1
                self.direction_total += 1

        except Exception as e:
            # Gracefully handle decoding errors
            print(f"Warning: Error in event metrics update: {e}")

    def compute_metrics(self) -> LOBMetrics:
        """Compute final metrics."""
        if self.total_tokens == 0:
            return self._empty_metrics()

        # Token-level metrics
        token_accuracy = self.correct_tokens / self.total_tokens
        token_perplexity = np.exp(self.total_loss / self.total_tokens) if self.total_loss > 0 else float('inf')

        # Event-level metrics
        event_type_accuracy = self.event_correct["type"] / max(self.event_total["type"], 1)
        side_accuracy = self.event_correct["side"] / max(self.event_total["side"], 1)
        price_bucket_accuracy = self.event_correct["price"] / max(self.event_total["price"], 1)
        size_bucket_accuracy = self.event_correct["size"] / max(self.event_total["size"], 1)

        # Financial metrics
        direction_accuracy = self.direction_correct / max(self.direction_total, 1)
        mid_price_mae = np.mean(self.mid_price_errors) if self.mid_price_errors else 0.0
        spread_mae = np.mean(self.spread_errors) if self.spread_errors else 0.0

        # Sequence-level metrics
        sequence_similarity = self._compute_sequence_similarity()
        event_rate_error = self._compute_event_rate_error()

        # Distribution metrics
        token_distribution_kl = self._compute_token_distribution_kl()
        event_timing_mae = 0.0  # TODO: Implement if timing data available

        return LOBMetrics(
            token_accuracy=token_accuracy,
            token_perplexity=token_perplexity,
            event_type_accuracy=event_type_accuracy,
            side_accuracy=side_accuracy,
            price_bucket_accuracy=price_bucket_accuracy,
            size_bucket_accuracy=size_bucket_accuracy,
            direction_accuracy=direction_accuracy,
            mid_price_mae=mid_price_mae,
            spread_mae=spread_mae,
            sequence_similarity=sequence_similarity,
            event_rate_error=event_rate_error,
            token_distribution_kl=token_distribution_kl,
            event_timing_mae=event_timing_mae
        )

    def _empty_metrics(self) -> LOBMetrics:
        """Return empty metrics when no data available."""
        return LOBMetrics(
            token_accuracy=0.0,
            token_perplexity=float('inf'),
            event_type_accuracy=0.0,
            side_accuracy=0.0,
            price_bucket_accuracy=0.0,
            size_bucket_accuracy=0.0,
            direction_accuracy=0.0,
            mid_price_mae=0.0,
            spread_mae=0.0,
            sequence_similarity=0.0,
            event_rate_error=0.0,
            token_distribution_kl=0.0,
            event_timing_mae=0.0
        )

    def _compute_sequence_similarity(self) -> float:
        """Compute average sequence similarity using edit distance."""
        if not self.sequences_pred or not self.sequences_true:
            return 0.0

        similarities = []
        for pred_seq, true_seq in zip(self.sequences_pred, self.sequences_true):
            # Simple normalized edit distance
            max_len = max(len(pred_seq), len(true_seq))
            if max_len == 0:
                similarities.append(1.0)
                continue

            # Count matching positions
            min_len = min(len(pred_seq), len(true_seq))
            matches = sum(1 for i in range(min_len) if pred_seq[i] == true_seq[i])
            similarity = matches / max_len
            similarities.append(similarity)

        return np.mean(similarities)

    def _compute_event_rate_error(self) -> float:
        """Compute error in predicted event rates."""
        if not self.sequences_pred or not self.sequences_true:
            return 0.0

        pred_rates = [len(seq) for seq in self.sequences_pred]
        true_rates = [len(seq) for seq in self.sequences_true]

        return np.mean(np.abs(np.array(pred_rates) - np.array(true_rates)))

    @staticmethod
    def _price_bucket_sign(bucket: PriceBucket) -> int:
        if bucket == PriceBucket.AT_REFERENCE:
            return 0
        return -1 if bucket.value < PriceBucket.AT_REFERENCE.value else 1

    def _compute_token_distribution_kl(self) -> float:
        """Compute KL divergence between predicted and true token distributions."""
        if not self.predictions or not self.targets:
            return 0.0

        vocab_size = len(self.tokenizer.vocab)

        # Count occurrences
        pred_counts = np.bincount(self.predictions, minlength=vocab_size)
        true_counts = np.bincount(self.targets, minlength=vocab_size)

        # Convert to probabilities (add smoothing)
        pred_probs = (pred_counts + 1) / (np.sum(pred_counts) + vocab_size)
        true_probs = (true_counts + 1) / (np.sum(true_counts) + vocab_size)

        # KL divergence
        kl_div = np.sum(true_probs * np.log(true_probs / pred_probs))
        return kl_div

    def plot_confusion_matrices(self, save_path: Optional[str] = None):
        """Plot confusion matrices for different event components."""
        if not self.predictions or not self.targets:
            print("No data available for confusion matrices")
            return

        fig, axes = plt.subplots(2, 2, figsize=(15, 12))

        # Extract event components
        pred_events = []
        true_events = []

        for pred_token, true_token in zip(self.predictions, self.targets):
            try:
                pred_event = self.tokenizer.decode_token(pred_token)
                true_event = self.tokenizer.decode_token(true_token)

                if pred_event is not None and true_event is not None:
                    pred_events.append(pred_event)
                    true_events.append(true_event)
            except:
                continue

        if not pred_events:
            print("No valid events for confusion matrices")
            return

        # Event type confusion matrix
        event_types = [e.event_type.value for e in true_events]
        pred_types = [e.event_type.value for e in pred_events]

        cm = confusion_matrix(event_types, pred_types, labels=[et.value for et in EventType])
        sns.heatmap(cm, annot=True, fmt='d', ax=axes[0,0],
                   xticklabels=[et.value for et in EventType],
                   yticklabels=[et.value for et in EventType])
        axes[0,0].set_title('Event Type Confusion Matrix')
        axes[0,0].set_xlabel('Predicted')
        axes[0,0].set_ylabel('True')

        # Side confusion matrix
        sides = [e.side.value for e in true_events]
        pred_sides = [e.side.value for e in pred_events]

        cm = confusion_matrix(sides, pred_sides, labels=[s.value for s in Side])
        sns.heatmap(cm, annot=True, fmt='d', ax=axes[0,1],
                   xticklabels=[s.value for s in Side],
                   yticklabels=[s.value for s in Side])
        axes[0,1].set_title('Side Confusion Matrix')
        axes[0,1].set_xlabel('Predicted')
        axes[0,1].set_ylabel('True')

        # Price bucket confusion matrix
        price_levels = [e.price_bucket.value for e in true_events]
        pred_price_levels = [e.price_bucket.value for e in pred_events]

        cm = confusion_matrix(price_levels, pred_price_levels, labels=[pl.value for pl in PriceBucket])
        sns.heatmap(cm, annot=True, fmt='d', ax=axes[1,0],
                   xticklabels=[pl.value for pl in PriceBucket],
                   yticklabels=[pl.value for pl in PriceBucket])
        axes[1,0].set_title('Price Bucket Confusion Matrix')
        axes[1,0].set_xlabel('Predicted')
        axes[1,0].set_ylabel('True')

        # Token distribution comparison
        vocab_size = len(self.tokenizer.vocab)
        pred_counts = np.bincount(self.predictions, minlength=vocab_size)
        true_counts = np.bincount(self.targets, minlength=vocab_size)

        # Show top 20 most frequent tokens
        top_tokens = np.argsort(true_counts)[-20:]

        x = np.arange(len(top_tokens))
        width = 0.35

        axes[1,1].bar(x - width/2, true_counts[top_tokens], width, label='True', alpha=0.7)
        axes[1,1].bar(x + width/2, pred_counts[top_tokens], width, label='Predicted', alpha=0.7)
        axes[1,1].set_xlabel('Token ID')
        axes[1,1].set_ylabel('Count')
        axes[1,1].set_title('Token Distribution (Top 20)')
        axes[1,1].set_xticks(x)
        axes[1,1].set_xticklabels(top_tokens)
        axes[1,1].legend()

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.show()

    def generate_report(self) -> str:
        """Generate a comprehensive evaluation report."""
        metrics = self.compute_metrics()

        report = f"""
LOB Transformer Evaluation Report
================================

Token-Level Metrics:
- Accuracy: {metrics.token_accuracy:.4f}
- Perplexity: {metrics.token_perplexity:.4f}

Event-Level Metrics:
- Event Type Accuracy: {metrics.event_type_accuracy:.4f}
- Side Accuracy: {metrics.side_accuracy:.4f}
- Price Bucket Accuracy: {metrics.price_bucket_accuracy:.4f}
- Size Bucket Accuracy: {metrics.size_bucket_accuracy:.4f}

Financial Metrics:
- Direction Accuracy: {metrics.direction_accuracy:.4f}
- Mid-Price MAE: {metrics.mid_price_mae:.4f}
- Spread MAE: {metrics.spread_mae:.4f}

Sequence-Level Metrics:
- Sequence Similarity: {metrics.sequence_similarity:.4f}
- Event Rate Error: {metrics.event_rate_error:.4f}

Distribution Metrics:
- Token Distribution KL: {metrics.token_distribution_kl:.4f}
- Event Timing MAE: {metrics.event_timing_mae:.4f}

Dataset Statistics:
- Total Tokens: {self.total_tokens:,}
- Total Sequences: {len(self.sequences_true):,}
- Vocabulary Coverage: {len(set(self.predictions)) / len(self.tokenizer.vocab):.4f}
"""
        return report


def evaluate_model(
    model,
    dataloader,
    tokenizer: LOBTokenizer,
    device: str = "cuda"
) -> LOBMetrics:
    """Evaluate a model on a dataset."""
    model.eval()
    evaluator = LOBEvaluator(tokenizer)

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)

            # Shift for next-token prediction
            if input_ids.size(1) <= 1:
                continue

            inputs = input_ids[:, :-1]
            targets = input_ids[:, 1:]

            # Forward pass
            logits, _ = model(inputs)

            # Calculate loss
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                targets.reshape(-1),
                ignore_index=-100
            )

            # Get predictions
            predictions = torch.argmax(logits, dim=-1)

            # Update evaluator
            evaluator.update_batch(predictions, targets, loss)

    return evaluator.compute_metrics()
